# Standard
from enum import StrEnum
from pathlib import Path
import gc
import json
import os
import typing as t

# Third Party
from accelerate import Accelerator
from lm_eval.evaluator import simple_evaluate
from torch import cuda
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Local
from .evaluator import Evaluator


class ParsedScores(t.TypedDict):
    """
    Just an ordinary dict that contains both the overall score as well as per-subtask scores.
    """

    score: float
    subtasks: t.NotRequired[t.Dict[str, float]]


class LeaderboardV2EvalResult(t.TypedDict):
    overall_score: float
    leaderboard_gpqa: t.NotRequired[ParsedScores]
    leaderboard_ifeval: t.NotRequired[ParsedScores]
    leaderboard_bbh: t.NotRequired[ParsedScores]
    leaderboard_mmlu_pro: t.NotRequired[ParsedScores]
    leaderboard_musr: t.NotRequired[ParsedScores]
    leaderboard_math_hard: t.NotRequired[ParsedScores]


class LeaderboardV2Tasks(StrEnum):
    MATH_HARD = "leaderboard_math_hard"
    IFEVAL = "leaderboard_ifeval"
    MMLU_PRO = "leaderboard_mmlu_pro"
    GPQA = "leaderboard_gpqa"
    MUSR = "leaderboard_musr"
    BBH = "leaderboard_bbh"


class LeaderboardArgs(t.TypedDict):
    model_path: str
    num_gpus: int
    tasks: t.List[str]


class TaskGrouping(t.TypedDict):
    """
    Class used to group the tasks by their optimal runtime.
    """

    huggingface: t.List[str]
    vllm: t.List[str]


# generative tasks go here
LEADERBOARD_V2_GENERATIVE_TASKS = [
    LeaderboardV2Tasks.MATH_HARD.value,
    LeaderboardV2Tasks.IFEVAL.value,
]

# all the MCQ-style tasks in leaderboard v2
LEADERBOARD_V2_MCQ_TASKS = [
    LeaderboardV2Tasks.BBH.value,
    LeaderboardV2Tasks.MUSR.value,
    LeaderboardV2Tasks.GPQA.value,
    LeaderboardV2Tasks.MMLU_PRO.value,
]


def evaluate_with_vllm(args: LeaderboardArgs) -> t.Dict[str, t.Any]:
    results = simple_evaluate(
        tasks=args["tasks"],
        model="vllm",
        model_args={
            "pretrained": args["model_path"],
            "dtype": "float16",
            "data_parallel_size": args["num_gpus"],
            "gpu_memory_utilization": 0.8,
            "max_model_len": 32768,
            "disable_custom_all_reduce": True,
            "enforce_eager": False,
        },
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        batch_size="auto",
        confirm_run_unsafe_code=True,
    )
    return results


def worker(rank, world_size, args: LeaderboardArgs, result_queue: mp.Queue):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"  # hopefully nobody else is using this port

    accelerator = Accelerator()
    device = accelerator.device
    assert device.type == "cuda", f"device is not a cuda device: {device}"

    results = simple_evaluate(
        model="hf",
        model_args={
            "pretrained": args["model_path"],
            "dtype": "float16",
            "trust_remote_code": True,
        },
        tasks=args["tasks"],
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        batch_size="auto",
        device=f"cuda:{device.index}",
        cache_requests=True,
        confirm_run_unsafe_code=True,
    )

    result_queue.put((rank, results))

    # clear torch memory
    gc.collect()
    torch.cuda.empty_cache()

    dist.destroy_process_group()


def evaluate_with_hf(args: LeaderboardArgs) -> t.Dict[str, t.Any]:
    # we need to use torch.multiprocessing to run each task in a separate process,
    # and then combine the results
    # Third Party
    import torch.multiprocessing as mp

    num_processes = args["num_gpus"]

    # Create the context and queue within the same context
    mp_ctx = mp.get_context("spawn")  # Explicitly use spawn context
    result_queue = mp_ctx.Queue()

    # Use the same context's Process
    processes = []
    for rank in range(num_processes):
        p = mp_ctx.Process(
            target=worker, args=(rank, num_processes, args, result_queue)
        )
        p.start()
        processes.append(p)

    results = {}
    for _ in range(num_processes):
        rank, result = result_queue.get()
        results[rank] = result

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # extract the result which is not None
    assert len([res for res in results.values() if res is not None]) == 1, (
        "we expect exactly 1 process to return a results dict properly"
    )
    results_dict = [res for res in results.values() if res is not None][0]
    return results_dict


def get_score_by_metric(score_dict: t.Dict[str, t.Any], metric: str) -> t.Any:
    extracted_value = None
    for key, value in score_dict.items():
        if "," not in key:
            continue
        parsed_metric, _ = key.split(",")
        if parsed_metric == metric:
            extracted_value = value
            break

    if not extracted_value:
        if alias := score_dict.get("alias", None):
            error_msg = (
                f"Failed to find a metric matching '{metric}' for task '{alias}'."
            )
        else:
            error_msg = f"Failed to find a metric matching '{metric}'."
        error_msg += f"\nAvailable fields: {list(score_dict.keys())}"
        raise ValueError(error_msg)
    return extracted_value


def parse_multitask_results(
    result_dict: t.Dict[str, t.Any], benchmark: str, metric: str
) -> ParsedScores:
    """
    Parse out the results of the given benchmark into a single floating-point score that can be consumed.

    The rules are like this: for a multi-task benchmark, the entry matching the exact benchmark name contains nothing.
    Everything else is a subtask and contains a score.

    The end result is an unweighted average of all the subtasks, as well as a per-subtask breakdown.
    """
    parsed_scores: ParsedScores = {"score": 0.0, "subtasks": {}}
    subtask_scores = {}
    target_subtasks = result_dict["group_subtasks"].get(benchmark)
    if not target_subtasks:
        raise ValueError(f"Couldnt find '{benchmark}' in the group_subtasks section")

    for subtask in target_subtasks:
        # pull out the score
        subtask_results = result_dict["results"][subtask]
        subtask_score = get_score_by_metric(subtask_results, metric)
        subtask_scores[subtask] = subtask_score

    # exit early, base case
    if not subtask_scores:
        return parsed_scores

    parsed_scores["score"] = sum(subtask_scores.values()) / len(subtask_scores)
    parsed_scores["subtasks"] = subtask_scores
    return parsed_scores


def parse_bbh(result_dict: t.Dict[str, t.Any]) -> ParsedScores:
    """
    Parses out the bbh scores from the result dict
    """
    parsed_scores = parse_multitask_results(
        result_dict, LeaderboardV2Tasks.BBH.value, "acc_norm"
    )
    assert len(parsed_scores["subtasks"]) == 24, (
        "there should be 24 subtasks of bbh run"
    )
    return parsed_scores


def parse_mmlu_pro(result_dict: t.Dict[str, t.Any]) -> ParsedScores:
    """
    Parses out the mmlu_pro scores from the result dict
    """
    mmlu_pro_results = result_dict["results"].get("leaderboard_mmlu_pro", None)
    return {
        "score": get_score_by_metric(mmlu_pro_results, "acc"),
    }


def parse_ifeval(result_dict: t.Dict[str, t.Any]) -> ParsedScores:
    """
    Parses out the ifeval scores from the result dict.
    In particular, we only compute the average between the strict prompts
    """
    ifeval_results = result_dict["results"].get("leaderboard_ifeval", None)
    if not ifeval_results:
        raise ValueError(
            f"Failed to find `leaderboard_ifeval` in scores. Available results: {list(result_dict.keys())}"
        )

    # The format of ifeval looks like this:
    # {
    #   "alias": "leaderboard_ifeval",
    #   "prompt_level_strict_acc,none": 0.6876155268022182,
    #   "prompt_level_strict_acc_stderr,none": 0.019944386293758908,
    #   "inst_level_strict_acc,none": 0.7745803357314148,
    #   "inst_level_strict_acc_stderr,none": "N/A",
    #   "prompt_level_loose_acc,none": 0.722735674676525,
    #   "prompt_level_loose_acc_stderr,none": 0.019263706963479364,
    #   "inst_level_loose_acc,none": 0.8033573141486811,
    #   "inst_level_loose_acc_stderr,none": "N/A"
    # }
    #

    target_metrics = {"prompt_level_strict_acc", "inst_level_strict_acc"}
    scores = []

    for key, value in ifeval_results.items():
        if "," not in key or "stderr" in key:
            continue

        metric, _ = key.split(",")
        if metric in target_metrics:
            scores.append(value)
            target_metrics.remove(metric)

    assert len(scores) == 2, (
        f"there should only be 2 values extracted in ifeval, got: {len(scores)}"
    )
    return {
        "score": sum(scores) / 2,
    }


def parse_musr(result_dict: t.Dict[str, t.Any]) -> ParsedScores:
    """
    Parses out the musr scores from the result dict
    """
    parsed_scores = parse_multitask_results(
        result_dict, LeaderboardV2Tasks.MUSR.value, "acc_norm"
    )
    assert len(parsed_scores["subtasks"]) == 3
    return parsed_scores


def parse_gpqa(result_dict: t.Dict[str, t.Any]) -> ParsedScores:
    """
    Parses out the gpqa scores from the result dict
    """
    parsed_scores = parse_multitask_results(
        result_dict, LeaderboardV2Tasks.GPQA.value, "acc_norm"
    )
    assert len(parsed_scores["subtasks"]) == 3, (
        f"Expected 3 gpqa scores, got {len(parsed_scores['subtasks'])}"
    )
    return parsed_scores


def parse_math_hard(result_dict: t.Dict[str, t.Any]) -> ParsedScores:
    """
    Parses out the math_hard scores from the result dict. Result is an unweighted average.
    """
    parsed_scores = parse_multitask_results(
        result_dict, LeaderboardV2Tasks.MATH_HARD.value, "exact_match"
    )
    assert len(parsed_scores["subtasks"]) == 7, (
        f"leaderboard_math_hard should have 7 subtasks, found: {len(parsed_scores['subtasks'])}"
    )
    return parsed_scores


def get_parser(subtask: str) -> t.Callable[[t.Dict[str, t.Any]], ParsedScores]:
    parser_map = {
        LeaderboardV2Tasks.BBH.value: parse_bbh,
        LeaderboardV2Tasks.GPQA.value: parse_gpqa,
        LeaderboardV2Tasks.IFEVAL.value: parse_ifeval,
        LeaderboardV2Tasks.MATH_HARD.value: parse_math_hard,
        LeaderboardV2Tasks.MMLU_PRO.value: parse_mmlu_pro,
        LeaderboardV2Tasks.MUSR.value: parse_musr,
    }
    return parser_map[
        LeaderboardV2Tasks(subtask)
    ]  # this will either parse and map into the correct section, or error


def build_leaderboard_v2_result(
    parsed_scores: t.Dict[str, ParsedScores],
) -> LeaderboardV2EvalResult:
    """
    Build the leaderboard v2 result from the parsed scores.
    """
    # now let's build the overall score
    leaderboard_result: LeaderboardV2EvalResult = {
        "overall_score": calculate_overall_leaderboard_score(parsed_scores),
    }

    # explicitly set the score for each subtask in order to satisfy mypy
    if "leaderboard_bbh" in parsed_scores:
        leaderboard_result["leaderboard_bbh"] = parsed_scores["leaderboard_bbh"]
    if "leaderboard_gpqa" in parsed_scores:
        leaderboard_result["leaderboard_gpqa"] = parsed_scores["leaderboard_gpqa"]
    if "leaderboard_ifeval" in parsed_scores:
        leaderboard_result["leaderboard_ifeval"] = parsed_scores["leaderboard_ifeval"]
    if "leaderboard_math_hard" in parsed_scores:
        leaderboard_result["leaderboard_math_hard"] = parsed_scores[
            "leaderboard_math_hard"
        ]
    if "leaderboard_mmlu_pro" in parsed_scores:
        leaderboard_result["leaderboard_mmlu_pro"] = parsed_scores[
            "leaderboard_mmlu_pro"
        ]
    if "leaderboard_musr" in parsed_scores:
        leaderboard_result["leaderboard_musr"] = parsed_scores["leaderboard_musr"]

    return leaderboard_result


def get_scores_from_result_dicts(
    *result_dicts: t.Dict[str, t.Any],
) -> LeaderboardV2EvalResult:
    """
    Parse out the scores of all the subtasks of leaderboard and return.
    """
    parsed_scores: t.Dict[str, ParsedScores] = {}
    for result_dict in result_dicts:
        benchmarks_we_got = set(result_dict["results"].keys())
        benchmarks_we_care_about = set(
            LEADERBOARD_V2_GENERATIVE_TASKS + LEADERBOARD_V2_MCQ_TASKS
        )
        benchmarks_to_parse = benchmarks_we_got & benchmarks_we_care_about

        # this is just a sanity check step
        benchmarks_already_covered = set(parsed_scores.keys())
        overlapping_benchmarks = benchmarks_already_covered & benchmarks_to_parse
        assert len(benchmarks_already_covered & benchmarks_to_parse) == 0, (
            f"expected no overlapping benchmarks but found the following to overlap: {list(overlapping_benchmarks)}"
        )

        # now actually add them
        for benchmark in benchmarks_to_parse:
            parse_benchmark_fn = get_parser(benchmark)
            parsed_scores[benchmark] = parse_benchmark_fn(result_dict)

    return build_leaderboard_v2_result(parsed_scores)


def validate_output_path(output_file: str) -> None:
    """
    Validates that we can write to the specified output path.
    Creates parent directories if they don't exist.

    Args:
        output_file: Path to the desired output file

    Raises:
        ValueError: If the path is invalid or we don't have proper permissions
    """
    if not output_file:
        raise ValueError("Output file path cannot be empty")

    # Convert to Path object for easier handling
    output_path = Path(output_file)

    try:
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Test if we can write to the file by opening it in append mode
        # We don't actually write anything
        output_path.open("a").close()

    except PermissionError:
        raise ValueError(f"Permission denied: Cannot write to {output_file}")
    except OSError as e:
        raise ValueError(f"Invalid output path: {output_file}. Error: {str(e)}")


def validate_leaderboard_v2_tasks(tasks: t.List[str]):
    invalid_tasks = set(tasks) - set(
        LEADERBOARD_V2_GENERATIVE_TASKS + LEADERBOARD_V2_MCQ_TASKS
    )
    if invalid_tasks:
        raise ValueError(
            f"the following tasks were provided but are not valid leaderboard tasks: {list(invalid_tasks)}.\n"
            f"Supported tasks are: {LEADERBOARD_V2_GENERATIVE_TASKS + LEADERBOARD_V2_MCQ_TASKS}"
        )


def get_task_groupings(tasks: t.List[str]) -> TaskGrouping:
    """
    Given a list of tasks, bucket them per their optimal runtime.
    """
    task_grouping: TaskGrouping = {
        "vllm": [task for task in tasks if task in LEADERBOARD_V2_GENERATIVE_TASKS],
        "huggingface": [task for task in tasks if task in LEADERBOARD_V2_MCQ_TASKS],
    }
    overlapping_tasks = set(task_grouping["vllm"]) & set(task_grouping["huggingface"])
    assert not overlapping_tasks
    return task_grouping


def calculate_overall_leaderboard_score(results: t.Dict[str, ParsedScores]) -> float:
    """
    Given a dict with leaderboard metrics, compute the average of the scores
    """
    all_scores = [res["score"] for res in results.values() if "score" in res]

    return sum(all_scores) / len(all_scores) if len(all_scores) > 0 else 0.0


# here we assume that we can
class LeaderboardV2Evaluator(Evaluator):
    """
    Evaluator for Open Leaderboard v2.
    """

    name = "leaderboard_v2"

    def __init__(
        self,
        model_path: str,
        tasks: t.Optional[t.List[str]] = None,
        num_gpus: t.Optional[int] = None,
        output_file: t.Optional[str] = None,
    ):
        self.model_path = model_path
        if not cuda.is_available():
            raise ValueError(
                "Running without CUDA is currently unsupported. Contributions are welcome."
            )

        # set whatever we need here
        self.num_gpus = num_gpus
        self.tasks = tasks

        # validate output file
        self.output_file = output_file
        self._results: t.Optional[LeaderboardV2EvalResult] = None
        self._lm_eval_results: t.List[
            t.Dict[str, t.Any]
        ] = []  # TODO: make it merge everything back into a single result

    @property
    def results(self) -> t.Optional[LeaderboardV2EvalResult]:
        """
        Returns the results of the most reccent leaderboard evaluation.

        Returns:
            LeaderboardV2EvalResult: A dict containing the overall leaderboard score and the breakdown per subtask.
        """
        return self._results

    @property
    def lm_eval_results(self) -> t.List[t.Dict[str, t.Any]]:
        """
        Returns the results of the most recent leaderboard evaluation.

        Returns:
            t.List[t.Dict[str, t.Any]]: A list of dicts containing the results of the most recent leaderboard evaluation.
        """
        return self._lm_eval_results

    def save_to_file(self, output_file: t.Optional[str] = None) -> None:
        """
        Saves the results to a file.

        Args:
            output_file: The path to the file to save the results to.
        """
        if output_file is None:
            output_file = self.output_file
        if output_file is None:
            raise ValueError("Output file path cannot be empty")

        # create the directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(self._results, f, indent=2)

    def run(
        self,
        model_path: t.Optional[str] = None,
        tasks: t.Optional[t.List[str]] = None,
        num_gpus: t.Optional[int] = None,
        output_file: t.Optional[str] = None,
    ) -> LeaderboardV2EvalResult:
        """
        Run the Open LLM Leaderboard v2 evaluation.

        This function will use both HF transformers and inline vLLM to run the evaluation.
        It will then parse the results and save them to a file.

        Args:
            model_path: The path to the model to evaluate.
            tasks: The tasks to evaluate.
            num_gpus: The number of GPUs to use.
            output_file: The path to the file to save the results to.

        Returns:
            LeaderboardV2EvalResult: A dict containing the overall leaderboard score and the breakdown per subtask.
        """
        model_path = self.model_path if model_path is None else model_path
        tasks = self.tasks if not tasks else tasks
        num_gpus = self.num_gpus if not num_gpus else num_gpus
        output_file = self.output_file if not output_file else output_file

        if not tasks:
            tasks = LEADERBOARD_V2_MCQ_TASKS + LEADERBOARD_V2_GENERATIVE_TASKS

        # validation logic
        # no need to validate model path -- the inference libraries will either be able to
        # load it, or they won't

        validate_leaderboard_v2_tasks(tasks)
        if not num_gpus:
            num_gpus = cuda.device_count()
        if num_gpus <= 0 or num_gpus > cuda.device_count():
            raise ValueError(
                f"invalid value for num_gpus, must be between 1 and {cuda.device_count()}; got: {num_gpus}"
            )
        if output_file:
            validate_output_path(output_file)

        # now we just have to run the task group in their most appropriate runtime
        # this is important because certain tasks like MCQ are better-suited to be
        # excuted in raw transformers due to the lack of KV-Cache overhead,
        # whereas generative tasks are better suited for vLLM due to their need for
        # accessing previous tokens

        grouped_tasks = get_task_groupings(tasks)
        self._lm_eval_results = []
        vllm_results, hf_results = None, None
        if vllm_tasks := grouped_tasks["vllm"]:
            args_vllm: LeaderboardArgs = {
                "model_path": model_path,
                "num_gpus": num_gpus,
                "tasks": vllm_tasks,
            }
            vllm_results = evaluate_with_vllm(args_vllm)
            self._lm_eval_results.append(vllm_results)
        if hf_tasks := grouped_tasks["huggingface"]:
            args_hf: LeaderboardArgs = {
                "model_path": model_path,
                "num_gpus": num_gpus,
                "tasks": hf_tasks,
            }
            hf_results = evaluate_with_hf(args_hf)
            self._lm_eval_results.append(hf_results)

        # convert the output of lm-eval into something that's already parsed
        results: LeaderboardV2EvalResult = get_scores_from_result_dicts(
            *self._lm_eval_results
        )

        self._results = results
        if output_file:
            self.save_to_file(output_file)
        return results
