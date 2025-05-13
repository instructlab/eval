# Standard
from collections import defaultdict
import json
import os
import typing as t

# Third Party
from lm_eval.evaluator import simple_evaluate
from torch import cuda

# Local
from .evaluator import Evaluator


class LongBenchResult(t.TypedDict):
    """Dict containing averages for each task type and language"""

    overall_score: float
    en_multidoc: t.NotRequired[float]
    zh_multidoc: t.NotRequired[float]
    en_singledoc: t.NotRequired[float]
    zh_singledoc: t.NotRequired[float]
    en_summ: t.NotRequired[float]
    zh_summ: t.NotRequired[float]
    en_fewshot: t.NotRequired[float]
    zh_fewshot: t.NotRequired[float]
    en_synthetic: t.NotRequired[float]
    zh_synthetic: t.NotRequired[float]
    code_avg: t.NotRequired[float]


# Define task categories
TASK_CATEGORIES = {
    "en_multidoc": ["longbench_hotpotqa", "longbench_2wikimqa", "longbench_musique"],
    "zh_multidoc": ["longbench_dureader"],
    "en_singledoc": [
        "longbench_multifieldqa_en",
        "longbench_narrativeqa",
        "longbench_qasper",
    ],
    "zh_singledoc": ["longbench_multifieldqa_zh"],
    "en_summ": ["longbench_gov_report", "longbench_qmsum", "longbench_multi_news"],
    "zh_summ": ["longbench_vcsum"],
    "en_fewshot": ["longbench_triviaqa", "longbench_samsum", "longbench_trec"],
    "zh_fewshot": ["longbench_lsht"],
    "en_synthetic": ["longbench_passage_retrieval_en", "longbench_passage_count"],
    "zh_synthetic": ["longbench_passage_retrieval_zh"],
    "code_avg": ["longbench_lcc", "longbench_repobench-p"],
}

# Flatten the categories to get all tasks
ALL_LONGBENCH_TASKS = []
for task in TASK_CATEGORIES.values():
    ALL_LONGBENCH_TASKS.extend(task)

# Task to metric mapping
TASK_METRICS = {
    "longbench_hotpotqa": "qa_f1_score",
    "longbench_2wikimqa": "qa_f1_score",
    "longbench_musique": "qa_f1_score",
    "longbench_dureader": "rouge_zh_score",
    "longbench_multifieldqa_en": "qa_f1_score",
    "longbench_narrativeqa": "qa_f1_score",
    "longbench_qasper": "qa_f1_score",
    "longbench_multifieldqa_zh": "qa_f1_zh_score",
    "longbench_gov_report": "rouge_score",
    "longbench_qmsum": "rouge_score",
    "longbench_multi_news": "rouge_score",
    "longbench_vcsum": "rouge_zh_score",
    "longbench_triviaqa": "qa_f1_score",
    "longbench_samsum": "rouge_score",
    "longbench_trec": "classification_score",
    "longbench_lsht": "classification_score",
    "longbench_passage_retrieval_en": "retrieval_score",
    "longbench_passage_count": "count_score",
    "longbench_passage_retrieval_zh": "retrieval_zh_score",
    "longbench_lcc": "code_sim_score",
    "longbench_repobench-p": "code_sim_score",
}

# Default configuration parameters
# pylint: disable=use-dict-literal
DEFAULT_EVAL_CONFIG = dict(
    batch_size="auto",
    apply_chat_template=True,
    fewshot_as_multiturn=True,
    confirm_run_unsafe_code=True,
    system_instruction=None,
    cache_requests=False,
)

# vLLM-specific configuration - using longer context window than leaderboard
# pylint: disable=use-dict-literal
DEFAULT_VLLM_CONFIG = dict(
    dtype="float16",
    gpu_memory_utilization=0.8,
    disable_custom_all_reduce=True,
    enforce_eager=False,
    max_model_len=131072,  # 128K context for LongBench
)

# OpenAI API configuration parameters
# pylint: disable=use-dict-literal
DEFAULT_OPENAI_CONFIG = dict(
    max_tokens=768,
    temperature=0.0,
    seed=1337,
)


class LongBenchEvaluator(Evaluator):
    """
    Evaluator for LongBench benchmark.

    Attributes:
        model_path: Path to the model or model name for API
        tasks: List of subtasks to evaluate (default is all tasks)
        num_gpus: Number of GPUs to use for local evaluation
        output_file: Path to save results to
        eval_config: Configuration for evaluation parameters
        vllm_config: Configuration for vLLM-specific parameters
        openai_config: Configuration for OpenAI API parameters
        api_endpoint: Optional OpenAI-compatible API endpoint
    """

    name = "longbench"

    def __init__(
        self,
        model_path: str,
        model_name: str,
        tasks: t.Optional[t.List[str]] = None,
        num_gpus: t.Optional[int] = None,
        output_file: t.Optional[str] = None,
        eval_config: t.Optional[t.Dict[str, t.Any]] = None,
        vllm_config: t.Optional[t.Dict[str, t.Any]] = None,
        openai_config: t.Optional[t.Dict[str, t.Any]] = None,
        api_endpoint: t.Optional[str] = None,
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.tasks = tasks or ALL_LONGBENCH_TASKS

        # If using API, no need to check CUDA
        self.api_endpoint = api_endpoint
        if not api_endpoint and not cuda.is_available():
            raise ValueError(
                "Running without CUDA is currently unsupported unless using an API endpoint"
            )

        self.num_gpus = num_gpus or cuda.device_count()
        self.output_file = output_file
        self.eval_config = eval_config if eval_config else {}
        self.vllm_config = vllm_config if vllm_config else {}
        self.openai_config = openai_config if openai_config else {}
        self._results: t.Optional[LongBenchResult] = None
        self._lm_eval_results: t.Optional[t.Dict[str, t.Any]] = None

    def _get_task_averages(self, results: dict) -> LongBenchResult:
        """Calculate averages for each task type and language from raw results"""
        eval_results = defaultdict(float)
        results_data = results["results"]

        # Track which categories have data
        active_categories = {}

        # Process each category
        for category, category_tasks in TASK_CATEGORIES.items():
            # Filter tasks that were actually run
            active_tasks = [task for task in category_tasks if task in results_data]

            if active_tasks:
                # Get scores for active tasks
                scores = []
                # pylint: disable=redefined-outer-name
                for task in active_tasks:
                    metric_key = f"{TASK_METRICS[task]},none"
                    if task in results_data and metric_key in results_data[task]:
                        scores.append(results_data[task][metric_key])

                if scores:
                    # Calculate average for this category
                    eval_results[category] = sum(scores) / len(scores)
                    active_categories[category] = len(scores)

        # Calculate overall score from active categories
        category_scores = [v for k, v in eval_results.items() if k != "overall_score"]
        if category_scores:
            eval_results["overall_score"] = sum(category_scores) / len(category_scores)
        else:
            eval_results["overall_score"] = 0.0

        return t.cast(LongBenchResult, dict(eval_results))

    def run(
        self,
        model_path: t.Optional[str] = None,
        model_name: t.Optional[str] = None,
        tasks: t.Optional[t.List[str]] = None,
        num_gpus: t.Optional[int] = None,
        output_file: t.Optional[str] = None,
        eval_config: t.Optional[t.Dict[str, t.Any]] = None,
        vllm_config: t.Optional[t.Dict[str, t.Any]] = None,
        openai_config: t.Optional[t.Dict[str, t.Any]] = None,
        api_endpoint: t.Optional[str] = None,
    ) -> LongBenchResult:
        """Run the LongBench evaluation"""
        model_path = model_path or self.model_path
        model_name = model_name or self.model_name
        tasks = tasks or self.tasks
        num_gpus = num_gpus or self.num_gpus
        output_file = output_file or self.output_file
        api_endpoint = api_endpoint or self.api_endpoint

        # Merge configurations
        final_eval_config = {}
        final_eval_config.update(DEFAULT_EVAL_CONFIG)
        final_eval_config.update(self.eval_config)
        if eval_config:
            final_eval_config.update(eval_config)

        final_vllm_config = {}
        final_vllm_config.update(DEFAULT_VLLM_CONFIG)
        final_vllm_config.update(self.vllm_config)
        if vllm_config:
            final_vllm_config.update(vllm_config)

        final_openai_config = {}
        final_openai_config.update(DEFAULT_OPENAI_CONFIG)
        final_openai_config.update(self.openai_config)
        if openai_config:
            final_openai_config.update(openai_config)

        # Extract system_instruction if provided
        system_instruction = final_eval_config.pop("system_instruction", None)

        # Run evaluation with the appropriate backend
        if api_endpoint:
            base_url = api_endpoint
            api_key = final_openai_config.pop("api_key", None)

            # Build model args
            model_args = {
                "model": model_name,
                "tokenizer": model_path,
                "base_url": base_url,
            }
            # Optionally add max_length
            if "max_length" in final_openai_config:
                model_args["max_length"] = str(final_openai_config["max_length"])

            if api_key:
                model_args["api_key"] = str(api_key)

            # Add any other openai_config keys if needed
            # model_args.update(final_openai_config)

            # Run evaluation
            results = simple_evaluate(
                tasks=tasks,
                model="local-completions",
                model_args=model_args,
                system_instruction=system_instruction,
                **final_eval_config,
            )
        else:
            # Prepare vLLM model args
            model_args = {
                "pretrained": model_path,
                "data_parallel_size": str(num_gpus),
            }
            # Add vllm config properly - convert all values to strings
            string_vllm_config = {k: str(v) for k, v in final_vllm_config.items()}
            model_args.update(string_vllm_config)

            # Run evaluation
            results = simple_evaluate(
                tasks=tasks,
                model="vllm",
                model_args=model_args,
                system_instruction=system_instruction,
                **final_eval_config,
            )

        self._lm_eval_results = results
        self._results = self._get_task_averages(results)

        if output_file:
            self.save_to_file(output_file)

        return self._results

    @property
    def results(self) -> t.Optional[LongBenchResult]:
        """Returns the results of the most recent evaluation"""
        return self._results

    def save_to_file(self, output_file: t.Optional[str] = None) -> None:
        """Save results to a JSON file"""
        output_file = output_file or self.output_file
        if not output_file:
            raise ValueError("Output file path cannot be empty")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self._results, f, indent=2)
