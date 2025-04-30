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
    en_multidoc: float
    zh_multidoc: float
    en_singledoc: float
    zh_singledoc: float
    en_summ: float
    zh_summ: float
    en_fewshot: float
    zh_fewshot: float
    en_synthetic: float
    zh_synthetic: float
    code_avg: float


# Default configuration parameters
DEFAULT_EVAL_CONFIG = {
    "batch_size": "auto",
    "apply_chat_template": True,
    "fewshot_as_multiturn": True,
    "confirm_run_unsafe_code": True,
    "system_instruction": None,
    "cache_requests": False,
}

DEFAULT_VLLM_CONFIG = {
    "dtype": "float16",
    "gpu_memory_utilization": 0.8,
    "disable_custom_all_reduce": True,
    "enforce_eager": False,
    "max_model_len": 131072,
}


class LongBenchEvaluator(Evaluator):
    """
    Evaluator for LongBenchV2 benchmark.

    Attributes:
        model_path: Path to the model to evaluate
        num_gpus: Number of GPUs to use
        output_file: Path to save results to
        eval_config: Configuration for evaluation parameters
        vllm_config: Configuration for vLLM-specific parameters
    """

    name = "longbench"

    def __init__(
        self,
        model_path: str,
        num_gpus: t.Optional[int] = None,
        output_file: t.Optional[str] = None,
        eval_config: t.Optional[t.Dict[str, t.Any]] = None,
        vllm_config: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        self.model_path = model_path
        if not cuda.is_available():
            raise ValueError("Running without CUDA is currently unsupported")

        self.num_gpus = num_gpus or cuda.device_count()
        self.output_file = output_file
        self.eval_config = eval_config or {}
        self.vllm_config = vllm_config or {}
        self._results: t.Optional[LongBenchResult] = None
        self._lm_eval_results: t.Optional[t.Dict[str, t.Any]] = None

    def _get_task_averages(self, results: dict) -> LongBenchResult:
        """Calculate averages for each task type and language from raw results"""
        eval_results = defaultdict(float)
        results = results["results"]

        # Multi-doc QA
        eval_results["en_multidoc"] = (
            results["longbench_hotpotqa"]["qa_f1_score,none"]
            + results["longbench_2wikimqa"]["qa_f1_score,none"]
            + results["longbench_musique"]["qa_f1_score,none"]
        ) / 3

        eval_results["zh_multidoc"] = results["longbench_dureader"][
            "rouge_zh_score,none"
        ]

        # Single-doc QA
        eval_results["en_singledoc"] = (
            results["longbench_multifieldqa_en"]["qa_f1_score,none"]
            + results["longbench_narrativeqa"]["qa_f1_score,none"]
            + results["longbench_qasper"]["qa_f1_score,none"]
        ) / 3

        eval_results["zh_singledoc"] = results["longbench_multifieldqa_zh"][
            "qa_f1_zh_score,none"
        ]

        # Summarization
        eval_results["en_summ"] = (
            results["longbench_gov_report"]["rouge_score,none"]
            + results["longbench_qmsum"]["rouge_score,none"]
            + results["longbench_multi_news"]["rouge_score,none"]
        ) / 3

        eval_results["zh_summ"] = results["longbench_vcsum"]["rouge_zh_score,none"]

        # Few-shot
        eval_results["en_fewshot"] = (
            results["longbench_triviaqa"]["qa_f1_score,none"]
            + results["longbench_samsum"]["rouge_score,none"]
            + results["longbench_trec"]["classification_score,none"]
        ) / 3

        eval_results["zh_fewshot"] = results["longbench_lsht"][
            "classification_score,none"
        ]

        # Synthetic
        eval_results["en_synthetic"] = (
            results["longbench_passage_retrieval_en"]["retrieval_score,none"]
            + results["longbench_passage_count"]["count_score,none"]
        ) / 2

        eval_results["zh_synthetic"] = results["longbench_passage_retrieval_zh"][
            "retrieval_zh_score,none"
        ]

        # Code (language-agnostic)
        eval_results["code_avg"] = (
            results["longbench_lcc"]["code_sim_score,none"]
            + results["longbench_repobench-p"]["code_sim_score,none"]
        ) / 2

        # Calculate overall score
        all_scores = [v for k, v in eval_results.items() if k != "overall_score"]
        eval_results["overall_score"] = sum(all_scores) / len(all_scores)

        return dict(eval_results)

    def run(
        self,
        model_path: t.Optional[str] = None,
        num_gpus: t.Optional[int] = None,
        output_file: t.Optional[str] = None,
        eval_config: t.Optional[t.Dict[str, t.Any]] = None,
        vllm_config: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> LongBenchResult:
        """Run the LongBench evaluation"""
        model_path = model_path or self.model_path
        num_gpus = num_gpus or self.num_gpus
        output_file = output_file or self.output_file

        # Merge configurations
        final_eval_config = {
            **DEFAULT_EVAL_CONFIG,
            **self.eval_config,
            **(eval_config or {}),
        }
        final_vllm_config = {
            **DEFAULT_VLLM_CONFIG,
            **self.vllm_config,
            **(vllm_config or {}),
        }

        # Prepare model args
        model_args = {
            "pretrained": model_path,
            "data_parallel_size": num_gpus,
            **final_vllm_config,
        }

        # Extract system_instruction if provided
        system_instruction = final_eval_config.pop("system_instruction", None)

        # Run evaluation
        results = simple_evaluate(
            tasks=["longbench"],
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
