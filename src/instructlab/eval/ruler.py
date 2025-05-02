# Standard
from typing import Any, Dict, List, Optional
import json
import os
import pathlib

# Third Party
from lm_eval.evaluator import simple_evaluate

# First Party
from instructlab.eval.evaluator import Evaluator

RULER_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "ruler_vt",
    "ruler_cwe",
    "ruler_fwe",
    "ruler_qa_hotpot",
    "ruler_qa_squad",
]

DEFAULT_MAX_LENGTH = 4096


class RulerEvaluator(Evaluator):
    """
    Class definition for running RULER benchmarking tasks.
    """

    name = "ruler"

    def __init__(
        self,
        model_path: Optional[str] = None,
        output_file: Optional[str] = None,
        tasks: list[str] = RULER_TASKS,
        api_endpoint: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> None:
        self.model_path = model_path
        self.tasks = tasks
        self.results: Dict[Any, Any] = {}
        self.output_file = output_file

        self.api_endpoint = api_endpoint or None
        self.max_length = max_length or 4096

    def save_to_file(self, output_file: Optional[str] = None) -> None:
        """Save results to a JSON file"""
        output_file = output_file if output_file else self.output_file
        if not output_file:
            raise ValueError("Output file path cannot be empty")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

    def process_lm_eval_results(
        self,
        fpath: Optional[pathlib.Path] = None,
        raw_results: Optional[dict[str, Any]] = None,
    ) -> dict[str, float]:
        """
        Process the evaluation results from lm_eval for the given file path and extract
        aggregarted scores for each context length
        Args:
            fpath (pathlib.Path): The file path to the evaluation results.

        """
        unqiue_metrics_dict: dict[str, Any] = {}

        # This is required because the lm_eval results are nested under 'ruler' if
        # that is the supplied task to it. The output contains a nested dictionary
        # in this case, using RULER tasks as the key. Each context length is a further subkey
        # in the dictionary. There is an additional key per context length which also
        # contains score adjusted for stderr, which we are ignoring here.
        def extract_metrics(results: dict, unqiue_metrics_dict: dict = {}):
            for k, v in results.items():
                if isinstance(v, dict):
                    extract_metrics(v, unqiue_metrics_dict)
                else:
                    if "stderr" not in k:
                        metric = k.split(",")[0]
                        if metric not in unqiue_metrics_dict:
                            unqiue_metrics_dict[metric] = []
                        unqiue_metrics_dict[metric].append(v)

            return unqiue_metrics_dict

        if fpath:
            with open(fpath, "r", encoding="utf-8") as f:
                raw_results = json.load(f)

        if raw_results is not None:
            extract_metrics(raw_results["results"], unqiue_metrics_dict)
        unique_float_metrics = {}
        # if value is list of floats, average the list
        for k, v in unqiue_metrics_dict.items():
            if isinstance(v, list) and all(isinstance(i, float) for i in v):
                unique_float_metrics[k] = sum(v) / len(v)

        # find average of all float values in dict
        float_values = [
            v for v in unique_float_metrics.values() if isinstance(v, float)
        ]
        if float_values:
            unique_float_metrics["avg"] = sum(float_values) / len(float_values)
        else:
            unique_float_metrics["avg"] = 0.0

        # result format
        # {'8192': 0.90, '32768': 0.82, '65536': 0.77, '131072': 0.71, 'avg': 0.80}
        return unique_float_metrics

    def run(
        self,
        model_path: Optional[str] = None,
        tasks: Optional[List[str]] = None,
        output_file: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        max_length: Optional[int] = DEFAULT_MAX_LENGTH,
    ) -> None:
        """
        Run the RULER evaluation using the specified model and tasks.
        """

        model_path = self.model_path if model_path is None else model_path
        tasks = self.tasks if not tasks else tasks
        output_file = self.output_file if not output_file else output_file

        # validate above params are not none and output file can be written to
        if not model_path:
            raise ValueError("Model path cannot be empty")
        if not output_file:
            raise ValueError("Output file path cannot be empty")
        if not api_endpoint:
            raise ValueError("API endpoint cannot be empty")

        # Prepare model_args
        model_args = {
            "pretrained": model_path,
            "base_url": api_endpoint,
            "max_length": max_length,
        }

        self.lm_eval_results = simple_evaluate(
            model="local-completions",
            model_args=model_args,
            tasks=tasks,
        )

        self.result = self.process_lm_eval_results(
            raw_results=self.lm_eval_results,
        )

        # write results to file
        if output_file:
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(self.result, f, indent=2)
            except (OSError, IOError) as e:
                raise ValueError(f"Failed to write to output file: {e}") from e
