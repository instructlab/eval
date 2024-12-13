# SPDX-License-Identifier: Apache-2.0

"""
MMLU - Massive Multitask Language Understanding
https://en.wikipedia.org/wiki/MMLU
https://arxiv.org/abs/2009.03300
"""

# Standard
from typing import Any, Dict, Optional, Union
import os

# Third Party
from lm_eval.evaluator import simple_evaluate
from lm_eval.tasks import TaskManager
import torch

# First Party
from instructlab.eval.evaluator import Evaluator
from instructlab.eval.exceptions import (
    InvalidModelError,
    InvalidTasksDirError,
    ModelNotFoundError,
    TasksDirNotFoundError,
)

# Local
from .logger_config import setup_logger

logger = setup_logger(__name__)

MMLU_TASKS = [
    "mmlu_abstract_algebra",
    "mmlu_anatomy",
    "mmlu_astronomy",
    "mmlu_business_ethics",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_chemistry",
    "mmlu_college_computer_science",
    "mmlu_college_mathematics",
    "mmlu_college_medicine",
    "mmlu_college_physics",
    "mmlu_computer_security",
    "mmlu_conceptual_physics",
    "mmlu_econometrics",
    "mmlu_electrical_engineering",
    "mmlu_elementary_mathematics",
    "mmlu_formal_logic",
    "mmlu_global_facts",
    "mmlu_high_school_biology",
    "mmlu_high_school_chemistry",
    "mmlu_high_school_computer_science",
    "mmlu_high_school_european_history",
    "mmlu_high_school_geography",
    "mmlu_high_school_government_and_politics",
    "mmlu_high_school_macroeconomics",
    "mmlu_high_school_mathematics",
    "mmlu_high_school_microeconomics",
    "mmlu_high_school_physics",
    "mmlu_high_school_psychology",
    "mmlu_high_school_statistics",
    "mmlu_high_school_us_history",
    "mmlu_high_school_world_history",
    "mmlu_human_aging",
    "mmlu_human_sexuality",
    "mmlu_international_law",
    "mmlu_jurisprudence",
    "mmlu_logical_fallacies",
    "mmlu_machine_learning",
    "mmlu_management",
    "mmlu_marketing",
    "mmlu_medical_genetics",
    "mmlu_miscellaneous",
    "mmlu_moral_disputes",
    "mmlu_moral_scenarios",
    "mmlu_nutrition",
    "mmlu_philosophy",
    "mmlu_prehistory",
    "mmlu_professional_accounting",
    "mmlu_professional_law",
    "mmlu_professional_medicine",
    "mmlu_professional_psychology",
    "mmlu_public_relations",
    "mmlu_security_studies",
    "mmlu_sociology",
    "mmlu_us_foreign_policy",
    "mmlu_virology",
    "mmlu_world_religions",
]


class AbstractMMLUEvaluator(Evaluator):
    """
    Abstract child class of an Evaluator for Massive Multitask Language Understanding Branch

    Attributes:
        model_path      absolute path to or name of a huggingface model
        tasks_dir       path where the <TASK_NAME>.jsonl and <TASK_NAME>_task.yaml files for the branches being evaluated are stored
        tasks           list of tasks for MMLU to test the model with
        model_dtype     dtype of model when served
        few_shots       number of examples
        batch_size      batch size for evaluation. Valid values are a positive integer or 'auto' to select the largest batch size that will fit in memory, or 'auto:N' to reselect the largest batch size N times'.
        device          PyTorch device (e.g. "cpu" or "cuda:0") for running models
        system_prompt   system prompt to be used when applying the chat template
        results         full output from the `lm_eval.evaluator.simple_evaluate` function after MMLU has run.
    """

    def __init__(
        self,
        model_path,
        tasks_dir: Optional[str],
        tasks: list[str],
        model_dtype="bfloat16",
        few_shots: int = 5,
        batch_size: Optional[Union[int, str]] = "auto",
        device: str = ("cuda" if torch.cuda.is_available() else "cpu"),
        system_prompt: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.tasks_dir = tasks_dir
        self.tasks = tasks
        self.model_dtype = model_dtype
        self.few_shots = few_shots
        self.batch_size = batch_size
        self.device = device
        self._results = None

    @property
    def results(self) -> Dict[str, Any] | None:
        """
        Returns the results of the last MMLU evaluation, if one has taken place.

        Returns:
            Dict[str, Any] | None: The output from `lm_eval.evaluator.simple_evaluate`
        """
        return self._results

    def run(
        self, server_url: str | None = None, extra_args: Dict[str, Any] | None = None
    ) -> tuple:
        """
        Runs evaluation

        Attributes
            server_url          Model server endpoint (Ex: http://localhost:8000/v1) for the model being evaluated
            extra_args          Dictionary containing any extra arguments to be passed into the lm_eval `lm_eval.evaluator.simple_evaluate` function.

        Returns:
            overall_score       Average score for the task group
            individual_scores   Individual scores for each task in the task group
        """
        extra_args = {} if not extra_args else extra_args
        logger.debug(locals())

        # TODO: make this a parameter for class?
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        individual_scores: dict = {}
        agg_score: float = 0.0

        results = self._run_mmlu(server_url)
        for task, result in results.items():
            agg_score += float(result["acc,none"])
            individual_scores[task] = {
                "score": float(result["acc,none"]),
                "stderr": float(result["acc_stderr,none"]),
            }

        overall_score = float(agg_score / len(individual_scores))

        return overall_score, individual_scores

    def _run_mmlu(
        self, server_url: str | None = None, extra_args: Dict[str, Any] | None = None
    ) -> dict:
        extra_args = {} if not extra_args else extra_args
        if server_url is not None:
            # Requires lm_eval >= 0.4.4
            model_args = f"base_url={server_url}/completions,model={self.model_path},tokenizer_backend=huggingface"
            model = "local-completions"
        else:
            model_args = f"pretrained={self.model_path},dtype={self.model_dtype}"
            model = "hf"
        tm = None
        if self.tasks_dir is not None:
            if not os.path.exists(self.tasks_dir):
                raise TasksDirNotFoundError(self.tasks_dir)
            if not os.access(self.tasks_dir, os.R_OK):
                raise InvalidTasksDirError(self.tasks_dir)
            tm = TaskManager(verbosity="DEBUG", include_path=self.tasks_dir)
        should_apply_chat_template = self.system_prompt is not None

        # configure the args here so users can override them as necessary
        simple_evaluate_kwargs = {
            "model": model,
            "model_args": model_args,
            "tasks": self.tasks,
            "num_fewshot": self.few_shots,
            "batch_size": self.batch_size,
            "device": self.device,
            "task_manager": tm,
            "system_instruction": self.system_prompt,
            "apply_chat_template": should_apply_chat_template,
        }
        simple_evaluate_kwargs.update(extra_args)

        results = self._simple_evaluate_with_error_handling(**simple_evaluate_kwargs)
        self._results = results
        return results["results"]

    # This method converts general errors from simple_evaluate
    # into a more user-understandable error
    def _simple_evaluate_with_error_handling(self, **kwargs):
        try:
            return simple_evaluate(**kwargs)
        except KeyError as ke:
            # If the first task key file cannot be found in tasks_dir, simple_evaluate() will return
            # an obscure KeyError(first task key)
            if (
                self.tasks_dir is not None
                and len(self.tasks) > 0
                and ke.args[0] == self.tasks[0]
            ):
                raise InvalidTasksDirError(self.tasks_dir) from ke
            raise
        except OSError as ose:
            # If a model can not be found, simple_evaluate() will return
            # an obscure OSError with a message
            if "is not a valid model" in str(
                ose
            ) or "does not appear to have a file named" in str(ose):
                raise ModelNotFoundError(self.model_path) from ose
            if "is not a valid JSON file" in str(ose):
                reason = "Looked for valid JSON file but couldn't find one - are you pointing at a directory with a 'config.json'?"
                raise InvalidModelError(self.model_path, reason) from ose
            raise


class MMLUEvaluator(AbstractMMLUEvaluator):
    """
    Evaluator for Massive Multitask Language Understanding (MMLU)

    Attributes:
        model_path      absolute path to or name of a huggingface model
        tasks           list of tasks for MMLU to test the model with
        model_dtype     dtype of model when served
        few_shots       number of examples
        batch_size      batch size for evaluation. Valid values are a positive integer or 'auto' to select the largest batch size that will fit in memory, or 'auto:N' to reselect the largest batch size N times'.
        device          PyTorch device (e.g. "cpu" or "cuda:0") for running models
        system_prompt   system prompt to be used when applying the chat template
    """

    name = "mmlu"

    def __init__(
        self,
        model_path,
        tasks: list[str] = MMLU_TASKS,
        model_dtype="bfloat16",
        few_shots: int = 5,
        batch_size: Optional[Union[int, str]] = "auto",
        device: str = ("cuda" if torch.cuda.is_available() else "cpu"),
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_path,
            None,
            tasks,
            model_dtype,
            few_shots,
            batch_size,
            device,
            system_prompt=system_prompt,
        )


class MMLUBranchEvaluator(AbstractMMLUEvaluator):
    """
    Evaluator for Massive Multitask Language Understanding Branch (MMLUBranch)

    Attributes:
        model_path      absolute path to or name of a huggingface model
        system_prompt   system prompt to be used when applying the chat template
        tasks_dir       path where the <TASK_NAME>.jsonl and <TASK_NAME>_task.yaml files for the branches being evaluated are stored
        tasks           group name that is shared by all the MMLUBranch tasks
        model_dtype     dtype of model when served
        few_shots       number of examples
        batch_size      batch size for evaluation. Valid values are a positive integer or 'auto' to select the largest batch size that will fit in memory, or 'auto:N' to reselect the largest batch size N times'.
        device          PyTorch device (e.g. "cpu" or "cuda:0") for running models
    """

    name = "mmlu_branch"
