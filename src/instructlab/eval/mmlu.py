# SPDX-License-Identifier: Apache-2.0

# Standard
import os

# Third Party
from lm_eval.evaluator import simple_evaluate  # type: ignore
from lm_eval.tasks import TaskManager  # type: ignore
import torch

# First Party
from instructlab.eval.evaluator import Evaluator


class MMLUEvaluator(Evaluator):
    """
    Child class of an Evaluator for Massive Multitask Language Understanding (MMLU)

    Attributes:
        model_path   absolute path to or name of a huggingface model
        tasks        list of tasks for MMLU to test the model with
        model_dtype  dtype of model when served
        few_shots    number of examples
        batch_size   number of GPUs
    """

    def __init__(
        self,
        model_path,
        tasks: list[str],
        model_dtype="bfloat16",
        few_shots: int = 2,
        batch_size: int = 5,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.tasks = tasks
        self.model_dtype = model_dtype
        self.few_shots = few_shots
        self.batch_size = batch_size

    def run(self) -> tuple:
        """
        Runs MMLU evaluation

        Returns:
            overall_score       MMLU score for the overall model evaluation
            individual_scores   Individual MMLU score for each task
        """
        # TODO: make this a parameter for class?
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        individual_scores: dict = {}
        agg_score: float = 0.0
        model_args = f"pretrained={self.model_path},dtype={self.model_dtype}"
        mmlu_output = simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=self.tasks,
            num_fewshot=self.few_shots,
            batch_size=self.batch_size,
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )

        results = mmlu_output["results"]

        for task in self.tasks:
            mmlu_res = results[task]
            agg_score += float(mmlu_res["acc,none"])
            individual_scores[task] = {}
            individual_scores[task]["score"] = float(mmlu_res["acc,none"])
            individual_scores[task]["stderr"] = float(mmlu_res["acc_stderr,none"])

        overall_score = float(agg_score / len(self.tasks))
        return overall_score, individual_scores


class MMLUBranchEvaluator(Evaluator):
    """
    Child class of an Evaluator for Massive Multitask Language Understanding Branch (MMLUBranch)

    Attributes:
        model_path  absolute path to or name of a huggingface model
        sdg_path    path where the <TASK_NAME>.jsonl and <TASK_NAME>_task.yaml files for the branches being evaluated are stored
        tasks       group name that is shared by all the MMLUBranch tasks
        few_shots   number of examples
        batch_size  number of GPUs
    """

    def __init__(
        self,
        model_path,
        sdg_path: str,
        tasks: list[str],
        model_dtype="bfloat16",
        few_shots: int = 2,
        batch_size: int = 5,
    ) -> None:
        self.model_path = model_path
        self.sdg_path = sdg_path
        self.tasks = tasks
        self.model_dtype = model_dtype
        self.few_shots = few_shots
        self.batch_size = batch_size

    def run(self) -> tuple:
        """
        Runs MMLUBranch evaluation

        Returns:
            overall_score       Average MMLUBranch score for the task group
            individual_scores   Individual MMLUBranch scores for each task in the task group
        """
        # TODO: make this a parameter for class?
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        individual_scores: dict = {}
        agg_score: float = 0.0
        model_args = f"pretrained={self.model_path},dtype={self.model_dtype}"

        tm = TaskManager(verbosity="DEBUG", include_path=self.sdg_path)

        mmlu_output = simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=self.tasks,
            num_fewshot=self.few_shots,
            batch_size=self.batch_size,
            task_manager=tm,
        )
        results = mmlu_output["results"]

        for task in self.tasks:
            mmlu_res = results[task]
            agg_score += float(mmlu_res["acc,none"])
            individual_scores[task] = {}
            individual_scores[task]["score"] = float(mmlu_res["acc,none"])
            individual_scores[task]["stderr"] = float(mmlu_res["acc_stderr,none"])

        overall_score = float(agg_score / len(self.tasks))
        return overall_score, individual_scores
