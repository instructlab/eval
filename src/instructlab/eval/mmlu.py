# SPDX-License-Identifier: Apache-2.0

# Local
from .evaluator import Evaluator


class MMLU_Evaluator(Evaluator):
    """
    Child class of an Evaluator for Massive Multitask Language Understanding (MMLU)

    Attributes:
        tasks        list of tasks for MMLU to test the model with
        few_shots    number of examples
        batch_size   number of GPUs
    """

    def __init__(
        self, model_path, tasks: list[str], few_shots: int = 2, batch_size: int = 5
    ) -> None:
        self.model_path = model_path
        self.tasks = tasks
        self.few_shots = few_shots
        self.batch_size = batch_size

    def run(self) -> tuple:
        """
        Runs MMLU evaluation

        Returns:
            overall_score       MMLU score for the overall model evaluation
            individual_scores   Individual MMLU score for each task
        """
        individual_scores: dict[str, float] = {}
        overall_score: float = 0.0
        return overall_score, individual_scores


class PR_MMLU_Evaluator(Evaluator):
    """
    Child class of an Evaluator for PR Massive Multitask Language Understanding (PR MMLU)

    Attributes:
        sdg_path    path where all the PR MMLU tasks are stored
        task        group name that is shared by all the PR MMLU tasks
        few_shots   number of examples
        batch_size  number of GPUs
    """

    def __init__(
        self,
        model_path,
        sdg_path: str,
        task: str = "mmlu_pr",
        few_shots: int = 2,
        batch_size: int = 5,
    ) -> None:
        self.model_path = model_path
        self.sdg_path = sdg_path
        self.task = task
        self.few_shots = few_shots
        self.batch_size = batch_size

    def run(self) -> tuple:
        """
        Runs PR MMLU evaluation

        Returns:
            overall_score       PR MMLU score for the overall model evaluation
            individual_scores   Individual PR MMLU scores for each task
            qa_pairs            Question and answer pairs from the evaluation
        """
        individual_scores: dict[str, float] = {}
        overall_score: float = 0.0
        qa_pairs: list[tuple] = []
        return overall_score, individual_scores, qa_pairs
