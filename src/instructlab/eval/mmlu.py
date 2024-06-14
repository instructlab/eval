# SPDX-License-Identifier: Apache-2.0

# Local
from .evaluator import Evaluator


class MMLU_Evaluator(Evaluator):
    """
    Child class of an Evaluator for Massive Multitask Language Understanding (MMLU)

    Attributes:
        tasks       list of tasks for MMLU to test the model with
        fewshots    number of examples
        batchsize   number of GPUs
    """

    def __init__(
        self, model, tasks: list[str], fewshots: int = 2, batchsize: int = 5
    ) -> None:
        super().__init__(model)
        self.tasks = tasks
        self.fewshots = fewshots
        self.batchsize = batchsize

    def run(self) -> dict:
        individual_scores: dict[str, float] = {}
        overall_score: float = 0.0
        payload = {
            "individual_scores": individual_scores,
            "overall_score": overall_score,
        }
        return payload


class PR_MMLU_Evaluator(Evaluator):
    """
    Child class of an Evaluator for PR Massive Multitask Language Understanding (PR MMLU)

    Attributes:
        sdg_path    path where all the PR MMLU tasks are stored
        task        group name that is shared by all the PR MMLU tasks
        fewshots    number of examples
        batchsize   number of GPUs
    """

    def __init__(
        self,
        model,
        sdg_path: str,
        task: str = "mmlu_pr",
        fewshots: int = 2,
        batchsize: int = 5,
    ) -> None:
        super().__init__(model)
        self.sdg_path = sdg_path
        self.task = task
        self.fewshots = fewshots
        self.batchsize = batchsize

    def run(self) -> dict:
        individual_scores: dict[str, float] = {}
        overall_score: float = 0.0
        qa_pairs: list[tuple] = []
        payload = {
            "individual_scores": individual_scores,
            "overall_score": overall_score,
            "qa_pairs": qa_pairs,
        }
        return payload
