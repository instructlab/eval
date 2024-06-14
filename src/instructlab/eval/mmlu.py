# SPDX-License-Identifier: Apache-2.0

# Local
from .evaluator import Evaluator


class MMLU_Evaluator(Evaluator):
    """
    Child class of an Evaluator for Massive Multitask Language Understanding (MMLU)
    """

    def __init__(self, model, tasks: list[str], fewshots: int, batchsize: int) -> None:
        super().__init__(model)
        self.tasks = tasks
        self.fewshots = fewshots
        self.batchsize = batchsize


class PR_MMLU_Evaluator(Evaluator):
    """
    Child class of an Evaluator for PR Massive Multitask Language Understanding (PR MMLU)
    """

    def __init__(
        self, model, task: str, sdg_path: str, fewshots: int, batchsize: int
    ) -> None:
        super().__init__(model)
        self.task = task
        self.sdg_path = sdg_path
        self.fewshots = fewshots
        self.batchsize = batchsize
