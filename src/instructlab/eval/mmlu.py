# SPDX-License-Identifier: Apache-2.0

# Local
from .evaluator import Evaluator


class MMLUEvaluator(Evaluator):
    """
    Child class of an Evaluator for Massive Multitask Language Understanding (MMLU)
    """

    def __init__(self, model, tasks: list[str], fewshots: int, batchsize: int) -> None:
        super().__init__(model)
        self.tasks = tasks
        self.fewshots = fewshots
        self.batchsize = batchsize
