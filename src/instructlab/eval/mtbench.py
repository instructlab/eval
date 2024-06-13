# SPDX-License-Identifier: Apache-2.0

# Local
from .evaluator import Evaluator


class MTBenchEvaluator(Evaluator):
    """
    Child class of an Evaluator for Multi-turn Benchmark (MT-Bench)
    """

    def __init__(self, model, server: str) -> None:
        super().__init__(model)
        self.server = server
