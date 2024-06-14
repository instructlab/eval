# SPDX-License-Identifier: Apache-2.0

# Local
from .evaluator import Evaluator


class MT_Bench_Evaluator(Evaluator):
    """
    Child class of an Evaluator for Multi-turn Benchmark (MT-Bench)
    """

    def __init__(self, model, server: str) -> None:
        super().__init__(model)
        self.server = server


class PR_Bench_Evaluator(Evaluator):
    """
    Child class of an Evaluator for PR-Bench Benchmark (PR-Bench)
    """

    def __init__(self, model, server: str, questions: str) -> None:
        super().__init__(model)
        self.server = server
        self.questions = questions
