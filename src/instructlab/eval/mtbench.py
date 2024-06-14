# SPDX-License-Identifier: Apache-2.0

# Local
from .evaluator import Evaluator


class MT_Bench_Evaluator(Evaluator):
    """
    Child class of an Evaluator for Multi-turn Benchmark (MT-Bench)

    Attributes
        server_url  vLLM server endpoint
    """

    def __init__(self, model, server_url: str) -> None:
        super().__init__(model)
        self.server_url = server_url


class PR_Bench_Evaluator(Evaluator):
    """
    Child class of an Evaluator for PR-Bench Benchmark (PR-Bench)

    Attributes
        server_url  vLLM server endpoint
        questions   questions to be asked
    """

    def __init__(self, model, server_url: str, questions: str) -> None:
        super().__init__(model)
        self.server_url = server_url
        self.questions = questions
