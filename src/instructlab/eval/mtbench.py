# SPDX-License-Identifier: Apache-2.0

# Local
from .evaluator import Evaluator


class MT_Bench_Evaluator(Evaluator):
    """
    Child class of an Evaluator for Multi-turn Benchmark (MT-Bench)

    Attributes
        server_url  vLLM server endpoint
    """

    def __init__(self, model_path, server_url: str) -> None:
        super().__init__(model_path)
        self.server_url = server_url

    def run(self) -> dict:
        overall_score: float = 0.0
        qa_pairs: list[tuple] = []
        payload = {"overall_score": overall_score, "qa_pairs": qa_pairs}
        return payload


class PR_Bench_Evaluator(Evaluator):
    """
    Child class of an Evaluator for PR-Bench Benchmark (PR-Bench)

    Attributes
        server_url  vLLM server endpoint
        questions   questions to be asked
    """

    def __init__(self, model_path, server_url: str, questions: str) -> None:
        super().__init__(model_path)
        self.server_url = server_url
        self.questions = questions

    def run(self) -> dict:
        overall_score = 0.0
        qa_pairs: list[tuple] = []
        payload = {"overall_score": overall_score, "qa_pairs": qa_pairs}
        return payload
