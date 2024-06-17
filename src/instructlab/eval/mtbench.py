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

    def run(self) -> tuple:
        """
        Runs MT-Bench evaluation

        Returns:
            overall_score   MT-Bench score for the overall model evaluation
            qa_pairs        Question and answer pairs from the evaluation
        """
        overall_score: float = 0.0
        qa_pairs: list[tuple] = []
        return overall_score, qa_pairs


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

    def run(self) -> tuple:
        """
        Runs PR-Bench evaluation

        Returns:
            overall_score   MT-Bench score for the overall model evaluation
            qa_pairs        Question and answer pairs from the evaluation
        """
        overall_score = 0.0
        qa_pairs: list[tuple] = []
        return overall_score, qa_pairs
