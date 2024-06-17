# SPDX-License-Identifier: Apache-2.0
import os

# Local
from .evaluator import Evaluator
import instructlab.eval.gen_api_answer as gen_api_answer
import instructlab.eval.gen_judgement as gen_judgement


class MT_Bench_Evaluator(Evaluator):
    """
    Child class of an Evaluator for Multi-turn Benchmark (MT-Bench)

    Attributes
        server_url  vLLM server endpoint
    """

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url

    def gen_answers(self, answer_file, server_url) -> str:
        """ Asks questions to model, returns path to answers"""
        os.environ['OPENAI_API_KEY'] = "NO_API_KEY"
        gen_api_answer.run(answer_file=answer_file, model_name="instructlab/granite-7b-lab", openai_api_base=server_url)
        return answer_file

    #def judge_answers(self, judge_endpoint) -> tuple:
    def judge_answers(self, judge_endpoint) -> str:
        """
        Runs MT-Bench judgement

        Returns:
            overall_score   MT-Bench score for the overall model evaluation
            qa_pairs        Question and answer pairs from the evaluation
        """
        os.environ['OPENAI_API_BASE'] = judge_endpoint
        os.environ['OPENAI_API_KEY'] = "NO_API_KEY"
        output_file = gen_judgement.run(parallel=40)
        return output_file


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
