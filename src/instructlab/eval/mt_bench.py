# SPDX-License-Identifier: Apache-2.0

"""
Multi-Turn Benchmark
https://klu.ai/glossary/mt-bench-eval
https://arxiv.org/html/2306.05685
"""

# First Party
from instructlab.eval import (
    mt_bench_answers,
    mt_bench_branch_generator,
    mt_bench_judgment,
)

# Local
from .evaluator import Evaluator
from .logger_config import setup_logger

logger = setup_logger(__name__)


class MTBenchEvaluator(Evaluator):
    """
    Child class of an Evaluator for Multi-turn Benchmark (MT-Bench)

    Attributes
        model_name                  Name of the model to evaluate
        judge_model_name            Name of the judge model
        output_dir                  The directory to use for evaluation output
        max_workers                 Max parallel workers to run the evaluation with
        merge_system_user_message   Boolean indicating whether to merge system and user messages (required for Mistral based judges)
    """

    name = "mt_bench"

    def __init__(
        self,
        model_name: str,
        judge_model_name: str,
        output_dir: str = "eval_output",
        max_workers: int = 40,
        merge_system_user_message: bool = False,
    ) -> None:
        self.model_name = model_name
        self.judge_model_name = judge_model_name
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.merge_system_user_message = merge_system_user_message

    def gen_answers(self, server_url) -> None:
        """
        Asks questions to model

        Attributes
            server_url      Model server endpoint (Ex: http://localhost:8000/v1) for the model being evaluated
        """
        logger.debug(locals())
        mt_bench_answers.generate_answers(
            self.model_name,
            server_url,
            output_dir=self.output_dir,
            max_workers=self.max_workers,
        )

    def judge_answers(self, server_url) -> tuple:
        """
        Runs MT-Bench judgment

        Attributes
            server_url      Model server endpoint (Ex: http://localhost:8000/v1) for the judge model

        Returns:
            overall_score   MT-Bench score for the overall model evaluation
            qa_pairs        Question and answer pairs (with scores) from the evaluation
            turn_scores     A list of indexed turn scores
        """
        logger.debug(locals())
        return mt_bench_judgment.generate_judgment(
            self.model_name,
            self.judge_model_name,
            server_url,
            max_workers=self.max_workers,
            output_dir=self.output_dir,
            merge_system_user_message=self.merge_system_user_message,
        )


class MTBenchBranchEvaluator(Evaluator):
    """
    Child class of an Evaluator for MT-Bench-Branch Benchmark

    Attributes
        model_name                  Name of the model to evaluate
        judge_model_name            Name of the judge model
        taxonomy_git_repo_path      Taxonomy git repo path
        branch                      Branch of taxonomy repo to eval QNAs against model
        output_dir                  The directory to use for evaluation output
        max_workers                 Max parallel workers to run the evaluation with
        merge_system_user_message   Boolean indicating whether to merge system and user messages (required for Mistral based judges)
    """

    name = "mt_bench_branch"

    def __init__(
        self,
        model_name: str,
        judge_model_name: str,
        taxonomy_git_repo_path: str,
        branch: str,
        output_dir: str = "eval_output",
        max_workers: int = 40,
        merge_system_user_message: bool = False,
    ) -> None:
        self.model_name = model_name
        self.judge_model_name = judge_model_name
        self.taxonomy_git_repo_path = taxonomy_git_repo_path
        self.branch = branch
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.merge_system_user_message = merge_system_user_message

    def gen_answers(self, server_url) -> None:
        """
        Asks questions to model

        Attributes
            server_url  Model server endpoint (Ex: http://localhost:8000/v1) for the model being evaluated
        """
        logger.debug(locals())
        mt_bench_branch_generator.generate(
            self.judge_model_name,
            self.branch,
            self.taxonomy_git_repo_path,
            self.output_dir,
        )
        mt_bench_answers.generate_answers(
            self.model_name,
            server_url,
            branch=self.branch,
            output_dir=self.output_dir,
            data_dir=self.output_dir,
            max_workers=self.max_workers,
            bench_name="mt_bench_branch",
        )

    def judge_answers(self, server_url) -> tuple:
        """
        Runs MT-Bench-Branch judgment.  Judgments can be compared across runs with consistent question_id -> qna file name.

        Attributes
            server_url      Model server endpoint (Ex: http://localhost:8000/v1) for the judge model

        Returns:
            qa_pairs        Question and answer pairs (with scores) from the evaluation
        """
        logger.debug(locals())
        _, qa_pairs, _, error_rate = mt_bench_judgment.generate_judgment(
            self.model_name,
            self.judge_model_name,
            server_url,
            branch=self.branch,
            max_workers=self.max_workers,
            output_dir=self.output_dir,
            data_dir=self.output_dir,
            bench_name="mt_bench_branch",
            merge_system_user_message=self.merge_system_user_message,
        )
        return qa_pairs, error_rate
