# Standard
from typing import List, TypedDict

# Third Party
from langchain_community.chat_models import ChatOpenAI
from ragas.evaluation import EvaluationDataset, EvaluationResult, RunConfig, evaluate
from ragas.metrics import RubricsScore
from ragas.metrics._domain_specific_rubrics import DEFAULT_WITH_REFERENCE_RUBRICS

# Local
from .evaluator import Evaluator


class Sample(TypedDict):
    # question
    user_input: str

    # model answer
    response: str

    # golden answer
    reference: str


class RagasEvaluator(Evaluator):
    # most basic implementation, we just assume that the user will bring the existing model responses
    name = "ragas"

    def __init__(self):
        pass

    def run(
        self, dataset: List[Sample], run_config: RunConfig | None = None
    ) -> EvaluationResult:
        """
        Evaluates the quality of model responses against a graded rubric.

        Args:
            dataset (List[Sample]):
                List of model questions and answers
            run_config (RunConfig | None, optional):
                Configuration to use when running evaluations. If none is provided, then
                a default one is created containing extremely permissive settings when handling
                timeouts. This is because by default, OpenAI tier-1 usage accounts have very high
                rate limits resulting in heavy throttling during evaluations.

        Returns:
            EvaluationResult: The results of all evaluations performed by Ragas
        """
        if not run_config:
            # we set extreme timeout/retry values by default since OpenAI tier-1 rate limits
            # are horrible and will result in half of our evaluation results being NaN or 0
            run_config = RunConfig(
                max_retries=120,
                max_wait=7200,
                seed=42,
                timeout=3600,
            )

        # we will be using gpt-4o for the foreseeable future, we hardcode this
        # for consistency of answers
        input_ds = EvaluationDataset.from_list(dataset)

        # default set of metrics
        metrics = [
            RubricsScore(
                rubrics=DEFAULT_WITH_REFERENCE_RUBRICS,
            )
        ]

        critic_lm = ChatOpenAI(model="gpt-4o")
        results = evaluate(
            dataset=input_ds,
            batch_size=4,
            run_config=run_config,
            llm=critic_lm,
            metrics=metrics,
            show_progress=True,
        )
        return results
