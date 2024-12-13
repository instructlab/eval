# # SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from typing import List, Optional, TypedDict

# Third Party
from langchain_community.chat_models import ChatOpenAI
from pandas import DataFrame, read_json
from pydantic import BaseModel, ConfigDict, field_validator
from ragas.evaluation import EvaluationDataset, EvaluationResult, RunConfig, evaluate
from ragas.metrics import Metric
from ragas.metrics._domain_specific_rubrics import (  # the rubrics we must instantiate are located inside of a file marked as private
    DEFAULT_WITH_REFERENCE_RUBRICS,
    RubricsScore,
)

# Local
from .evaluator import Evaluator
from .mt_bench_common import get_openai_client


class Sample(TypedDict):
    """
    TypedDict of a sample that we accept when doing eval with Ragas.
    We specifically use TypedDict here to be flexible with the input data we accept.
    """

    # question
    user_input: str

    # model answer
    response: Optional[str]

    # golden answer
    reference: str


# default system prompt we'll use when none is provided. Make it private as we don't intend this to be a public object
_DEFAULT_SYSTEM_PROMPT = """You are an advanced AI assistant designed to provide precise and accurate information.
Your primary goal is to answer queries with the most up-to-date and factual information available.
Focus on delivering clear, concise, and correct responses.
If you're uncertain about any aspect of the query, state your level of confidence and provide the most accurate information you can.
Your responses should prioritize accuracy over all other considerations."""

DEFAULT_SEED = 1337
DEFAULT_JUDGE_MODEL = "gpt-4o"


class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # URL of the OpenAI server where the model shall be hosted.
    base_url: str

    # name of the model to use.
    model_name: str

    # The system prompt to be used when applying the chat template.
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT

    # We do NOT read from OPENAI_API_KEY for the student model for security reasons (e.g. sending the API key to another client)
    # To provide an OpenAI key, you must set it here; else the default is used.
    api_key: str = "no-api-key"

    # "model randomness" aka likelihood of sampling something other than the likeliest token
    temperature: float = 0.0

    # Max amount of tokens to generate.
    max_tokens: int = 768

    # Random seed for reproducibility. Caution: this isn't supported by all model serving runtimes.
    seed: int = DEFAULT_SEED

    @field_validator("temperature")
    @classmethod
    def check_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        return v


class RagasEvaluator(Evaluator):
    # most basic implementation, we just assume that the user will bring the existing model responses
    name = "ragas"

    def __init__(
        self,
        student_model: ModelConfig | None = None,
        run_config: RunConfig | None = None,
    ):
        self.student_model = student_model
        self.run_config = run_config

    def run(
        self,
        dataset: List[Sample] | Path,
        student_model: ModelConfig | None = None,
        run_config: RunConfig | None = None,
    ) -> EvaluationResult:
        """
        Evaluates the quality of model responses against a graded rubric.

        When the `dataset` lacks the `response` field, then `student_model` must be provided
        in order to generate the answers.

        Args:
            dataset (List[Sample] | Path):
                Can be either a list of `Sample` objects or a path to a jsonl file containing
                records matching `Sample`.
            student_model: (StudentModelConfig):
                When this parameter is provided, we'll attempt to use the described model in order to
                generate the responses from the given list of questions.
            run_config (RunConfig | None, optional):
                Configuration to use when running evaluations. If none is provided, then
                a default one is created containing extremely permissive settings when handling
                timeouts. This is because by default, OpenAI tier-1 usage accounts have very high
                rate limits resulting in heavy throttling during evaluations.

        Returns:
            EvaluationResult: The results of all evaluations performed by Ragas
        """
        student_model = student_model if student_model else self.student_model
        run_config = run_config if run_config else self.run_config

        if not dataset:
            raise ValueError(
                "no dataset was provided, please specify the `dataset` argument"
            )

        # ensure we are in the dataframe format
        input_df = None
        if isinstance(dataset, list):
            input_df = DataFrame(dataset)
        elif isinstance(dataset, Path):
            input_df = read_json(dataset, orient="records", lines=True)
        else:
            raise TypeError(f"invalid type of dataset: {type(dataset)}")

        # this should never happen, but pylint is not smart enough to detect it
        assert input_df is not None

        need_to_generate_questions = "response" not in input_df.columns
        if need_to_generate_questions and not student_model:
            raise ValueError(
                "provided dataset doesn't contain the model `response`, but no `student_model` was provided for inference"
            )

        # if the student model was provided then we always generate regardless
        if student_model:
            input_df = self._generate_answers_from_model(input_df, student_model)

        if not run_config:
            # we set extreme timeout/retry values by default since OpenAI tier-1 rate limits
            # are horrible and will result in half of our evaluation results being NaN or 0
            run_config = RunConfig(
                max_retries=120,
                max_wait=7200,
                seed=DEFAULT_SEED,
                timeout=3600,
            )

        metrics = self._get_metrics()
        evaluation_ds = EvaluationDataset.from_pandas(input_df)

        # we will be using gpt-4o for the foreseeable future, we hardcode this
        # for consistency of answers
        critic_lm = ChatOpenAI(model=DEFAULT_JUDGE_MODEL)
        results = evaluate(
            dataset=evaluation_ds,
            batch_size=4,
            run_config=run_config,
            llm=critic_lm,
            metrics=metrics,
            show_progress=True,
        )
        return results

    def _generate_answers_from_model(
        self, questions: DataFrame, student_model: ModelConfig
    ) -> DataFrame:
        """
        Given a DataFrame containing `user_input` columns, generates responses from the given model
        and returns a new DataFrame containing its answers in the `response` column.
        """
        client = get_openai_client(
            model_api_base=student_model.base_url, api_key=student_model.api_key
        )

        # initialize response to write into
        updated_df = questions.copy()
        updated_df["response"] = ""

        for i, qna in updated_df.iterrows():
            messages = [
                student_model.system_prompt,
                qna["user_input"],
            ]
            response = client.chat.completions.create(
                messages=messages,
                model=student_model.model_name,
                # specify the seed so we can at least try to have some reproducibility when the clients support it
                seed=42,
                max_tokens=student_model.max_tokens,
                temperature=student_model.temperature,
            )
            updated_df.at[i, "response"] = response.choices[0].message.content
        return updated_df

    def _get_metrics(self) -> List[Metric]:
        # default set of metrics
        return [
            RubricsScore(
                rubrics=DEFAULT_WITH_REFERENCE_RUBRICS,
            )
        ]
