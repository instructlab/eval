# # SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, List, Optional, TypedDict
import os

# Third Party
from langchain_community.chat_models import ChatOpenAI
from openai import Client as OpenAIClient
from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, Field
from ragas.evaluation import EvaluationDataset, EvaluationResult, RunConfig, evaluate
from ragas.metrics import Metric
from ragas.metrics._domain_specific_rubrics import (  # the rubrics we must instantiate are located inside of a file marked as private
    RubricsScore,
)
import openai

# First Party
from instructlab.eval import exceptions

# Local
from .evaluator import Evaluator
from .logger_config import setup_logger

logger = setup_logger(__name__)

OLD_DEFAULT_WITH_REFERENCE_RUBRICS = {
    "score1_description": "The response is incorrect, irrelevant, or does not align with the ground truth.",
    "score2_description": "The response partially matches the ground truth but includes significant errors, omissions, or irrelevant information.",
    "score3_description": "The response generally aligns with the ground truth but may lack detail, clarity, or have minor inaccuracies.",
    "score4_description": "The response is mostly accurate and aligns well with the ground truth, with only minor issues or missing details.",
    "score5_description": "The response is fully accurate, aligns completely with the ground truth, and is clear and detailed.",
}


class Sample(TypedDict):
    """
    TypedDict of a sample that we accept when doing eval with Ragas used
    prototyping and getting started using the library quickly. We specifically
    use TypedDict here to be flexible with the input data we accept.
    """

    # question
    user_input: str

    # model answer
    response: Optional[str]

    # golden answer
    reference: str


# default system prompt we'll use when none is provided.
DEFAULT_SYSTEM_PROMPT = "You are an advanced AI assistant designed to provide precise and accurate information. Your primary goal is to answer queries with the most up-to-date and factual information available. Focus on delivering clear, concise, and correct responses. If you're uncertain about any aspect of the query, state your level of confidence and provide the most accurate information you can. Your responses should prioritize accuracy over all other considerations."

DEFAULT_SEED = 1337
# we will be using gpt-4o for the foreseeable future, we hardcode this
# for consistency of answers
DEFAULT_JUDGE_MODEL = "gpt-4o"

# we set extreme timeout/retry values by default since OpenAI tier-1 rate limits
# are horrible and will result in half of our evaluation results being NaN or 0
DEFAULT_RUN_CONFIG = RunConfig(
    max_retries=120,
    max_wait=7200,
    seed=DEFAULT_SEED,
    timeout=3600,
)


class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # name of the model to use.
    model_name: str

    # The system prompt to be used when applying the chat template.
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # "model randomness" aka likelihood of sampling something other than the likeliest token
    temperature: float = Field(default=0.0, le=1.0, ge=0.0)

    # Max amount of tokens to generate.
    max_tokens: int = 768

    # Random seed for reproducibility. Caution: this isn't supported by all model serving runtimes.
    seed: int = 42


class RagasEvaluator(Evaluator):
    """
    Evaluator for Domain Knowledge Evaluation

    Attributes:
        model_config                Configuration for model when it's
                                    generating responses.
        run_config                  Configuration to use when running
                                    evaluations. If none is provided, then
                                    a default one is created containing
                                    extremely permissive settings when handling
                                    timeouts. This is because by default, OpenAI
                                    tier-1 usage accounts have very high
                                    rate limits resulting in heavy throttling
                                    during evaluations.
        model_openai_client         The client to use when generating questions
                                    from the model, must be compatible with the
                                    OpenAI API.
        judge_model_name            Name of the OpenAI model to use as the
                                    judge model. Defaults to "gpt-4o" when none
                                    is specified.
        judge_openai_api_key        The API key to use for evaluating the given
                                    dataset. When this isn't provided,
                                    `OPENAI_API_KEY` is read instead.
    """

    name = "ragas"

    def __init__(
        self,
        judge_openai_api_key: str | None = None,
        judge_model_name: str = DEFAULT_JUDGE_MODEL,
        run_config: RunConfig | None = DEFAULT_RUN_CONFIG,
    ):
        self.judge_model_name = judge_model_name
        self.judge_openai_api_key = judge_openai_api_key
        self.run_config = run_config

    @staticmethod
    def validate_dataset(df: DataFrame, requires_response: bool = False):
        """
        Validates whether or not the given `df` is a valid dataset of `Sample`
        objects.

        Args:
            df (DataFrame):             DataFrame containing the dataset to be
                                        evaluated.
            requires_response (bool):   Whether or not a "response" column is
                                        reqired in the DataFrame.
        """
        required_columns = set()
        optional_columns = set()

        if requires_response:
            required_columns = {"user_input", "reference", "response"}
        else:
            required_columns = {"user_input", "reference"}
            optional_columns = {"response"}

        # combine both sets into valid_columns set
        valid_columns = required_columns | optional_columns
        input_columns = set(df.columns)

        missing_required_columns = not required_columns.issubset(input_columns)
        has_extra_columns = not input_columns.issubset(valid_columns)

        if missing_required_columns:
            # set of required columns that are missing
            missing_columns = required_columns - input_columns
            raise ValueError(
                f"Dataset requires the following columns: {', '.join(required_columns)}. Missing columns: {', '.join(missing_columns)}"
            )

        if has_extra_columns:
            # set of columns that are not valid
            invalid_columns = input_columns - valid_columns
            raise ValueError(
                f"Dataset can only have the following columns: {', '.join(valid_columns)}. Invalid columns provided were: {', '.join(invalid_columns)}"
            )

    def run(
        self,
        dataset: List[Sample] | DataFrame,
        run_config: RunConfig | None = None,
        judge_model_name: str | None = None,
        judge_openai_api_key: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluates the quality of model responses against a graded rubric.
        The `dataset` must have 'user_input, 'reference' and `response` fields.

        Args:
            dataset (List[Sample] | DataFrame):
                Can be either a list of `Sample` objects or a Pandas DataFrame
                that has records matching `Sample`.
            run_config (RunConfig | None, optional):
                Configuration to use when running evaluations. If none is provided, then
                a default one is created containing extremely permissive settings when handling
                timeouts. This is because by default, OpenAI tier-1 usage accounts have very high
                rate limits resulting in heavy throttling during evaluations.
            judge_model_name (str | None, optional):
                Name of the OpenAI model to use as the judge model. Defaults to "gpt-4o" when none is specified.
            judge_openai_api_key (str | None, optional):
                The API key to use for evaluating the given dataset. When this isn't provided, `OPENAI_API_KEY` is read instead.

        Returns:
            EvaluationResult: The results of all evaluations performed by Ragas
        """
        # judge_model_name and judge_openai_api_key will always call back on defaults
        judge_model_name = (
            judge_model_name if judge_model_name else self.judge_model_name
        )
        run_config = run_config if run_config else self.run_config

        if judge_openai_api_key is None:
            logger.debug(
                "Judge OpenAI API key not provided. Using environment variable 'OPEN_AI_API_KEY' for judge API key."
            )

            judge_openai_api_key = os.environ.get("OPENAI_API_KEY", None)
            if judge_openai_api_key is None:
                raise EnvironmentError(
                    "Environment variable 'OPENAI_API_KEY' not found. 'OPENAI_API_KEY' must be set to run the judge model in LLMaaJ."
                )

        if isinstance(dataset, list):
            input_df = DataFrame(dataset)
        elif isinstance(dataset, DataFrame):
            input_df = dataset
        else:
            raise TypeError(
                f"Invalid type of dataset: {type(dataset)}. Dataset must be of type List[Sample] or a Pandas DataFrame"
            )

        # this should never happen, but pylint is not smart enough to detect it
        # ensure we are in the dataframe format
        if TYPE_CHECKING:
            assert input_df is not None

        if not self.is_judge_model_name_valid(judge_model_name, judge_openai_api_key):
            raise ValueError("Judge model name must be a valid OpenAI GPT model")

        # ensure the dataframe is in the format we expect it
        self.validate_dataset(input_df, requires_response=True)

        metrics = self._get_metrics()
        evaluation_ds = EvaluationDataset.from_pandas(input_df)

        critic_lm = ChatOpenAI(model=judge_model_name, api_key=judge_openai_api_key)
        results = evaluate(
            dataset=evaluation_ds,
            batch_size=4,
            run_config=run_config,
            llm=critic_lm,
            metrics=metrics,
            show_progress=True,
        )
        return results

    @staticmethod
    def generate_answers_from_model(
        questions: DataFrame,
        model_config: ModelConfig,
        openai_client: OpenAIClient,
    ) -> DataFrame:
        """
        Given a DataFrame containing `user_input` columns, generates responses from the given model
        and returns a new DataFrame containing its answers in the `response` column.

        Args:
            questions: (DataFrame):
                Questions and refernce answers to be returned with the responses from the model
            model_config: (ModelConfig):
                Configuration settings for the model when getting responses.
            openai_client (openai.Client | None, optional):
                The client to use when generating questions from the model, must be compatible with the OpenAI API.
        Returns:
            DataFrame with user_input, reference, and response columns. Responses for the user_input from the model
        """
        # initialize response to write into
        updated_df = questions.copy()
        updated_df["response"] = ""

        for i, qna in updated_df.iterrows():
            try:
                messages: List[ChatCompletionMessageParam] = [
                    {
                        "role": "system",
                        "content": model_config.system_prompt,
                    },
                    {"role": "user", "content": qna["user_input"]},
                ]
                response = openai_client.chat.completions.create(
                    messages=messages,
                    model=model_config.model_name,
                    # specify the seed so we can at least try to have some reproducibility when the clients support it
                    seed=model_config.seed,
                    max_tokens=model_config.max_tokens,
                    temperature=model_config.temperature,
                )
                updated_df.at[i, "response"] = response.choices[0].message.content
            except openai.OpenAIError as e:
                raise exceptions.ModelResponseGenerationError(e)
        return updated_df

    @staticmethod
    def is_judge_model_name_valid(judge_model_name: str, api_key: str | None) -> bool:
        """
        Evaluates the quality of model responses against a graded rubric.
        The `dataset` must have 'user_input, 'reference' and `response` fields.

        Args:
            judge_model_name (str): Name of the judge model to validate.
            judge_openai_api_key (str | None, optional):
                The API key to use for evaluating the given dataset. When this isn't provided, `OPENAI_API_KEY` is read instead.

        Returns:
            bool: Whether or not the judge name is valid. The judge name should be a valid GPT model on OpenAI and start with "gpt-".
        """
        try:
            client = OpenAI(
                base_url="https://api.openai.com/v1/",
                api_key=api_key,
            )
            models = client.models.list()
        except OpenAIError as e:
            raise exceptions.ModelListGenerationError(e)

        model_ids = [model.id for model in models.data if model.id.startswith("gpt-")]
        return judge_model_name in model_ids

    @staticmethod
    def _get_metrics() -> List[Metric]:
        """
        Evaluates the quality of model responses against a graded rubric.
        The `dataset` must have 'user_input, 'reference' and `response` fields.

        Args:
            None
        Returns:
            List[Metric]: A list with a RubricsScore() initialized with the
                          proper rubric to grade responses on.
        """
        return [
            RubricsScore(
                rubrics=OLD_DEFAULT_WITH_REFERENCE_RUBRICS,
            )
        ]
