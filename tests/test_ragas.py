# # SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from unittest.mock import MagicMock, patch
import unittest

# Third Party
from pandas import DataFrame
from ragas.callbacks import ChainRun
from ragas.dataset_schema import EvaluationDataset, EvaluationResult
import pandas as pd

# First Party
from instructlab.eval.ragas import ModelConfig, RagasEvaluator, RunConfig, Sample


class TestRagasEvaluator(unittest.TestCase):
    @patch("instructlab.eval.ragas.get_openai_client")
    def test_generate_answers_from_model(self, mock_get_openai_client):
        # mock the OpenAI client to always return "london" for chat completions
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "London"
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_openai_client.return_value = mock_client

        # get answers
        questions = pd.DataFrame({"user_input": ["What is the capital of France?"]})
        student_model = ModelConfig(
            base_url="https://your.model.endpoint.com",
            model_name="jeeves-512B",
            api_key="test-api-key",
        )
        evaluator = RagasEvaluator()
        result_df = evaluator._generate_answers_from_model(questions, student_model)

        # what we expect to see
        expected_df = questions.copy()
        expected_df["response"] = ["London"]

        # perform the assertions
        pd.testing.assert_frame_equal(result_df, expected_df)
        mock_get_openai_client.assert_called_once_with(
            model_api_base=student_model.base_url, api_key=student_model.api_key
        )
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[student_model.system_prompt, "What is the capital of France?"],
            model=student_model.model_name,
            seed=42,
            max_tokens=student_model.max_tokens,
            temperature=student_model.temperature,
        )

    @patch("instructlab.eval.ragas.read_json")
    @patch("instructlab.eval.ragas.evaluate")
    @patch("instructlab.eval.ragas.ChatOpenAI")
    @patch.object(RagasEvaluator, "_generate_answers_from_model")
    @patch.object(RagasEvaluator, "_get_metrics")
    def test_run(
        self,
        mock_get_metrics: MagicMock,
        mock_generate_answers_from_model: MagicMock,
        mock_ChatOpenAI: MagicMock,
        mock_evaluate: MagicMock,
        mock_read_json: MagicMock,
    ):
        ########################################################################
        # SETUP EVERYTHING WE NEED FOR THE TESTS
        ########################################################################

        # These are the variables which will control the flow of the test.
        # Since we have to re-construct some Ragas components under the hood,

        student_model_response = "Paris"
        user_question = "What is the capital of France?"
        golden_answer = "The capital of France is Paris."
        base_ds = [{"user_input": user_question, "reference": golden_answer}]
        mocked_metric = "mocked-metric"
        mocked_metric_score = 4.0

        # The following section takes care of mocking function return calls.
        # Ragas is tricky because it has some complex data structures under the hood,
        # so what we have to do is configure the intermediate outputs that we expect
        # to receive from Ragas.

        mock_get_metrics.return_value = [mocked_metric]
        interim_df = DataFrame(
            {
                "user_input": [user_question],
                "response": [student_model_response],
                "reference": [golden_answer],
            }
        )
        mock_generate_answers_from_model.return_value = interim_df.copy()
        mocked_evaluation_ds = EvaluationDataset.from_pandas(interim_df)
        mock_ChatOpenAI.return_value = MagicMock()

        # Ragas requires this value to instantiate an EvaluationResult object, so we must provide it.
        # It isn't functionally used for our purposes though.

        _unimportant_ragas_traces = {
            "default": ChainRun(
                run_id="42",
                parent_run_id=None,
                name="root",
                inputs={"system": "null", "user": "null"},
                outputs={"assistant": "null"},
                metadata={"user_id": 1337},
            )
        }
        mock_evaluate.return_value = EvaluationResult(
            scores=[{mocked_metric: mocked_metric_score}],
            dataset=mocked_evaluation_ds,
            ragas_traces=_unimportant_ragas_traces,
        )

        ########################################################################
        # Run the tests
        ########################################################################

        # Configure all other inputs that Ragas does not depend on for proper mocking
        student_model = ModelConfig(
            base_url="https://api.openai.com",
            model_name="pt-3.5-turbo",
            api_key="test-api-key",
        )
        run_config = RunConfig(max_retries=3, max_wait=60, seed=42, timeout=30)
        evaluator = RagasEvaluator()

        ########################################################################
        # Test case: directly passing a dataset
        ########################################################################
        result = evaluator.run(
            dataset=base_ds, student_model=student_model, run_config=run_config
        )

        self.assertIsInstance(result, EvaluationResult)
        mock_generate_answers_from_model.assert_called_once()
        mock_evaluate.assert_called_once()
        mock_ChatOpenAI.assert_called_once_with(model="gpt-4o")

        ########################################################################
        # Test case: passing a dataset in via Path to JSONL file
        ########################################################################
        mock_read_json.return_value = DataFrame(base_ds)
        result = evaluator.run(
            dataset=Path("dummy_path.jsonl"),
            student_model=student_model,
            run_config=run_config,
        )

        self.assertIsInstance(result, EvaluationResult)
        mock_read_json.assert_called_once_with(
            Path("dummy_path.jsonl"), orient="records", lines=True
        )
        mock_generate_answers_from_model.assert_called()
        mock_evaluate.assert_called()


if __name__ == "__main__":
    unittest.main()
