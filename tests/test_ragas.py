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
from instructlab.eval.ragas import ModelConfig, RagasEvaluator, RunConfig


class TestRagasEvaluator(unittest.TestCase):
    def test_generate_answers_from_model(self):
        # mock the OpenAI client to always return "london" for chat completions
        user_input = "What is the capital of France?"
        model_response = "London"
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=model_response))]
        mock_client.chat.completions.create.return_value = mock_response

        # get answers
        questions = pd.DataFrame({"user_input": [user_input]})
        student_model = ModelConfig(
            model_name="super-jeeves-8x700B",
        )
        evaluator = RagasEvaluator()
        result_df = evaluator._generate_answers_from_model(
            questions, student_model, mock_client
        )

        # what we expect to see
        expected_df = questions.copy()
        expected_df["response"] = [model_response]

        # perform the assertions
        pd.testing.assert_frame_equal(result_df, expected_df)
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[student_model.system_prompt, user_input],
            model=student_model.model_name,
            seed=42,
            max_tokens=student_model.max_tokens,
            temperature=student_model.temperature,
        )

    @patch("instructlab.eval.ragas.ChatOpenAI")
    @patch("instructlab.eval.ragas.read_json")
    @patch("instructlab.eval.ragas.evaluate")
    @patch.object(RagasEvaluator, "_generate_answers_from_model")
    @patch.object(RagasEvaluator, "_get_metrics")
    def test_run(
        self,
        mock_get_metrics: MagicMock,
        mock_generate_answers_from_model: MagicMock,
        mock_evaluate: MagicMock,
        mock_read_json: MagicMock,
        mock_ChatOpenAI: MagicMock,
    ):
        ########################################################################
        # SETUP EVERYTHING WE NEED FOR THE TESTS
        ########################################################################

        # These are the variables which will control the flow of the test.
        # Since we have to re-construct some Ragas components under the hood,

        student_model_response = "Paris"
        user_question = "What is the capital of France?"
        golden_answer = "The capital of France is Paris."
        metric = "mocked-metric"
        metric_score = 4.0
        base_ds = [{"user_input": user_question, "reference": golden_answer}]
        student_model = ModelConfig(
            model_name="super-jeeves-8x700B",
        )
        run_config = RunConfig(max_retries=3, max_wait=60, seed=42, timeout=30)

        # The following section takes care of mocking function return calls.
        # Ragas is tricky because it has some complex data structures under the hood,
        # so what we have to do is configure the intermediate outputs that we expect
        # to receive from Ragas.

        mock_get_metrics.return_value = [metric]
        interim_df = DataFrame(
            {
                "user_input": [user_question],
                "response": [student_model_response],
                "reference": [golden_answer],
            }
        )
        mock_generate_answers_from_model.return_value = interim_df.copy()
        mocked_evaluation_ds = EvaluationDataset.from_pandas(interim_df)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=student_model_response))
        ]
        mock_client.chat.completions.create.return_value = mock_response

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
            scores=[{metric: metric_score}],
            dataset=mocked_evaluation_ds,
            ragas_traces=_unimportant_ragas_traces,
        )

        ########################################################################
        # Test case: directly passing a dataset
        ########################################################################
        evaluator = RagasEvaluator()
        result = evaluator.run(
            dataset=base_ds,
            student_model=student_model,
            run_config=run_config,
            openai_client=mock_client,
        )

        self.assertIsInstance(result, EvaluationResult)
        mock_generate_answers_from_model.assert_called_once()
        mock_evaluate.assert_called_once()
        mock_ChatOpenAI.assert_called_once_with(model="gpt-4o")

        ########################################################################
        # Test case: passing a dataset in via Path to JSONL file
        ########################################################################
        evaluator = RagasEvaluator()
        mock_read_json.return_value = DataFrame(base_ds)
        result = evaluator.run(
            dataset=Path("dummy_path.jsonl"),
            student_model=student_model,
            run_config=run_config,
            openai_client=mock_client,
        )

        self.assertIsInstance(result, EvaluationResult)
        mock_read_json.assert_called_once_with(
            Path("dummy_path.jsonl"), orient="records", lines=True
        )
        mock_generate_answers_from_model.assert_called()
        mock_evaluate.assert_called()

        ########################################################################
        # Test case: using the instance attributes
        ########################################################################
        evaluator = RagasEvaluator(
            student_model=student_model,
            openai_client=mock_client,
            run_config=run_config,
        )
        mock_read_json.return_value = DataFrame(base_ds)
        result = evaluator.run(dataset=Path("dummy_path.jsonl"))

        self.assertIsInstance(result, EvaluationResult)
        mock_read_json.assert_called_with(
            Path("dummy_path.jsonl"), orient="records", lines=True
        )
        mock_generate_answers_from_model.assert_called()
        mock_evaluate.assert_called()


if __name__ == "__main__":
    unittest.main()
