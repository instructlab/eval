# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from unittest.mock import MagicMock, patch
import unittest

# Third Party
from pandas import DataFrame
from ragas.callbacks import ChainRun
from ragas.dataset_schema import EvaluationDataset, EvaluationResult

# First Party
from instructlab.eval.ragas import ModelConfig, RagasEvaluator, RunConfig


class TestRagasEvaluator(unittest.TestCase):
    def setUp(self):
        # Common setup data for all tests
        self.student_model_response = "Paris"
        self.user_question = "What is the capital of France?"
        self.golden_answer = "The capital of France is Paris."
        self.metric = "mocked-metric"
        self.metric_score = 4.0
        self.base_ds = [
            {
                "user_input": self.user_question,
                "reference": self.golden_answer,
            }
        ]
        self.student_model = ModelConfig(
            model_name="super-jeeves-8x700B",
        )
        self.run_config = RunConfig(max_retries=3, max_wait=60, seed=42, timeout=30)

    @patch("instructlab.eval.ragas.ChatOpenAI")
    @patch("instructlab.eval.ragas.evaluate")
    @patch.object(RagasEvaluator, "_generate_answers_from_model")
    @patch.object(RagasEvaluator, "_get_metrics")
    def test_run_with_dataset(
        self,
        mock_get_metrics: MagicMock,
        mock_generate_answers_from_model: MagicMock,
        mock_evaluate: MagicMock,
        mock_ChatOpenAI: MagicMock,
    ):
        """
        Test case 1: Directly passing a Python list/dict dataset to `RagasEvaluator.run()`.
        """
        # Prepare mocks
        mock_get_metrics.return_value = [self.metric]
        interim_df = DataFrame(
            {
                "user_input": [self.user_question],
                "response": [self.student_model_response],
                "reference": [self.golden_answer],
            }
        )
        mock_generate_answers_from_model.return_value = interim_df
        mocked_evaluation_ds = EvaluationDataset.from_pandas(interim_df)
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
            scores=[{self.metric: self.metric_score}],
            dataset=mocked_evaluation_ds,
            ragas_traces=_unimportant_ragas_traces,
        )

        # Instantiate evaluator
        evaluator = RagasEvaluator()

        # Run test
        result = evaluator.run(
            dataset=self.base_ds,
            student_model=self.student_model,
            run_config=self.run_config,
            student_openai_client=MagicMock(),  # We pass a mock client
        )

        # Assertions
        self.assertIsInstance(result, EvaluationResult)
        mock_generate_answers_from_model.assert_called_once()
        mock_evaluate.assert_called_once()
        # we didn't provide an API key, so it expects to get `api_key=None`
        mock_ChatOpenAI.assert_called_once_with(model="gpt-4o", api_key=None)

    @patch("instructlab.eval.ragas.ChatOpenAI")
    @patch("instructlab.eval.ragas.read_json")
    @patch("instructlab.eval.ragas.evaluate")
    @patch.object(RagasEvaluator, "_generate_answers_from_model")
    @patch.object(RagasEvaluator, "_get_metrics")
    def test_run_with_dataset_via_path(
        self,
        mock_get_metrics: MagicMock,
        mock_generate_answers_from_model: MagicMock,
        mock_evaluate: MagicMock,
        mock_read_json: MagicMock,
        mock_ChatOpenAI: MagicMock,
    ):
        """
        Test case 2: Passing a Path to a JSONL file (containing the dataset) to `RagasEvaluator.run()`.
        """
        # Prepare mocks
        mock_get_metrics.return_value = [self.metric]
        interim_df = DataFrame(
            {
                "user_input": [self.user_question],
                "response": [self.student_model_response],
                "reference": [self.golden_answer],
            }
        )
        mock_generate_answers_from_model.return_value = interim_df
        mocked_evaluation_ds = EvaluationDataset.from_pandas(interim_df)
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
            scores=[{self.metric: self.metric_score}],
            dataset=mocked_evaluation_ds,
            ragas_traces=_unimportant_ragas_traces,
        )

        mock_read_json.return_value = DataFrame(self.base_ds)

        # Instantiate evaluator
        evaluator = RagasEvaluator()

        # Run test
        result = evaluator.run(
            dataset=Path("dummy_path.jsonl"),
            student_model=self.student_model,
            run_config=self.run_config,
            student_openai_client=MagicMock(),
        )

        # Assertions
        self.assertIsInstance(result, EvaluationResult)
        mock_read_json.assert_called_once_with(
            Path("dummy_path.jsonl"), orient="records", lines=True
        )
        mock_generate_answers_from_model.assert_called()
        mock_evaluate.assert_called()

    @patch("instructlab.eval.ragas.ChatOpenAI")
    @patch("instructlab.eval.ragas.read_json")
    @patch("instructlab.eval.ragas.evaluate")
    @patch.object(RagasEvaluator, "_generate_answers_from_model")
    @patch.object(RagasEvaluator, "_get_metrics")
    def test_run_with_instance_attributes(
        self,
        mock_get_metrics: MagicMock,
        mock_generate_answers_from_model: MagicMock,
        mock_evaluate: MagicMock,
        mock_read_json: MagicMock,
        mock_ChatOpenAI: MagicMock,
    ):
        """
        Test case 3: Using `RagasEvaluator` instance attributes for `student_model`, `run_config`,
                     and `student_openai_client` instead of passing them explicitly.
        """
        # Prepare mocks
        mock_get_metrics.return_value = [self.metric]
        interim_df = DataFrame(
            {
                "user_input": [self.user_question],
                "response": [self.student_model_response],
                "reference": [self.golden_answer],
            }
        )
        mock_generate_answers_from_model.return_value = interim_df
        mocked_evaluation_ds = EvaluationDataset.from_pandas(interim_df)
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
            scores=[{self.metric: self.metric_score}],
            dataset=mocked_evaluation_ds,
            ragas_traces=_unimportant_ragas_traces,
        )

        mock_read_json.return_value = DataFrame(self.base_ds)

        # Instantiate evaluator with instance-level configs
        evaluator = RagasEvaluator(
            student_model=self.student_model,
            student_openai_client=MagicMock(),
            run_config=self.run_config,
        )

        # Run test
        result = evaluator.run(dataset=Path("dummy_path.jsonl"))

        # Assertions
        self.assertIsInstance(result, EvaluationResult)
        mock_read_json.assert_called_with(
            Path("dummy_path.jsonl"), orient="records", lines=True
        )
        mock_generate_answers_from_model.assert_called()
        mock_evaluate.assert_called()


if __name__ == "__main__":
    unittest.main()
