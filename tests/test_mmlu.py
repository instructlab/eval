# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest import mock
from unittest.mock import patch
import os

# First Party
from instructlab.eval.mmlu import MMLUBranchEvaluator, MMLUEvaluator

MMLU_EXAMPLE_OUTPUT = {
    "results": {
        "mmlu_astronomy": {
            "alias": "astronomy",
            "acc,none": 0.5592105263157895,
            "acc_stderr,none": 0.04040311062490436,
        },
        "mmlu_anatomy": {
            "alias": "anatomy",
            "acc,none": 0.4444444444444444,
            "acc_stderr,none": 0.04292596718256981,
        },
        "mmlu_abstract_algebra": {
            "alias": "abstract_algebra",
            "acc,none": 0.35,
            "acc_stderr,none": 0.047937248544110196,
        },
    },
}

MODEL_EXAMPLE = "instructlab/granite-7b-lab"


def assert_example_mmlu_individual_scores(overall_score, individual_scores):
    assert round(overall_score, 2) == 0.45
    assert individual_scores == {
        "mmlu_abstract_algebra": {"score": 0.35, "stderr": 0.047937248544110196},
        "mmlu_anatomy": {"score": 0.4444444444444444, "stderr": 0.04292596718256981},
        "mmlu_astronomy": {"score": 0.5592105263157895, "stderr": 0.04040311062490436},
    }


@patch(
    "instructlab.eval.mmlu.AbstractMMLUEvaluator._simple_evaluate_with_error_handling",
    return_value=MMLU_EXAMPLE_OUTPUT,
)
def test_mmlu_branch(eval_mock):
    tasks_dir = f"{os.path.dirname(os.path.realpath(__file__))}/testdata/sdg"
    tasks = ["mmlu_pr"]
    mmlu = MMLUBranchEvaluator(
        model_path=MODEL_EXAMPLE, tasks_dir=tasks_dir, tasks=tasks
    )
    overall_score, individual_scores = mmlu.run()

    assert_example_mmlu_individual_scores(overall_score, individual_scores)
    eval_mock.assert_called()


@patch(
    "instructlab.eval.mmlu.AbstractMMLUEvaluator._simple_evaluate_with_error_handling",
    return_value=MMLU_EXAMPLE_OUTPUT,
)
def test_mmlu(eval_mock):
    tasks = ["mmlu_anatomy", "mmlu_astronomy", "mmlu_algebra"]
    mmlu = MMLUEvaluator(model_path=MODEL_EXAMPLE, tasks=tasks)
    overall_score, individual_scores = mmlu.run()

    eval_mock.assert_called()
    assert_example_mmlu_individual_scores(overall_score, individual_scores)
