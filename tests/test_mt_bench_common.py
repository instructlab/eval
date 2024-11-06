# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest import mock

# First Party
from instructlab.eval.mt_bench_common import Judge, check_data

CHECK_DATA_EXAMPLE_QUESTIONS = [
    {
        "question_id": 81,
        "category": "writing",
        "turns": [
            "Fake question",
            "Fake question",
        ],
    },
    {
        "question_id": 101,
        "category": "reasoning",
        "turns": [
            "Fake question",
            "Fake question",
        ],
    },
]
CHECK_DATA_EXAMPLE_MODEL_ANSWERS = {
    "granite-7b-lab": {
        81: {
            "question_id": 81,
            "answer_id": "c4j9vPyHM8w3JHPGohrJQG",
            "model_id": "granite-7b-lab",
            "choices": [
                {
                    "index": 0,
                    "turns": [
                        "Fake answer",
                        "Fake answer",
                    ],
                }
            ],
            "tstamp": 1730816201.883507,
        },
        101: {
            "question_id": 101,
            "answer_id": "kaQw7Fj2SDeE2VfvU25FJ4",
            "model_id": "granite-7b-lab",
            "choices": [
                {
                    "index": 0,
                    "turns": [
                        "Fake answer",
                        "Fake answer",
                    ],
                }
            ],
            "tstamp": 1730816166.3719094,
        },
    }
}
CHECK_DATA_EXAMPLE_REFERENCE_ANSWERS = {
    "merlinite-7b-lab": {
        101: {
            "question_id": 101,
            "answer_id": "TFomieEmmAgdeCkvmuvwbc",
            "model_id": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "turns": [
                        "Fake answer",
                        "Fake answer",
                    ],
                }
            ],
            "tstamp": 1686286924.844282,
        },
        102: {
            "question_id": 102,
            "answer_id": "hLH8WozvaB88bb5vV224H4",
            "model_id": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "turns": [
                        "Fake answer",
                        "Fake answer",
                    ],
                }
            ],
            "tstamp": 1686286937.7164738,
        },
    }
}

CHECK_DATA_EXAMPLE_MODELS = ["granite-7b-lab"]
CHECK_DATA_EXAMPLE_JUDGES = {
    "default": Judge(
        model_name="merlinite-7b-lab",
        prompt_template={
            "name": "single-v1",
            "type": "single",
            "system_prompt": "Fake prompt",
            "prompt_template": "Fake prompt",
            "description": "Prompt for general questions",
            "category": "general",
            "output_format": "[[rating]]",
        },
        ref_based=False,
        multi_turn=False,
    ),
    "math": Judge(
        model_name="merlinite-7b-lab",
        prompt_template={
            "name": "single-math-v1",
            "type": "single",
            "system_prompt": "Fake prompt",
            "prompt_template": "Fake prompt",
            "description": "Prompt for general questions",
            "category": "math",
            "output_format": "[[rating]]",
        },
        ref_based=True,
        multi_turn=False,
    ),
    "default-mt": Judge(
        model_name="merlinite-7b-lab",
        prompt_template={
            "name": "single-v1-multi-turn",
            "type": "single",
            "system_prompt": "Fake prompt",
            "prompt_template": "Fake prompt",
            "description": "Prompt for general questions",
            "category": "general",
            "output_format": "[[rating]]",
        },
        ref_based=False,
        multi_turn=True,
    ),
    "math-mt": Judge(
        model_name="merlinite-7b-lab",
        prompt_template={
            "name": "single-math-v1-multi-turn",
            "type": "single",
            "system_prompt": "Fake prompt",
            "prompt_template": "Fake prompt",
            "description": "Prompt for general questions",
            "category": "math",
            "output_format": "[[rating]]",
        },
        ref_based=True,
        multi_turn=True,
    ),
}


def test_check_data():
    check_data(
        CHECK_DATA_EXAMPLE_QUESTIONS,
        CHECK_DATA_EXAMPLE_MODEL_ANSWERS,
        CHECK_DATA_EXAMPLE_REFERENCE_ANSWERS,
        CHECK_DATA_EXAMPLE_MODELS,
        CHECK_DATA_EXAMPLE_JUDGES,
    )

    try:
        check_data(
            CHECK_DATA_EXAMPLE_QUESTIONS,
            {"granite-7b-lab": {}},
            CHECK_DATA_EXAMPLE_REFERENCE_ANSWERS,
            CHECK_DATA_EXAMPLE_MODELS,
            CHECK_DATA_EXAMPLE_JUDGES,
        )
    except Exception as e:
        assert "Missing model granite-7b-lab's answer to Question" in str(e)
    else:
        assert False, "Didn't fail with missing model answer"

    try:
        check_data(
            CHECK_DATA_EXAMPLE_QUESTIONS,
            CHECK_DATA_EXAMPLE_MODEL_ANSWERS,
            {"merlinite-7b-lab": {}},
            CHECK_DATA_EXAMPLE_MODELS,
            CHECK_DATA_EXAMPLE_JUDGES,
        )
    except Exception as e:
        assert "Missing reference answer to Question" in str(e)
    else:
        assert False, "Didn't fail with missing reference answer"
