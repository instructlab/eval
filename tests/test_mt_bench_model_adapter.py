# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from instructlab.eval.mt_bench_model_adapter import (
    GraniteAdapter,
    MistralAdapter,
    get_conversation_template,
    get_model_adapter,
)

MISTRAL_DEFAULT_MODEL_NAME = "mistral"
EXAMPLE_MISTRAL_MODEL_PATHS = [
    "mistral",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "/cache/instructlab/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    "prometheus-eval/prometheus-8x7b-v2.0",
    "/cache/instructlab/models/prometheus-eval/prometheus-8x7b-v2.0",
]

GRANITE_DEFAULT_MODEL_NAME = "granite"
EXAMPLE_GRANITE_MODEL_PATHS = [
    "granite",
    "instructlab/granite-7b-lab",
    "/cache/instructlab/models/instructlab/granite-7b-lab.gguf",
    "instructlab/granite-8b-lab",
]

TEST_TUPLES = [
    (
        MISTRAL_DEFAULT_MODEL_NAME,
        EXAMPLE_MISTRAL_MODEL_PATHS,
        MistralAdapter,
        MISTRAL_DEFAULT_MODEL_NAME,
    ),
    (
        GRANITE_DEFAULT_MODEL_NAME,
        EXAMPLE_GRANITE_MODEL_PATHS,
        GraniteAdapter,
        "ibm-generic",
    ),
]


def test_get_model_adapter():
    for model, model_paths, adapter, _ in TEST_TUPLES:
        for model_path in model_paths:
            assert isinstance(get_model_adapter(model_path, model), adapter)

    # Test default adapter overrides as expected
    assert isinstance(get_model_adapter("", MISTRAL_DEFAULT_MODEL_NAME), MistralAdapter)


def test_get_model_adapter_not_found():
    with pytest.raises(ValueError):
        get_model_adapter("unknown", "unknown")


def test_get_conversation_template():
    for model, model_paths, _, conv_template_name in TEST_TUPLES:
        for model_path in model_paths:
            assert (
                conv_template_name == get_conversation_template(model_path, model).name
            )
