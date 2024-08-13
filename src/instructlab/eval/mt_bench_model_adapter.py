# SPDX-License-Identifier: Apache-2.0
"""Model adapter registration."""

# Standard
from functools import cache
from typing import List
import abc
import os

# Local
from .logger_config import setup_logger
from .mt_bench_conversation import Conversation, get_conv_template

OPENAI_MODEL_LIST = ("gpt-4",)

logger = setup_logger(__name__)


class BaseModelAdapter:
    """The base and the default model adapter."""

    @abc.abstractmethod
    def match(self, model_path: str) -> bool:
        pass

    @abc.abstractmethod
    def get_default_conv_template(self, model_path: str) -> Conversation:
        pass


# A global registry for all model adapters
model_adapters: List[BaseModelAdapter] = []


def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls())


@cache
def get_model_adapter(model_path: str, default_adapter_name: str) -> BaseModelAdapter:
    """Get a model adapter for a model_path."""
    model_path_basename = os.path.basename(os.path.normpath(model_path))

    default_adapter = None

    # Try the basename of model_path at first
    for adapter in model_adapters:
        if adapter.match(model_path_basename):
            return adapter
        if adapter.match(default_adapter_name) and default_adapter is None:
            default_adapter = adapter

    # Then try the full path
    for adapter in model_adapters:
        if adapter.match(model_path):
            return adapter

    if default_adapter is not None:
        logger.warning(
            "No valid model adapter for %s, defaulting to %s adapter",
            model_path,
            default_adapter_name,
        )
        return default_adapter
    raise ValueError(f"No valid model adapter for {model_path}")


def get_conversation_template(
    model_path: str, default_adapter_name: str
) -> Conversation:
    """Get the default conversation template."""
    adapter = get_model_adapter(model_path, default_adapter_name)
    return adapter.get_default_conv_template(model_path)


class ChatGPTAdapter(BaseModelAdapter):
    """The model adapter for ChatGPT"""

    def match(self, model_path: str):
        return model_path in OPENAI_MODEL_LIST

    def get_default_conv_template(self, model_path: str) -> Conversation:
        if "browsing" in model_path:
            return get_conv_template("api_based_default")
        return get_conv_template("chatgpt")


class MistralAdapter(BaseModelAdapter):
    """The model adapter for Mistral AI models"""

    def match(self, model_path: str):
        model_path = model_path.lower()
        return (
            "mistral" in model_path
            or "mixtral" in model_path
            or "prometheus" in model_path
        )

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("mistral")


class LabradoriteAdapter(BaseModelAdapter):
    """The model adapter for ibm/labradorite-13b"""

    def match(self, model_path: str):
        return "labradorite" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("labrador-chat")


class MerliniteAdapter(BaseModelAdapter):
    """The model adapter for ibm/merlinite-7b and instructlab/merlinite-7b-lab"""

    def match(self, model_path: str):
        return "merlinite" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("ibm-generic")


class GraniteAdapter(BaseModelAdapter):
    """The model adapter for instructlab/granite-7b-lab"""

    def match(self, model_path: str):
        model_path = model_path.lower()
        return (
            "granite" in model_path
            and "granite-old" not in model_path
            and "granite-chat" not in model_path
            and "granite-code" not in model_path
        )

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("ibm-generic")


class LabradorAdapter(BaseModelAdapter):
    """The model adapter for ibm/labradorite-13b"""

    def match(self, model_path: str):
        model_path = model_path.lower()
        return ("granite-chat" in model_path) or (
            "labrador" in model_path and "labradorite" not in model_path
        )

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("granite-chat")


# Note: the registration order matters.
# The one registered earlier has a higher matching priority.
register_model_adapter(MistralAdapter)
register_model_adapter(LabradoriteAdapter)
register_model_adapter(MerliniteAdapter)
register_model_adapter(GraniteAdapter)
register_model_adapter(LabradorAdapter)
register_model_adapter(ChatGPTAdapter)
