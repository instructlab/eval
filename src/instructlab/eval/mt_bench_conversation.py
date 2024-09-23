# SPDX-License-Identifier: Apache-2.0
"""
Conversation prompt templates.
"""

# Standard
from enum import IntEnum, auto
from typing import Dict, List, Tuple, Union
import dataclasses


class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    DEFAULT = auto()


@dataclasses.dataclass
class Conversation:
    # pylint: disable=too-many-instance-attributes
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: Tuple[str, str] = ("USER", "ASSISTANT")
    # All messages. Each item is (role, message).
    # Each message is either a string or a tuple of (string, List[image_url]).
    messages: List[List[str | None]] = dataclasses.field(default_factory=list)
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str | None = "\n"
    sep2: str | None = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] | None = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] | None = None

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def get_system_message(self):
        """return the system message."""
        return self.system_message

    def append_message(self, role: str, message: str | None):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        if self.system_message == "":
            ret = []
        else:
            ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.extract_text_from_messages(),
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# An empty template for raw conversation.
register_conv_template(
    Conversation(
        name="raw",
        system_message="",
        roles=("", ""),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
    )
)


# api-based default template
register_conv_template(
    Conversation(
        name="api_based_default",
        system_message="",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
    )
)


# ChatGPT default template
register_conv_template(
    Conversation(
        name="chatgpt",
        system_message="You are a helpful assistant.",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
    )
)

# Mistral template
# source: https://docs.mistral.ai/llm/mistral-instruct-v0.1#chat-template
register_conv_template(
    Conversation(
        name="mistral",
        system_template="[INST] {system_message}\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2="</s>",
    )
)

register_conv_template(
    Conversation(
        name="labrador-chat",
        system_template="<|system|>\n{system_message}",
        system_message="""You are Labrador, an AI language model developed by IBM DMF (Data Model Factory) Alignment Team. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior. You always respond to greetings (for example, hi, hello, g'day, morning, afternoon, evening, night, what's up, nice to meet you, sup, etc) with "Hello! I am Labrador, created by the IBM DMF Alignment Team. How can I help you today?". Please do not say anything else and do not start a conversation.""",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n",
        stop_str="<|endoftext|>",
    )
)

register_conv_template(
    Conversation(
        name="ibm-generic",
        system_template="<|system|>\n{system_message}",
        system_message="""You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.""",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n",
        stop_str="<|endoftext|>",
    )
)

register_conv_template(
    Conversation(
        name="granite-chat",
        system_template="<|system|>\n{system_message}",
        system_message="""You are Granite Chat, an AI language model developed by IBM. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.""",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n",
        stop_str="<|endoftext|>",
    )
)
