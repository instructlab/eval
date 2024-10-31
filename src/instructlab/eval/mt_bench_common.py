# SPDX-License-Identifier: Apache-2.0
"""
Common data structures and utilities.
"""

# Standard
from typing import Optional, TypedDict
import ast
import dataclasses
import json
import os
import re
import time

# Third Party
import httpx
import openai

# First Party
from instructlab.eval import exceptions

# Local
from .logger_config import setup_logger
from .mt_bench_conversation import Conversation
from .mt_bench_model_adapter import get_conversation_template

logger = setup_logger(__name__)

# API setting constants
API_MAX_RETRY = 4
API_RETRY_SLEEP = 4
API_ERROR_OUTPUT = "$ERROR$"

# Categories that need reference answers
NEED_REF_CATS = ["math", "reasoning", "coding", "arena-hard-200", "taxonomy"]

# Extract scores from judgments
one_score_pattern = re.compile(r"\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile(r"\[(\d+\.?\d*)\]")

# Sampling temperature configs for categories
temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}


@dataclasses.dataclass
class Judge:
    model_name: str
    prompt_template: dict
    ref_based: bool = False
    multi_turn: bool = False


@dataclasses.dataclass
class MatchSingle:
    question: dict
    model: str
    answer: dict
    judge: Judge
    ref_answer: Optional[dict] = None
    multi_turn: bool = False


def bench_dir(output_dir, bench_name, branch) -> str:
    b_dir = os.path.join(output_dir, bench_name)
    if branch is not None:
        b_dir = os.path.join(b_dir, branch)
    return b_dir


def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r", encoding="utf-8") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions


def load_model_answers(answer_file: str, model_name: str | None = None) -> dict:
    """Load model answers from a single answer file

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    logger.debug(locals())
    file_model_name = os.path.splitext(os.path.basename(answer_file))[0]
    model_answers = {model_name or file_model_name: _load_answers(answer_file)}
    return model_answers


def _load_answers(answer_file):
    answers = {}
    with open(answer_file, encoding="utf-8") as fin:
        for line in fin:
            l = json.loads(line)
            answers[l["question_id"]] = l
    return answers


def load_judge_prompts(prompt_file: str) -> dict:
    """Load judge prompts.

    The return value is a python dict of type:
    Dict[judge_name: str -> dict]
    """
    logger.debug(locals())
    prompts = {}
    with open(prompt_file, encoding="utf-8") as fin:
        for line in fin:
            l = json.loads(line)
            prompts[l["name"]] = l
    return prompts


def run_judge_single(
    question,
    answer,
    judge,
    ref_answer,
    openai_client,
    multi_turn=False,
    judgment=None,
    merge_system_user_message=False,
):
    kwargs = {}
    model = judge.model_name
    if ref_answer is not None:
        kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
        if multi_turn:
            kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]

    if multi_turn:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question_1=question["turns"][0],
            question_2=question["turns"][1],
            answer_1=answer["choices"][0]["turns"][0],
            answer_2=answer["choices"][0]["turns"][1],
            **kwargs,
        )
    else:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question=question["turns"][0],
            answer=answer["choices"][0]["turns"][0],
            **kwargs,
        )

    rating = -1

    system_prompt = judge.prompt_template["system_prompt"]
    conv = get_conversation_template(model, "mixtral")
    conv.set_system_message(system_prompt)
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)

    if judgment is None:
        judgment = chat_completion_openai(
            openai_client,
            model,
            conv,
            temperature=0,
            max_tokens=2048,
            merge_system_user_message=merge_system_user_message,
        )

    if judge.prompt_template["output_format"] == "[[rating]]":
        match = re.search(one_score_pattern, judgment)
        if not match:
            match = re.search(one_score_pattern_backup, judgment)

        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = -1
            logger.debug(
                "Received invalid judgment for question %s with judgment: %s",
                question["question_id"],
                judgment,
            )
    else:
        raise ValueError(
            f"invalid output format: {judge.prompt_template['output_format']}"
        )

    return rating, user_prompt, judgment


def play_a_match_single(
    openai_client, match: MatchSingle, output_file: str, merge_system_user_message: bool
) -> dict:
    question, model, answer, judge, ref_answer, multi_turn = (
        match.question,
        match.model,
        match.answer,
        match.judge,
        match.ref_answer,
        match.multi_turn,
    )

    if judge.prompt_template["type"] == "single":
        judgment = None
        retval = run_judge_single(
            question,
            answer,
            judge,
            ref_answer,
            openai_client,
            multi_turn=multi_turn,
            judgment=judgment,
            merge_system_user_message=merge_system_user_message,
        )
        score, user_prompt, judgment = retval

        question_id = question["question_id"]
        turn = 1 if not multi_turn else 2
        result = {
            "question_id": question_id,
            "model": model,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "user_prompt": user_prompt,
            "judgment": judgment,
            "score": score,
            "turn": turn,
            "tstamp": time.time(),
        }

    else:
        raise ValueError(f"invalid judge type: {judge.prompt_template['type']}")

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(result) + "\n")

    return result


def _is_fatal_openai_error(e: openai.OpenAIError) -> bool:
    return isinstance(
        e,
        (
            openai.APIConnectionError,
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
        ),
    )


# TODO: copied from instructlab (cli) utils module; consolidate somewhere?
class Message(TypedDict):
    """
    Represents a message within an AI conversation.
    """

    content: str
    # one of: "user", "assistant", or "system"
    role: str


def _get_messages(conv: Conversation, merge_system_user_message: bool) -> list[Message]:
    messages = conv.to_openai_api_messages()
    if (
        (merge_system_user_message or conv.name == "mistral")
        and messages[0]["role"] == "system"
        and messages[1]["role"] == "user"
    ):
        messages[1]["content"] = f"{messages[0]['content']}\n{messages[1]['content']}"
        return messages[1:]
    return messages


def chat_completion_openai(
    openai_client,
    model,
    conv: Conversation,
    temperature,
    max_tokens,
    merge_system_user_message: bool = False,
) -> str:
    output = None
    messages = _get_messages(conv, merge_system_user_message)

    for i in range(API_MAX_RETRY):
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response.choices[0].message.content
            break
        except (
            # retry won't fix these errors
            openai.BadRequestError,  # 400
            openai.UnprocessableEntityError,  # 422
        ) as e:
            logger.debug(e)
            return API_ERROR_OUTPUT  # immediately soft fail
        except (
            # retry may help with these errors
            openai.APIConnectionError,
            openai.RateLimitError,  # 429
            openai.InternalServerError,  # >=500
            # NOTE: Errors listed below may need a revisit: we are not sure if
            # it's ever helpful to retry them. Leaving them intact for now.
            openai.AuthenticationError,  # 401
            openai.PermissionDeniedError,  # 403
            openai.NotFoundError,  # 404
            # Exceptions above (within the same except clause) are called out for documentation purposes.
            # Catching OpenAIError would cover everything by itself.
            openai.OpenAIError,
        ) as e:
            if not _is_fatal_openai_error(e):
                output = API_ERROR_OUTPUT  # disable hard fail (never raise!)
            # still, retry in the hope we'll get a successful reply
            if i == API_MAX_RETRY - 1:
                logger.error(e)
                break
            logger.debug(e)
            time.sleep(API_RETRY_SLEEP)

    if output is None:
        # not a single attempt was non-fatal; this is indicative of
        # basic connectivity or server issue -> hard fail
        raise exceptions.ModelServingAPIError
    return output


def check_data(questions, model_answers, ref_answers, models, judges):
    # check model answers
    for m in models:
        assert m in model_answers, f"Missing model answer for {m}"
        m_answer = model_answers[m]
        for q in questions:
            assert (
                q["question_id"] in m_answer
            ), f"Missing model {m}'s answer to Question {q['question_id']}"
    # check ref answers
    for jg in judges.values():
        if not jg.ref_based:
            continue
        for q in questions:
            if q["category"] not in NEED_REF_CATS:
                continue
            assert (
                q["question_id"] in ref_answers[jg.model_name]
            ), f"Missing reference answer to Question {q['question_id']} for judge {jg.model_name}"


def get_model_list(answer_file):
    logger.debug(locals())
    return [os.path.splitext(os.path.basename(answer_file))[0]]


def get_openai_client(
    model_api_base,
    api_key,
    http_client: httpx.Client | None = None,
):
    if api_key is None:
        api_key = "NO_API_KEY"
    openai_client = openai.OpenAI(
        base_url=model_api_base, api_key=api_key, http_client=http_client
    )
    return openai_client
