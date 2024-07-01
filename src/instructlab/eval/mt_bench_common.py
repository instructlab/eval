"""
Common data structures and utilities.
"""

# Standard
from typing import Optional
import ast
import dataclasses
import glob
import json
import os
import re
import time

# Third Party
from fastchat.model.model_adapter import get_conversation_template  # type: ignore
import openai

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
    ref_answer: dict = None  # type: ignore[assignment]
    multi_turn: bool = False


def bench_dir(output_dir, bench_name, branch) -> str:
    b_dir = f"{output_dir}/{bench_name}"
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


def load_model_answers(answer_dir: str, model_name=None) -> dict:
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    model_answers = {}
    for root, _, files in os.walk(answer_dir):
        for filename in files:
            if filename.endswith(".jsonl"):
                # Removing ".jsonl"
                file_model_name = filename[:-6]
                answer = {}
                file_path = os.path.join(root, filename)
                with open(file_path, encoding="utf-8") as fin:
                    for line in fin:
                        l = json.loads(line)
                        answer[l["question_id"]] = l
                model_answers[model_name or file_model_name] = answer
                if model_name == file_model_name:
                    break
    return model_answers


def load_judge_prompts(prompt_file: str) -> dict:
    """Load judge prompts.

    The return value is a python dict of type:
    Dict[judge_name: str -> dict]
    """
    prompts = {}
    with open(prompt_file, encoding="utf-8") as fin:
        for line in fin:
            l = json.loads(line)
            prompts[l["name"]] = l
    return prompts


def run_judge_single(
    question, answer, judge, ref_answer, openai_client, multi_turn=False, judgment=None
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
    conv = get_conversation_template(model)
    conv.set_system_message(system_prompt)
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)

    if judgment is None:
        judgment = chat_completion_openai(
            openai_client, model, conv, temperature=0, max_tokens=2048
        )

    if judge.prompt_template["output_format"] == "[[rating]]":
        match = re.search(one_score_pattern, judgment)
        if not match:
            match = re.search(one_score_pattern_backup, judgment)

        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = -1
    else:
        raise ValueError(
            f"invalid output format: {judge.prompt_template['output_format']}"
        )

    return rating, user_prompt, judgment


def play_a_match_single(openai_client, match: MatchSingle, output_file: str) -> dict:
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


def chat_completion_openai(openai_client, model, conv, temperature, max_tokens) -> str:
    output = API_ERROR_OUTPUT
    for i in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            if (
                os.environ.get("ILAB_EVAL_MERGE_SYS_USR")
                and messages[0]["role"] == "system"
                and messages[1]["role"] == "user"
            ):
                messages[1]["content"] = (
                    messages[0]["content"] + "\n" + messages[1]["content"]
                )
                messages = messages[1:]
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response.choices[0].message.content
            break
        except openai.OpenAIError as e:
            if i == API_MAX_RETRY - 1:
                # Print error on last try
                print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

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


def get_model_list(answer_dir):
    file_paths = glob.glob(f"{answer_dir}/*.jsonl")
    file_names = [os.path.splitext(os.path.basename(f))[0] for f in file_paths]
    return file_names
