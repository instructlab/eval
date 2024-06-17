"""Generate answers with GPT-4

Usage:
python3 gen_api_answer.py --model gpt-3.5-turbo
"""
import argparse
import json
import os
import time
import concurrent.futures

import openai
import shortuuid
import tqdm

from .common import (
    load_questions,
    temperature_config,
    chat_completion_openai,
    chat_completion_anthropic,
    chat_completion_palm,
    ANTHROPIC_MODEL_LIST,
)
#TODO need to look into this dependency
from fastchat.model.model_adapter import get_conversation_template

def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


def get_answer(
        question: dict, model: str, num_choices: int, max_tokens: int, answer_file: str, force_temperature: float
):
    assert (
        force_temperature is not None and "required_temperature" in question.keys()
    ) == False
    if force_temperature is not None:
        temperature = force_temperature
    elif "required_temperature" in question.keys():
        temperature = question["required_temperature"]
    elif question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7

    choices = []
    chat_state = None  # for palm-2 model
    for i in range(num_choices):
        conv = get_conversation_template(model)

        turns = []
        for j in range(len(question["turns"])):
            conv.append_message(conv.roles[0], question["turns"][j])
            conv.append_message(conv.roles[1], None)

            if model in ANTHROPIC_MODEL_LIST:
                output = chat_completion_anthropic(model, conv, temperature, max_tokens)
            elif model == "palm-2-chat-bison-001":
                chat_state, output = chat_completion_palm(
                    chat_state, model, conv, temperature, max_tokens
                )
            else:
                output = chat_completion_openai(model, conv, temperature, max_tokens)

            conv.update_last_message(output)
            turns.append(output)

        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans) + "\n")

def run(
        question_begin=None,
        question_end=None,
        force_temperature=None,
        answer_file=None,
        model_name="gpt-3.5-turbo",
        num_choices=1,
        max_tokens=1024,
        parallel=1,
        openai_api_base=None):

    os.environ(
    if openai_api_base is not None:
        openai.api_base = openai_api_base

    question_file = f"data/mt_bench/question.jsonl"
    questions = load_questions(question_file, question_begin, question_end)

    if not answer_file:
        answer_file = f"data/mt_bench/model_answer/{model_name}.jsonl"
    print(f"Output to {answer_file}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = []
        for question in questions:
            future = executor.submit(
                get_answer,
                question,
                model_name,
                num_choices,
                max_tokens,
                answer_file,
                force_temperature,
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    reorg_answer_file(answer_file)
