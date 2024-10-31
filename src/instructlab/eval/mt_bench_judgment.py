# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import ThreadPoolExecutor
import os

# Third Party
from tqdm import tqdm
import numpy as np
import pandas as pd

# Local
from .logger_config import setup_logger
from .mt_bench_common import (
    NEED_REF_CATS,
    Judge,
    MatchSingle,
    bench_dir,
    check_data,
    get_model_list,
    get_openai_client,
    load_judge_prompts,
    load_model_answers,
    load_questions,
    play_a_match_single,
)

logger = setup_logger(__name__)


def make_match_single(
    questions,
    models,
    model_answers,
    judge,
    ref_answers=None,
    multi_turn=False,
):
    """Setup a match"""
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        q_id = q["question_id"]
        for m in models:
            a = model_answers[m][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.model_name][q_id]
                matches.append(
                    MatchSingle(
                        dict(q), m, a, judge, ref_answer=ref, multi_turn=multi_turn
                    )
                )
            else:
                matches.append(MatchSingle(dict(q), m, a, judge, multi_turn=multi_turn))
    return matches


def make_judge_single(judge_model_name, judge_prompts) -> dict:
    """Setup the judge"""
    judges = {}
    judges["default"] = Judge(judge_model_name, judge_prompts["single-v1"])
    judges["math"] = Judge(
        judge_model_name, judge_prompts["single-math-v1"], ref_based=True
    )
    judges["default-mt"] = Judge(
        judge_model_name, judge_prompts["single-v1-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model_name,
        judge_prompts["single-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


def make_judgment(
    question_file,
    judgment_file,
    answer_file,
    bench_name="mt_bench",
):
    """Create judgment output"""
    logger.debug(locals())
    judgment_df_all = pd.read_json(
        judgment_file, lines=True, dtype={"question_id": str}
    )
    judgment_df = judgment_df_all[["model", "score", "turn"]]
    judgments_len = len(judgment_df)
    judgment_df = judgment_df[judgment_df["score"] != -1]
    error_free_judgments_len = len(judgment_df)
    error_rate = (judgments_len - error_free_judgments_len) / judgments_len
    logger.debug("#judgments: %s", judgments_len)
    logger.debug("#error free judgments: %s", error_free_judgments_len)
    logger.debug("error rate: %s", error_rate)

    turn_scores = []
    # First turn
    df_1 = judgment_df[judgment_df["turn"] == 1].groupby(["model", "turn"]).mean()
    overall_score = df_1["score"].iloc[0]
    turn_scores.append(overall_score)

    if bench_name == "mt_bench":
        # Second turn
        df_2 = judgment_df[judgment_df["turn"] == 2].groupby(["model", "turn"]).mean()
        if len(df_2.index) > 0:
            turn2_score = df_2["score"].iloc[0]
            turn_scores.append(turn2_score)

            # Average
            df_3 = judgment_df[["model", "score"]].groupby(["model"]).mean()
            overall_score = df_3["score"].iloc[0]
        else:
            turn_scores.append("N/A")

    question_df = pd.read_json(question_file, lines=True, dtype={"question_id": str})

    answer_df = pd.read_json(answer_file, lines=True, dtype={"question_id": str})

    # Join to get questions with answers
    join_columns = ["question_id", "choices", "turns", "category"]
    if bench_name == "mt_bench_branch":
        join_columns.append("qna_file")

    joined_df = question_df.join(
        answer_df.set_index("question_id"), on="question_id", rsuffix="_answer"
    )[join_columns]
    # Join to get scores
    join_columns.append("score")
    joined_df = judgment_df_all.join(
        joined_df.set_index("question_id"), on="question_id", lsuffix="_judgment"
    )[join_columns]
    joined_df = joined_df[joined_df["score"] != -1]

    qa_pairs = []
    for _, row in joined_df.iterrows():
        qa_pair = {
            "question_id": row["question_id"],
            "score": row["score"],
            "category": row["category"],
            "question": row["turns"],
            "answer": row["choices"],
        }
        if bench_name == "mt_bench_branch":
            qa_pair["qna_file"] = row["qna_file"]
        qa_pairs.append(qa_pair)
    return overall_score, qa_pairs, turn_scores, error_rate


def judge_model(
    model_name,
    judge_model_name,
    openai_client,
    branch=None,
    bench_name="mt_bench",
    output_dir="eval_output",
    data_dir=None,
    max_workers=1,
    first_n=None,
    merge_system_user_message=False,
):
    """Judge the model based on questions and reference answers"""
    logger.debug(locals())
    package_data_dir = os.path.join(os.path.dirname(__file__), "data")
    use_builtin_ref_answers = False
    if data_dir is None:
        use_builtin_ref_answers = True
        data_dir = package_data_dir

    data_base_dir = bench_dir(data_dir, bench_name, branch)
    output_base_dir = bench_dir(output_dir, bench_name, branch)

    judge_file = os.path.join(package_data_dir, bench_name, "judge_prompts.jsonl")

    question_file = os.path.join(data_base_dir, "question.jsonl")
    answer_file = os.path.join(output_base_dir, "model_answer", f"{model_name}.jsonl")
    if use_builtin_ref_answers:
        ref_answer_file = os.path.join(data_base_dir, "reference_answer", "gpt-4.jsonl")
    else:
        ref_answer_file = os.path.join(
            data_base_dir, "reference_answer", f"{judge_model_name}.jsonl"
        )

    # Load questions
    questions = load_questions(question_file, None, None)

    # Load answers
    model_answers = load_model_answers(answer_file)
    ref_answers = load_model_answers(ref_answer_file, judge_model_name)

    # Load judge
    judge_prompts = load_judge_prompts(judge_file)

    if first_n:
        questions = questions[:first_n]

    models = get_model_list(answer_file)

    judges = make_judge_single(judge_model_name, judge_prompts)
    output_file = os.path.join(
        output_base_dir, "model_judgment", f"{judge_model_name}_single.jsonl"
    )
    if os.path.isfile(output_file):
        os.remove(output_file)
        logger.debug("Removing previous judgment file: %s", output_file)

    check_data(questions, model_answers, ref_answers, models, judges)

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # Make matches
    matches = []
    matches += make_match_single(
        question_default, models, model_answers, judges["default"]
    )
    matches += make_match_single(
        question_math,
        models,
        model_answers,
        judges["math"],
        ref_answers,
    )
    matches += make_match_single(
        question_default,
        models,
        model_answers,
        judges["default-mt"],
        multi_turn=True,
    )
    matches += make_match_single(
        question_math,
        models,
        model_answers,
        judges["math-mt"],
        ref_answers,
        multi_turn=True,
    )

    logger.debug("bench_name=%s", bench_name)
    logger.debug("judge=%s", judge_model_name)
    logger.debug("model_list=%s", models)
    logger.debug("total_num_questions=%s", len(questions))
    logger.debug("total_num_matches=%s", len(matches))

    # Play matches
    if max_workers == 1:
        for match in tqdm(matches):
            play_a_match_single(
                openai_client,
                match,
                output_file=output_file,
                merge_system_user_message=merge_system_user_message,
            )
    else:

        def play_a_match_wrapper(match):
            play_a_match_single(
                openai_client,
                match,
                output_file=output_file,
                merge_system_user_message=merge_system_user_message,
            )

        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(max_workers) as executor:
            for match in tqdm(
                executor.map(play_a_match_wrapper, matches), total=len(matches)
            ):
                pass

    return question_file, output_file, answer_file


def generate_judgment(
    model_name,
    judge_model_name,
    model_api_base,
    api_key=None,
    bench_name="mt_bench",
    output_dir="eval_output",
    data_dir=None,
    branch=None,
    max_workers=1,
    first_n=None,
    merge_system_user_message=False,
    http_client=None,
):
    """Generate judgment with scores and qa_pairs for a model"""
    logger.debug(locals())

    openai_client = get_openai_client(model_api_base, api_key, http_client)

    first_n_env = os.environ.get("INSTRUCTLAB_EVAL_FIRST_N_QUESTIONS")
    if first_n_env is not None and first_n is None:
        first_n = int(first_n_env)
        logger.debug("INSTRUCTLAB_EVAL_FIRST_N_QUESTIONS=%s", first_n)

    question_file, judgment_file, answer_file = judge_model(
        model_name,
        judge_model_name,
        openai_client,
        bench_name=bench_name,
        output_dir=output_dir,
        data_dir=data_dir,
        branch=branch,
        max_workers=max_workers,
        first_n=first_n,
        merge_system_user_message=merge_system_user_message,
    )

    return make_judgment(
        question_file,
        judgment_file,
        answer_file,
        bench_name=bench_name,
    )
