# Standard
import hashlib
import json
import os
import time

# Third Party
from tqdm import tqdm
import git
import shortuuid
import yaml

# Local
from .mt_bench_common import bench_dir


def get_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.split("/")[-1] == "qna.yaml":
                file_paths.append(os.path.join(root, file))
    return file_paths


def read_qna(fn):
    with open(fn, "r", encoding="utf-8") as file:
        contents = yaml.safe_load(file)
    return contents.get("seed_examples")


def generate(judge_model_name, branch, taxonomy_dir, output_dir):
    """Create questions and reference answers from taxonomy"""
    restore_branch = None
    try:
        if branch is not None:
            taxonomy_repo = git.Repo(taxonomy_dir)
            restore_branch = taxonomy_repo.active_branch
            taxonomy_repo.git.checkout(branch)

        qna_file_list = get_file_paths(taxonomy_dir)

        question_lst = []
        reference_answers = []
        for qna_file_path in tqdm(qna_file_list):
            examples = read_qna(qna_file_path)
            qna_file = qna_file_path[len(taxonomy_dir) + 1 :]
            if examples is None:
                print(f"failed to load {qna_file}. skipping...")
                continue
            for ex in examples:
                q, a = ex["question"], ex["answer"]
                if q is None or a is None:
                    continue

                c = ex["question"] if "context" in ex else None
                if c is not None:
                    t_1 = (
                        "Given the context below:\n"
                        + c
                        + "\n"
                        + "Answer the following question: "
                        + q
                    )
                else:
                    t_1 = q

                # Generate a consistent hash to have consistent question_id across qna_files from different runs
                str_bytes = bytes(q, "UTF-8")
                m = hashlib.md5(str_bytes)
                question_id = str(int(m.hexdigest(), base=16))
                question_lst.append(
                    {
                        "qna_file": qna_file,
                        "question_id": question_id,
                        "category": "taxonomy",
                        "turns": [t_1],
                        "reference": [a],
                    }
                )

                reference_answers.append(
                    {
                        "question_id": question_id,
                        "answer_id": shortuuid.uuid(),
                        "model_id": judge_model_name,
                        "choices": [{"index": 0, "turns": [a]}],
                        "tstamp": time.time(),
                    }
                )

        print(f"generated {len(question_lst)} questions")

        output_base_dir = bench_dir(output_dir, "mt_bench_branch", branch)
        os.makedirs(output_base_dir, exist_ok=True)
        question_fn = "question.jsonl"
        with open(
            os.path.join(output_base_dir, question_fn), "w", encoding="utf-8"
        ) as outfile:
            for entry in question_lst:
                json.dump(entry, outfile)
                outfile.write("\n")

        answer_file_dir = os.path.join(output_base_dir, "reference_answer")
        answer_file = os.path.join(answer_file_dir, f"{judge_model_name}.jsonl")
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(
            answer_file,
            "w",
            encoding="utf-8",
        ) as outfile:
            for entry in reference_answers:
                json.dump(entry, outfile)
                outfile.write("\n")
    finally:
        if restore_branch is not None:
            taxonomy_repo.git.checkout(restore_branch)
