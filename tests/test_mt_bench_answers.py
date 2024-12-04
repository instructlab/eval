# SPDX-License-Identifier: Apache-2.0

# Standard
import json
import os
import random
import shutil
import tempfile

# First Party
from instructlab.eval.mt_bench_answers import reorg_answer_file


def test_reorg_answer_file():
    answer_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "instructlab",
        "eval",
        "data",
        "mt_bench",
        "reference_answer",
        "gpt-4.jsonl",
    )

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        temp_answer_file = temp_file.name

        # Copy the original file to the temp file
        shutil.copy(answer_file, temp_answer_file)

        orig_length = 0
        with open(temp_answer_file, "r+", encoding="utf-8") as f:
            answers = {}
            for l in f:
                orig_length += 1
                qid = json.loads(l)["question_id"]
                answers[qid] = l

            # Reset to the beginning of the file and clear it
            f.seek(0)
            f.truncate()

            # Randomize the values
            qids = sorted(list(answers.keys()), key=lambda answer: random.random())
            for qid in qids:
                f.write(answers[qid])
                # Write each answer twice
                f.write(answers[qid])

        # Run the reorg which should sort and dedup the file in place
        reorg_answer_file(temp_answer_file)

        new_length = 0
        with open(temp_answer_file, "r", encoding="utf-8") as fin:
            previous_question_id = -1
            for l in fin:
                new_length += 1
                qid = json.loads(l)["question_id"]
                assert qid > previous_question_id
                previous_question_id = qid

        assert new_length == orig_length
