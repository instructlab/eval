# SPDX-License-Identifier: Apache-2.0

# Standard
import os

# First Party
from instructlab.eval.mt_bench_common import Judge
from instructlab.eval.mt_bench_judgment import load_judge_prompts, make_judge_single


def test_make_judge_single():
    judge_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "instructlab",
        "eval",
        "data",
        "mt_bench",
        "judge_prompts.jsonl",
    )
    judge_prompts = load_judge_prompts(judge_file)
    judges = make_judge_single("prometheus-8x7b-v2-0", judge_prompts)
    assert len(judges) == 4
    assert isinstance(judges["default"], Judge)
    assert isinstance(judges["math"], Judge)
    assert judges["math"].ref_based
    assert isinstance(judges["default-mt"], Judge)
    assert judges["default-mt"].multi_turn
    assert isinstance(judges["math-mt"], Judge)
    assert judges["math-mt"].ref_based
    assert judges["math-mt"].multi_turn
