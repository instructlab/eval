# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest import mock
from unittest.mock import patch

# First Party
from instructlab.eval.mt_bench import MTBenchBranchEvaluator, MTBenchEvaluator


def gen_qa_pairs(odd):
    i = 1
    qa_pairs = []
    score = 0
    while i < 5:
        if i % 2:
            if odd:
                score = 0.2
            else:
                score = 0.1
        elif not i % 2:
            if odd:
                score = 0.3
            else:
                score = 0.4
        qa_pairs.append(
            {
                "question_id": i,
                "score": score,
                "qna_file": f"category{i}/qna.yaml",
            }
        )
        i = i + 1
    qa_pairs.append(
        {
            "question_id": i,
            "score": 0.5,
            "qna_file": f"category{i}/qna.yaml",
        }
    )
    if odd:
        qa_pairs.append(
            {
                "question_id": i + 1,
                "score": 0.6,
                "qna_file": f"category{i+1}/qna.yaml",
            }
        )
    return qa_pairs


@patch("instructlab.eval.mt_bench_branch_generator.generate")
@patch("instructlab.eval.mt_bench_answers.generate_answers")
@patch(
    "instructlab.eval.mt_bench_judgment.generate_judgment",
    return_value=(0, gen_qa_pairs(True), None, 0),
)
def test_mt_bench_branch(gen_judgment_mock, gen_answers_mock, generate_mock):
    mt_bench_branch = MTBenchBranchEvaluator(
        "instructlab/granite-7b-lab",
        "prometheus-eval/prometheus-8x7b-v2.0",
        "../taxonomy",
        "main",
    )
    mt_bench_branch.gen_answers(
        "http://localhost:8000/v1",
    )
    overall_score, qa_pairs, error_rate = mt_bench_branch.judge_answers(
        "http://localhost:8000/v1",
    )
    assert overall_score == 0
    assert qa_pairs == gen_qa_pairs(True)
    assert error_rate == 0

    gen_judgment_mock.assert_called()
    gen_answers_mock.assert_called()
    generate_mock.assert_called()


@patch("instructlab.eval.mt_bench_answers.generate_answers")
@patch(
    "instructlab.eval.mt_bench_judgment.generate_judgment",
    return_value=(1.5001, [{}, {}], [1.002, 2], 0),
)
def test_mt_bench(gen_judgment_mock, gen_answers_mock):
    mt_bench = MTBenchEvaluator(
        "instructlab/granite-7b-lab",
        "prometheus-eval/prometheus-8x7b-v2.0",
    )
    mt_bench.gen_answers(
        "http://localhost:8000/v1",
    )
    overall_score, qa_pairs, turn_scores, error_rate = mt_bench.judge_answers(
        "http://localhost:8000/v1",
    )

    assert overall_score == 1.5001
    assert qa_pairs == [{}, {}]
    assert turn_scores == [1.002, 2]
    assert error_rate == 0

    gen_judgment_mock.assert_called()
    gen_answers_mock.assert_called()
