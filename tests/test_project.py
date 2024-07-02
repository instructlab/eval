# SPDX-License-Identifier: Apache-2.0
# Standard
from importlib.metadata import entry_points

# First Party
from instructlab.eval.evaluator import Evaluator
from instructlab.eval.mmlu import MMLUBranchEvaluator, MMLUEvaluator
from instructlab.eval.mt_bench import MTBenchBranchEvaluator, MTBenchEvaluator


def test_evaluator_eps():
    expected = {
        "mmlu": MMLUEvaluator,
        "mmlu_branch": MMLUBranchEvaluator,
        "mt_bench": MTBenchEvaluator,
        "mt_bench_branch": MTBenchBranchEvaluator,
    }
    eps = entry_points(group="instructlab.eval.evaluator")
    found = {}
    for ep in eps:
        # different project
        if not ep.module.startswith("instructlab.eval"):
            continue
        evaluator = ep.load()
        assert issubclass(evaluator, Evaluator)
        assert evaluator.name == ep.name
        found[ep.name] = evaluator

    assert found == expected
