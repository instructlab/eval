# First Party
from instructlab.eval.mt_bench import MTBenchBranchEvaluator


def test_branch_gen_answers():
    mt_bench_branch = MTBenchBranchEvaluator(
        "instructlab/granite-7b-lab",
        "instructlab/granite-7b-lab",
        "taxonomy",
        "main",
    )
    mt_bench_branch.gen_answers("http://localhost:8000/v1")
