# Standard
import os

# First Party
from instructlab.eval.mmlu import MMLUBranchEvaluator


def test_mmlu_branch():
    print("===> Executing 'test_mmlu_branch'...")
    try:
        model_path = "instructlab/granite-7b-lab"
        tasks_dir = (
            f"{os.path.dirname(os.path.realpath(__file__))}/../tests/testdata/sdg"
        )
        tasks = ["mmlu_pr"]
        mmlu = MMLUBranchEvaluator(
            model_path=model_path, tasks_dir=tasks_dir, tasks=tasks
        )
        overall_score, individual_scores = mmlu.run()
        print(overall_score)
        print(individual_scores)
    except Exception as exc:
        print(f"'test_mmlu_branch' failed: {exc}")
        return False
    return True


if __name__ == "__main__":
    assert test_mmlu_branch() == True
