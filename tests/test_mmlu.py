# First Party
from instructlab.eval.mmlu import MMLUEvaluator


def test_minimal_mmlu():
    print("===> Executing 'test_minimal_mmlu'...")
    try:
        model_path = "instructlab/granite-7b-lab"
        tasks = ["mmlu_anatomy", "mmlu_astronomy"]
        mmlu = MMLUEvaluator(model_path=model_path, tasks=tasks)
        overall_score, individual_scores = mmlu.run()
        print(overall_score)
        print(individual_scores)
    except Exception as exc:
        print(f"'test_minimal_mmlu' failed: {exc}")
        return False
    return True


if __name__ == "__main__":
    assert test_minimal_mmlu() == True
