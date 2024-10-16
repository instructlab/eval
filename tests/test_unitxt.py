# First Party
from instructlab.eval.unitxt import UnitxtEvaluator


def test_unitxt():
    print("===> Executing 'test_unitxt'...")
    try:
        model_path = "instructlab/granite-7b-lab"
        tasks = ["my_task"]
        unitxt = UnitxtEvaluator(
            model_path=model_path, tasks_dir='./my_tasks/', tasks=tasks
        )
        overall_score, _ = unitxt.run()
        print(overall_score)
    except Exception as exc:
        print(f"'test_unitxt_branch' failed: {exc}")
        return False
    return True


if __name__ == "__main__":
    assert test_unitxt() == True