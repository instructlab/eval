# First Party
from instructlab.eval.unitxt import UnitxtEvaluator


def test_unitxt():
    print("===> Executing 'test_unitxt'...")
    try:
        model_path = "instructlab/granite-7b-lab"
        unitxt_recipe = "card=cards.wnli,template=templates.classification.multi_class.relation.default,max_train_instances=5,loader_limit=20,num_demos=3,demos_pool_size=10"
        unitxt = UnitxtEvaluator(model_path=model_path, unitxt_recipe=unitxt_recipe)
        overall_score, single_scores = unitxt.run()
        print(overall_score)
        sample_score = "f1_micro,none"
        assert sample_score in overall_score
        assert overall_score[sample_score] > 0
    except Exception as exc:
        print(f"'test_unitxt_branch' failed: {exc}")
        return False

    return True


if __name__ == "__main__":
    assert test_unitxt() == True
