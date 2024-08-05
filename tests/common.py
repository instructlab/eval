# First Party
from instructlab.eval.mt_bench_common import (
    API_ERROR_OUTPUT,
    bench_dir,
    load_model_answers,
)


def _is_bad_answer(a):
    return any(API_ERROR_OUTPUT in choice["turns"] for choice in a["choices"])


def is_bad_answer_file(evaluator):
    branch = getattr(evaluator, 'branch', None)
    base_dir = bench_dir(evaluator.output_dir, evaluator.name, branch)
    answer_dir = f"{base_dir}/model_answer"
    for dataset in load_model_answers(answer_dir).values():
        for answer in dataset.values():
            if _is_bad_answer(answer):
                return True
    return False
