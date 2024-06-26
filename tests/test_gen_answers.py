# First Party
from instructlab.eval.mt_bench import MTBenchEvaluator

mt_bench = MTBenchEvaluator("instructlab/granite-7b-lab", "instructlab/granite-7b-lab")
mt_bench.gen_answers("http://localhost:8000/v1")
