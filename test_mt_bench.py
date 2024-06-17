from instructlab.eval.mtbench import MT_Bench_Evaluator

mt_bench = MT_Bench_Evaluator(server_url="http://localhost:8000")
#path = mt_bench.gen_answers("test-answers.jsonl", "http://localhost:8000/v1")
#print(path)
output_file = mt_bench.judge_answers("http://localhost:8000/v1")
print(output_file)
