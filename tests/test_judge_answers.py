# Standard
import pprint

# First Party
from instructlab.eval.mt_bench import MTBenchEvaluator

mt_bench = MTBenchEvaluator("instructlab/granite-7b-lab", "instructlab/granite-7b-lab")
overall_score, qa_pairs, turn_scores = mt_bench.judge_answers(
    "http://localhost:8000/v1"
)

print(f"Overall Score: {overall_score}")
print(f"Turn 1 Score: {turn_scores[0]}")
print(f"Turn 2 Score: {turn_scores[1]}")
print(f"QA Pair 0:")
pprint.pprint(qa_pairs[0])

print(f"qa_pairs length: {len(qa_pairs)}")

for qa_pair in qa_pairs:
    assert qa_pair.get("question_id") is not None
    assert qa_pair.get("score") is not None
    assert qa_pair.get("category") is not None
    assert qa_pair.get("question") is not None
    assert qa_pair.get("answer") is not None
