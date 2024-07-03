# Standard
import pprint

# First Party
from instructlab.eval.mt_bench import MTBenchBranchEvaluator

mt_bench_branch = MTBenchBranchEvaluator(
    "instructlab/granite-7b-lab",
    "instructlab/granite-7b-lab",
    "../taxonomy",
    "main",
)
qa_pairs = mt_bench_branch.judge_answers("http://localhost:8000/v1")
print(f"QA Pair 0:")
pprint.pprint(qa_pairs[0])

print(f"qa_pairs length: {len(qa_pairs)}")

for qa_pair in qa_pairs:
    question_id = qa_pair.get("question_id")
    assert question_id is not None
    assert qa_pair.get("score") is not None
    assert qa_pair.get("category") is not None
    assert qa_pair.get("question") is not None
    assert qa_pair.get("answer") is not None
    assert qa_pair.get("qna_file") is not None
