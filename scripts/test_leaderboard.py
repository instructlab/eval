# First Party
from instructlab.eval.leaderboard import LeaderboardV2Evaluator

if __name__ == "__main__":
    evaluator = LeaderboardV2Evaluator(
        model_path="ibm-granite/granite-3.1-8b-instruct",
    )
    results = evaluator.run()
    print("got results from leaderboard v2")
    print(json.dumps(results, indent=2))
