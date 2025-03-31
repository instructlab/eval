#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

# NOTE: This script requires the leaderboard optional dependencies.
# Install with: pip install instructlab-eval[leaderboard]

# Standard
import json

# First Party
from instructlab.eval.leaderboard import LeaderboardV2Evaluator

if __name__ == "__main__":
    evaluator = LeaderboardV2Evaluator(
        model_path="ibm-granite/granite-3.1-8b-base",
        eval_config={
            "apply_chat_template": False,
        },
    )
    results = evaluator.run()
    print("got results from leaderboard v2")
    print(json.dumps(results, indent=2))
