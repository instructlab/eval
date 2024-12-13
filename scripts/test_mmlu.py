# Standard
from typing import Dict, List, Tuple, TypedDict

# First Party
from instructlab.eval.mmlu import MMLUEvaluator

SYSTEM_PROMPT = """I am, Red Hat® Instruct Model based on Granite 7B, an AI language model developed by Red Hat and IBM Research, based on the Granite-7b-base language model. My primary function is to be a chat assistant."""


class MMLUSample(TypedDict):
    """
    Example of a single sample returned from lm_eval when running MMLU.
    This is not a comprehensive type, just the subset of fields we care about for this test.
    """

    # Arguments is the list of (prompt, answer) pairs passed to MMLU as few-shot samples.
    # They will not be present with few_shot=0
    arguments: List[Tuple[str, str]]


def all_samples_contain_system_prompt(
    samples: Dict[str, List[MMLUSample]], prompt: str
) -> bool:
    """
    Given a mapping of evaluation --> list of results, validates that all few-shot examples
    included the system prompt
    """
    for topic, samples_set in samples.items():
        for sample in samples_set:
            for mmlu_prompt, _ in sample["arguments"]:
                if prompt not in mmlu_prompt:
                    # we are looking for the exact system prompt, so no need to convert to normalize to lowercase
                    print(f"found a sample in the '{topic}' MMLU topic set")
                    return False

    return True


def test_minimal_mmlu():
    print("===> Executing 'test_minimal_mmlu'...")
    try:
        model_path = "instructlab/granite-7b-lab"
        tasks = ["mmlu_anatomy", "mmlu_astronomy"]
        mmlu = MMLUEvaluator(
            model_path=model_path,
            tasks=tasks,
            system_prompt=SYSTEM_PROMPT,
        )
        overall_score, individual_scores = mmlu.run(
            extra_args={"log_samples": True, "write_out": True}
        )
        samples = mmlu.results["samples"]

        print(overall_score)
        print(individual_scores)

        # we need n-shots > 1 to be able to validate the inclusion of the system prompt
        eligible_samples = {
            topic: samples[topic]
            for topic, shot in mmlu.results["n-shot"].items()
            if shot > 1
        }
        if eligible_samples:
            if not all_samples_contain_system_prompt(eligible_samples, SYSTEM_PROMPT):
                return False
        else:
            print(
                "MMLU was run in zero-shot mode, cannot confirm that system prompt was included, skipping check..."
            )

    except Exception as exc:
        print(f"'test_minimal_mmlu' failed: {exc}")
        return False
    return True


if __name__ == "__main__":
    assert test_minimal_mmlu() == True
