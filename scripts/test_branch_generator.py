# Standard
import argparse
import os

# First Party
from instructlab.eval import mt_bench_branch_generator


def test_mt_bench_branch_generator(test_dir):
    output_dir = os.path.join(test_dir, "mt_bench_branch_generator")
    mt_bench_branch_generator.generate(
        "prometheus-eval/prometheus-8x7b-v2.0",
        "main",
        "taxonomy",
        output_dir,
    )
    main_dir = os.path.join(output_dir, "mt_bench_branch", "main")
    assert os.path.isfile(os.path.join(main_dir, "question.jsonl"))
    assert os.path.isfile(
        os.path.join(
            main_dir,
            "reference_answer",
            "prometheus-eval",
            "prometheus-8x7b-v2.0.jsonl",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Branch Generator")
    parser.add_argument("--test-dir", help="Base test working directory")
    args = parser.parse_args()
    test_dir = args.test_dir

    test_mt_bench_branch_generator(test_dir)
