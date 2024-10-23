# eval

![Lint](https://github.com/instructlab/eval/actions/workflows/lint.yml/badge.svg?branch=main)
![Build](https://github.com/instructlab/eval/actions/workflows/pypi.yaml/badge.svg?branch=main)
![Release](https://img.shields.io/github/v/release/instructlab/eval)
![License](https://img.shields.io/github/license/instructlab/eval)

![`e2e-nvidia-a10g-x1.yml` on `main`](https://github.com/instructlab/eval/actions/workflows/e2e-nvidia-a10g-x1.yml/badge.svg?branch=main)
![`e2e-nvidia-l40s-x4.yml` on `main`](https://github.com/instructlab/eval/actions/workflows/e2e-nvidia-l40s-x4.yml/badge.svg?branch=main)

Python Library for Evaluation

## What is Evaluation?

Evaluation allows us to assess how a given model is performing against a set of specific tasks. This is done by running a set of standardized benchmark tests against
the model. Running evaluation produces numerical scores across these various benchmarks, as well as logs excerpts/samples of the outputs the model produced during these
benchmarks. Using a combination of these artifacts as reference, along with a manual smoke test, allows us to get the best idea about whether or not a model has learned
and improved on something we are trying to teach it. There are 2 stages of model evaluation in the InstructLab process:

### Inter-checkpoint Evaluation

This step occurs during multi-phase training. Each phase of training produces multiple different “checkpoints” of the model that are taken at various stages during
the phase. At the end of each phase, we evaluate all the checkpoints in order to find the one that provides the best results. This is done as part of the
[InstructLab Training](https://github.com/instructlab/training) library.

### Full-scale final Evaluation

Once training is complete, and we have picked the best checkpoint from the output of the final phase, we can run full-scale evaluation suite which runs MT-Bench, MMLU,
MT-Bench Branch and MMLU Branch.

## Methods of Evaluation

Below are more in-depth explanations of the suite of benchmarks we are using as methods for evaluation of models.

### Multi-turn benchmark (MT-Bench)

**tl;dr** Full model evaluation of performance on **skills**

MT-Bench is a type of benchmarking that involves asking a model 80 multi-turn questions - i.e.

```text
<Question 1> → <model’s answer 1> → <Follow-up question> → <model’s answer 2>
```

A “judge” model reviews the given multi-turn question, the provided model answer, and rate the answer with a score out of 10. The scores are then averaged out
and the final score produced is the “MT-bench score” for that model. This benchmark assumes no factual knowledge on the model’s part. The questions are static, but do not get obsolete with time.

You can read more about MT-Bench [here](https://arxiv.org/abs/2306.05685)

### MT-Bench Branch

MT-Bench Branch is an adaptation of MT-Bench that is designed to test custom skills that are added to the model with the InstructLab project. These new skills
come in the form of question/answer pairs in a Git branch of the [taxonomy](https://github.com/instructlab/taxonomy).

MT-Bench Branch uses the user supplied seed questions to have the candidate model generate answers to, which are then judged by the judge model using the user supplied
seed answers as a reference.

### Massive Multitask Language Understanding (MMLU)

**tl;dr** Full model evaluation of performance on **knowledge**

MMLU is a type of benchmarking that involves a series of fact-based multiple choice questions, along with 4 options for answers. It tests if a model is able to interpret
the questions correctly, along the answers, formulate its own answer, then selects the correct option out of the provided ones. The questions are designed as a set
of 57 “tasks”, and each task has a given domain. The domains cover a number of topics ranging from Chemistry and Biology to US History and Math.

The performance number is then compared against the set of known correct answers for each question to determine how many the model got right. The final MMLU score is the
average of its scores. This benchmark does not involve any reference/critic model, and is a completely objective benchmark. This benchmark does assume factual knowledge
on the model’s part. The questions are static, therefore MMLU cannot be used to gauge the model’s knowledge on more recent topics.

InstructLab uses an implementation found [here](https://github.com/EleutherAI/lm-evaluation-harness) for running MMLU.

You can read more about MMLU [here](https://arxiv.org/abs/2306.05685)

### MMLU Branch

MMLU Branch is an adaptation of MMLU that is designed to test custom knowledge that is being added to the model via a Git branch of the [taxonomy](https://github.com/instructlab/taxonomy).

A teacher model is used to generate new multiple choice questions based on the knowledge document included in the taxonomy Git branch. A “task” is then constructed that references the newly generated answer choices. These tasks are then used to score the model’s grasp on new knowledge the same way MMLU works. Generation of these tasks are done as part of the [InstructLab SDG](https://github.com/instructlab/sdg) library.

## MT-Bench / MT-Bench Branch Testing Steps

> **⚠️ Note:** Must use Python version 3.10 or later.

```shell
# Optional: Use cloud-instance.sh (https://github.com/instructlab/instructlab/tree/main/scripts/infra) to launch and setup the instance
scripts/infra/cloud-instance.sh ec2 launch -t g5.4xlarge
scripts/infra/cloud-instance.sh ec2 setup-rh-devenv
scripts/infra/cloud-instance.sh ec2 install-rh-nvidia-drivers
scripts/infra/cloud-instance.sh ec2 ssh sudo reboot
scripts/infra/cloud-instance.sh ec2 ssh


# Regardless of how you setup your instance
git clone https://github.com/instructlab/taxonomy.git && pushd taxonomy && git branch rc && popd
git clone --bare https://github.com/instructlab/eval.git && git clone eval.git/ && cd eval && git remote add syncrepo ../eval.git
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
pip install vllm
python -m vllm.entrypoints.openai.api_server --model instructlab/granite-7b-lab --tensor-parallel-size 1
```

In another shell window

```shell
export INSTRUCTLAB_EVAL_FIRST_N_QUESTIONS=10 # Optional if you want to shorten run times
# Commands relative to eval directory
python3 tests/test_gen_answers.py
python3 tests/test_branch_gen_answers.py
```

Example output tree

```shell
eval_output/
├── mt_bench
│   └── model_answer
│       └── instructlab
│           └── granite-7b-lab.jsonl
└── mt_bench_branch
    ├── main
    │   ├── model_answer
    │   │   └── instructlab
    │   │       └── granite-7b-lab.jsonl
    │   ├── question.jsonl
    │   └── reference_answer
    │       └── instructlab
    │           └── granite-7b-lab.jsonl
    └── rc
        ├── model_answer
        │   └── instructlab
        │       └── granite-7b-lab.jsonl
        ├── question.jsonl
        └── reference_answer
            └── instructlab
                └── granite-7b-lab.jsonl
```

```shell
python3 tests/test_judge_answers.py
python3 tests/test_branch_judge_answers.py
```

Example output tree

```shell
eval_output/
├── mt_bench
│   ├── model_answer
│   │   └── instructlab
│   │       └── granite-7b-lab.jsonl
│   └── model_judgment
│       └── instructlab
│           └── granite-7b-lab_single.jsonl
└── mt_bench_branch
    ├── main
    │   ├── model_answer
    │   │   └── instructlab
    │   │       └── granite-7b-lab.jsonl
    │   ├── model_judgment
    │   │   └── instructlab
    │   │       └── granite-7b-lab_single.jsonl
    │   ├── question.jsonl
    │   └── reference_answer
    │       └── instructlab
    │           └── granite-7b-lab.jsonl
    └── rc
        ├── model_answer
        │   └── instructlab
        │       └── granite-7b-lab.jsonl
        ├── model_judgment
        │   └── instructlab
        │       └── granite-7b-lab_single.jsonl
        ├── question.jsonl
        └── reference_answer
            └── instructlab
                └── granite-7b-lab.jsonl
```
