# SPDX-License-Identifier: Apache-2.0

# Standard
import os

# Third Party
from lm_eval.evaluator import simple_evaluate  # type: ignore
from lm_eval.tasks import TaskManager  # type: ignore
import torch

# First Party
from instructlab.eval.evaluator import Evaluator

MMLU_TASKS = [
    "mmlu_abstract_algebra",
    "mmlu_anatomy",
    "mmlu_astronomy",
    "mmlu_business_ethics",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_chemistry",
    "mmlu_college_computer_science",
    "mmlu_college_mathematics",
    "mmlu_college_medicine",
    "mmlu_college_physics",
    "mmlu_computer_security",
    "mmlu_conceptual_physics",
    "mmlu_econometrics",
    "mmlu_electrical_engineering",
    "mmlu_elementary_mathematics",
    "mmlu_formal_logic",
    "mmlu_global_facts",
    "mmlu_high_school_biology",
    "mmlu_high_school_chemistry",
    "mmlu_high_school_computer_science",
    "mmlu_high_school_european_history",
    "mmlu_high_school_geography",
    "mmlu_high_school_government_and_politics",
    "mmlu_high_school_macroeconomics",
    "mmlu_high_school_mathematics",
    "mmlu_high_school_microeconomics",
    "mmlu_high_school_physics",
    "mmlu_high_school_psychology",
    "mmlu_high_school_statistics",
    "mmlu_high_school_us_history",
    "mmlu_high_school_world_history",
    "mmlu_human_aging",
    "mmlu_human_sexuality",
    "mmlu_international_law",
    "mmlu_jurisprudence",
    "mmlu_logical_fallacies",
    "mmlu_machine_learning",
    "mmlu_management",
    "mmlu_marketing",
    "mmlu_medical_genetics",
    "mmlu_miscellaneous",
    "mmlu_moral_disputes",
    "mmlu_moral_scenarios",
    "mmlu_nutrition",
    "mmlu_philosophy",
    "mmlu_prehistory",
    "mmlu_professional_accounting",
    "mmlu_professional_law",
    "mmlu_professional_medicine",
    "mmlu_professional_psychology",
    "mmlu_public_relations",
    "mmlu_security_studies",
    "mmlu_sociology",
    "mmlu_us_foreign_policy",
    "mmlu_virology",
    "mmlu_world_religions",
]


class MMLUEvaluator(Evaluator):
    """
    Child class of an Evaluator for Massive Multitask Language Understanding (MMLU)

    Attributes:
        model_path   absolute path to or name of a huggingface model
        tasks        list of tasks for MMLU to test the model with
        model_dtype  dtype of model when served
        few_shots    number of examples
        batch_size   number of GPUs
    """

    name = "mmlu"

    def __init__(
        self,
        model_path,
        tasks: list[str] = MMLU_TASKS,
        model_dtype="bfloat16",
        few_shots: int = 2,
        batch_size: int = 5,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.tasks = tasks
        self.model_dtype = model_dtype
        self.few_shots = few_shots
        self.batch_size = batch_size

    def run(self) -> tuple:
        """
        Runs MMLU evaluation

        Returns:
            overall_score       MMLU score for the overall model evaluation
            individual_scores   Individual MMLU score for each task
        """
        # TODO: make this a parameter for class?
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        individual_scores: dict = {}
        agg_score: float = 0.0
        model_args = f"pretrained={self.model_path},dtype={self.model_dtype}"
        mmlu_output = simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=self.tasks,
            num_fewshot=self.few_shots,
            batch_size=self.batch_size,
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )

        results = mmlu_output["results"]

        for task in self.tasks:
            mmlu_res = results[task]
            agg_score += float(mmlu_res["acc,none"])
            individual_scores[task] = {}
            individual_scores[task]["score"] = float(mmlu_res["acc,none"])
            individual_scores[task]["stderr"] = float(mmlu_res["acc_stderr,none"])

        overall_score = float(agg_score / len(self.tasks))
        return overall_score, individual_scores


class MMLUBranchEvaluator(Evaluator):
    """
    Child class of an Evaluator for Massive Multitask Language Understanding Branch (MMLUBranch)

    Attributes:
        model_path  absolute path to or name of a huggingface model
        sdg_path    path where the <TASK_NAME>.jsonl and <TASK_NAME>_task.yaml files for the branches being evaluated are stored
        tasks       group name that is shared by all the MMLUBranch tasks
        few_shots   number of examples
        batch_size  number of GPUs
    """

    name = "mmlu_branch"

    def __init__(
        self,
        model_path,
        sdg_path: str,
        tasks: list[str],
        model_dtype="bfloat16",
        few_shots: int = 2,
        batch_size: int = 5,
    ) -> None:
        self.model_path = model_path
        self.sdg_path = sdg_path
        self.tasks = tasks
        self.model_dtype = model_dtype
        self.few_shots = few_shots
        self.batch_size = batch_size

    def run(self) -> tuple:
        """
        Runs MMLUBranch evaluation

        Returns:
            overall_score       Average MMLUBranch score for the task group
            individual_scores   Individual MMLUBranch scores for each task in the task group
        """
        # TODO: make this a parameter for class?
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        individual_scores: dict = {}
        agg_score: float = 0.0
        model_args = f"pretrained={self.model_path},dtype={self.model_dtype}"

        tm = TaskManager(verbosity="DEBUG", include_path=self.sdg_path)

        mmlu_output = simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=self.tasks,
            num_fewshot=self.few_shots,
            batch_size=self.batch_size,
            task_manager=tm,
        )
        results = mmlu_output["results"]

        for task, result in results.items():
            if task in self.tasks:
                agg_score += float(result["acc,none"])
            else:
                individual_scores[task] = {
                    "score": float(result["acc,none"]),
                    "stderr": float(result["acc_stderr,none"]),
                }

        overall_score = float(agg_score / len(self.tasks))

        return overall_score, individual_scores
