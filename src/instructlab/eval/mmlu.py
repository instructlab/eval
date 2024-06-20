# SPDX-License-Identifier: Apache-2.0

# Local
from instructlab.eval.evaluator import Evaluator

# Third Party 
from lm_eval.evaluator import simple_evaluate
import os

class MMLU_Evaluator(Evaluator):
    """
    Child class of an Evaluator for Massive Multitask Language Understanding (MMLU)

    Attributes:
        model_path   absolute path to or name of a huggingface model
        tasks        list of tasks for MMLU to test the model with
        model_dtype  dtype of model when served
        few_shots    number of examples
        batch_size   number of GPUs
    """

    def __init__(
        self, model_path, tasks: list[str], model_dtype = 'bfloat16', few_shots: int = 2, batch_size: int = 5
    ) -> None:
        super().__init__(model_path)
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
        #TODO: make this a parameter for class?
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'

        individual_scores: dict = {}
        agg_score: float = 0.0
        model_args = "pretrained=" + self.model_path + ",dtype=" + self.model_dtype

        mmlu_output = simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=self.tasks,
            num_fewshot=self.few_shots,
            batch_size=self.batch_size
        )

        results = mmlu_output["results"]

        for task in self.tasks:
            mmlu_res = results[task]
            agg_score += float(mmlu_res['acc,none'])
            individual_scores[task] = {}
            individual_scores[task]['score'] = float(mmlu_res['acc,none'])
            individual_scores[task]['stderr'] = float(mmlu_res['acc_stderr,none'])

        overall_score = float(agg_score/len(self.tasks))
        return overall_score, individual_scores
    

############# Testing Code Follows ##############
def main():
    # Path to the granite model in the aliryan vm on AWS
    model_path = "/home/ec2-user/instructlab/models/instructlab/granite-7b-lab"
    #TODO: all 57 tasks need to be parameterized possibly by CLI
    tasks = ["mmlu_abstract_algebra","mmlu_anatomy","mmlu_astronomy"]
    dtype = "float16"
    mmlu = MMLU_Evaluator(model_path, tasks, dtype, 2, 5)
    overall_score, individual_scores = mmlu.run()
    print(overall_score)
    print(individual_scores)

if __name__ == "__main__":
    main()
############# Testing Code Ends ##############

class PR_MMLU_Evaluator(Evaluator):
    """
    Child class of an Evaluator for PR Massive Multitask Language Understanding (PR MMLU)

    Attributes:
        sdg_path    path where all the PR MMLU tasks are stored
        task        group name that is shared by all the PR MMLU tasks
        few_shots   number of examples
        batch_size  number of GPUs
    """

    def __init__(
        self,
        model_path,
        sdg_path: str,
        task: str = "mmlu_pr",
        few_shots: int = 2,
        batch_size: int = 5,
    ) -> None:
        self.model_path = model_path
        self.sdg_path = sdg_path
        self.task = task
        self.few_shots = few_shots
        self.batch_size = batch_size

    def run(self) -> tuple:
        """
        Runs PR MMLU evaluation

        Returns:
            overall_score       PR MMLU score for the overall model evaluation
            individual_scores   Individual PR MMLU scores for each task
            qa_pairs            Question and answer pairs from the evaluation
        """
        individual_scores: dict[str, float] = {}
        overall_score: float = 0.0
        qa_pairs: list[tuple] = []
        return overall_score, individual_scores, qa_pairs
