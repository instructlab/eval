# SPDX-License-Identifier: Apache-2.0

# Local
from .evaluator import Evaluator

# Third Party 
from lm_eval.evaluator import simple_evaluate

class MMLU_Evaluator(Evaluator):
    """
    Child class of an Evaluator for Massive Multitask Language Understanding (MMLU)

    Attributes:
        tasks        list of tasks for MMLU to test the model with
        few_shots    number of examples
        batch_size   number of GPUs
    """

    def __init__(
        self, model, model_args, model_path, tasks: list[str], few_shots: int = 2, batch_size: int = 5
    ) -> None:
        super().__init__(model_path)
        self.model = model
        self.model_args = model_args
        self.tasks = tasks
        self.few_shots = few_shots
        self.batch_size = batch_size

    def run(self) -> tuple:
        """
        Runs MMLU evaluation

        Returns:
            overall_score       MMLU score for the overall model evaluation
            individual_scores   Individual MMLU score for each task
        """
        individual_scores: dict[str, float] = {}
        overall_score: float = 0.0
        results = lm_eval.simple_evaluate(
            model=self.model,
            model_args=self.model_args,
            tasks=self.tasks,
            num_fewshot=self.few_shots,
            batch_size=self.batch_size,
            log_samples=True,
        )
        #TODO: see what the output of results looks like 
        #print(results)
        #calculate_overall_score(results)
        return overall_score, individual_scores
    
    def calculate_overall_score(scores):
        pass # Placeholder for calculating overall score:
             # overall score = (num model answered correctly / num questions)

############# Testing Code Follows ##############
def main():
    # TODO: change this- cli uses HuggingFace to access the model 
    model = "hf"
    model_args = "pretrained=$MODEL_PATH,dtype=bfloat16"
    # Path to the granite model in the aliryan vm on AWS
    model_path = "/home/ec2-user/instructlab/models/instructlab/granite-7b-lab"
    #TODO: all 57 tasks need to be parameterized possibly by CLI
    tasks = "mmlu_abstract_algebra"
    mmlu = MMLU_Evaluator(model, model_args, model_path, tasks, 2, 5)

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
