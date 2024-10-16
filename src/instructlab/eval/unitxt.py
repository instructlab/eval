"""
Unitxt - Flexible, Shareable and Reusable Data Preparation and Evaluation for Generative AI
https://github.com/IBM/unitxt
https://arxiv.org/abs/2401.14019
"""

# Standard
import os

# First Party
from instructlab.eval.mmlu import MMLUBranchEvaluator

# Local
from .logger_config import setup_logger

logger = setup_logger(__name__)

class UnitxtEvaluator(MMLUBranchEvaluator):
    name = "unitxt"
    def __init__(
        self,
        model_path,
        tasks_dir: str,
        tasks: list[str],    
        # unitxt_recipe: str,
    ):
        # tasks,tasks_dir = self.prepare_files(unitxt_recipe)
        super().__init__(
            model_path = model_path,
            tasks_dir = tasks_dir,
            tasks = tasks,
            few_shots = 0
        )

    def prepare_files(self, unitxt_recipe)->tuple:
        tasks = ''
        tasks_dir = ''
        return tasks,tasks_dir

    def run(self,server_url: str | None = None) -> tuple:
        """
        Runs evaluation

        Returns:
            overall_scores       Average scores for the task group
            individual_scores   Individual scores for each task in the task group
        """
        logger.debug(locals())
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        results = self._run_mmlu(server_url=server_url)
        with open('my_tasks/output.txt', 'w') as f:
            print(results, file=f)
        taskname = self.tasks[0]
        global_scores = results[taskname]
        global_scores.pop('alias')
        instance_scores = None
        # instances = results['samples'][taskname]
        # instance_scores = {}
        # metrics = [metric.replace('metrics.','') for metric in instances[0]['doc']['metrics']]
        # for i,instance in enumerate(instances):
        #     scores = {}
        #     for metric in metrics:
        #         scores[metric] = instance[metric][0]
        #     instance_scores[i] = scores
        return global_scores,instance_scores
