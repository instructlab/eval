"""
Unitxt - Flexible, Shareable and Reusable Data Preparation and Evaluation for Generative AI
https://github.com/IBM/unitxt
https://arxiv.org/abs/2401.14019
"""

# Standard
import os, shutil
import yaml
from uuid import uuid4

# Third Party
from lm_eval.tasks.unitxt import task

# First Party
from instructlab.eval.mmlu import MMLUBranchEvaluator

# Local
from .logger_config import setup_logger

logger = setup_logger(__name__)

class UnitxtEvaluator(MMLUBranchEvaluator):
    """
    An evaluator class, running Unitxt evaluation

    Attributes:
        model_path      absolute path to or name of a huggingface model
        unitxt_recipe   unitxt recipe (see unitxt.ai for more information)
                        A Recipe holds a complete specification of a unitxt pipeline 
                        Example: card=cards.wnli,template=templates.classification.multi_class.relation.default,max_train_instances=5,loader_limit=20,num_demos=3,demos_pool_size=10
    
    """
    name = "unitxt"
    def __init__(
        self,
        model_path,   
        unitxt_recipe: str,
    ):
        tasks,tasks_dir = self.prepare_unitxt_files(unitxt_recipe)
        super().__init__(
            model_path = model_path,
            tasks_dir = tasks_dir,
            tasks = tasks,
            few_shots = 0
        )

    def prepare_unitxt_files(self, unitxt_recipe)->tuple:
        temp_task = str(uuid4())
        temp_tasks_dir = f'unitxt_temp_{temp_task}'
        yaml_file = os.path.join(temp_tasks_dir,f"{temp_task}.yaml")
        create_unitxt_pointer(temp_tasks_dir)
        create_unitxt_yaml(yaml_file=yaml_file, unitxt_recipe=unitxt_recipe, task_name=temp_task)
        return temp_task,temp_tasks_dir

    def remove_temp_files(self):
        if self.tasks_dir.startswith('temp_'): #to avoid unintended deletion if this class is inherited
            shutil.rmtree(self.tasks_dir)
        else:
            logger.warning("unitxt tasks dir did not start with 'temp_' and therefor was not deleted")

    def run(self,server_url: str | None = None) -> tuple:
        """
        Runs evaluation

        Returns:
            overall_scores       Average scores for the task group
            individual_scores   Individual scores for each task in the task group
        """
        logger.debug(locals())
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        results = self._run_mmlu(server_url=server_url, return_all_results=True)
        taskname = self.tasks[0]
        global_scores = results['results'][taskname]
        global_scores.pop('alias')
        try:
            instances = results['samples'][taskname]
            instance_scores = {}
            metrics = [metric.replace('metrics.','') for metric in instances[0]['doc']['metrics']]
            for i,instance in enumerate(instances):
                scores = {}
                for metric in metrics:
                    scores[metric] = instance[metric][0]
                instance_scores[i] = scores
        except Exception as e:
            logger.error("Error in extracting single instance scores")
            logger.error(e)
            logger.error(e.__traceback__)
            instance_scores = None
        self.remove_temp_files()
        return global_scores,instance_scores


def create_unitxt_yaml(yaml_file,unitxt_recipe, task_name):
    data = {
    'task': f'{task_name}',
    'include': 'unitxt',
    'recipe': f'{unitxt_recipe}'
    }
    with open(yaml_file, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    logger.info(f"task {task} unitxt recipe written to {yaml_file}")

def create_unitxt_pointer(tasks_dir):
    class_line = "class: !function " + task.__file__.replace("task.py", "task.Unitxt")
    output_file = os.path.join(tasks_dir,'unitxt')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(class_line)
    logger.info(f"Unitxt task pointer written to {output_file}")
