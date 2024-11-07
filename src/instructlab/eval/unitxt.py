"""
Unitxt - Flexible, Shareable and Reusable Data Preparation and Evaluation for Generative AI
https://github.com/IBM/unitxt
https://arxiv.org/abs/2401.14019
"""

# Standard
from uuid import uuid4
import os
import shutil

# Third Party
from lm_eval.tasks.unitxt import task
import yaml

# First Party
from instructlab.eval.mmlu import MMLUBranchEvaluator

# Local
from .logger_config import setup_logger

logger = setup_logger(__name__)

TEMP_DIR_PREFIX = "unitxt_temp"


class UnitxtEvaluator(MMLUBranchEvaluator):
    """
    An evaluator class, running Unitxt evaluation

    Attributes:
        model_path      Absolute path to or name of a huggingface model
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
        unitxt_task = self.assign_task_name()
        tasks_dir = self.assign_tasks_dir(unitxt_task)
        super().__init__(
            model_path=model_path, tasks_dir=tasks_dir, tasks=[unitxt_task], few_shots=0
        )
        self.unitxt_recipe = unitxt_recipe

    def assign_tasks_dir(self, task_name):
        return os.path.join("eval_output", f"{TEMP_DIR_PREFIX}_{task_name}")

    def assign_task_name(self):
        return str(uuid4())

    def prepare_unitxt_files(self) -> None:
        taskname = self.tasks[0]
        yaml_file = os.path.join(str(self.tasks_dir), f"{taskname}.yaml")
        create_unitxt_pointer(self.tasks_dir)
        create_unitxt_yaml(
            yaml_file=yaml_file, unitxt_recipe=self.unitxt_recipe, task_name=taskname
        )

    def remove_unitxt_files(self):
        if self.tasks_dir.startswith(
            TEMP_DIR_PREFIX
        ):  # to avoid unintended deletion if this class is inherited
            shutil.rmtree(self.tasks_dir)
        else:
            logger.warning(
                "unitxt tasks dir '%s' did not start with '%s' prefix and therefore was not deleted",
                self.tasks_dir,
                TEMP_DIR_PREFIX,
            )

    def run(self, server_url: str | None = None) -> tuple:
        """
        Runs evaluation

        Attributes:
            server_url(str|None)    Model server endpoint (Ex: http://localhost:8000/v1) for the model being evaluated

        Returns:
            overall_scores      Average scores for the task group
            individual_scores   Individual scores for each task in the task group
        """
        self.prepare_unitxt_files()
        logger.debug(locals())
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        try:
            results = self._run_mmlu(server_url=server_url)
            taskname = self.tasks[0]
            global_scores = results["results"][taskname]
            global_scores.pop("alias")
            try:
                instances = results["samples"][taskname]
                instance_scores = {}
                metrics = [
                    metric.replace("metrics.", "")
                    for metric in instances[0]["doc"]["metrics"]
                ]
                for i, instance in enumerate(instances):
                    scores = {}
                    for metric in metrics:
                        scores[metric] = instance[metric][0]
                    instance_scores[i] = scores
            except KeyError as e:
                logger.error("Error in extracting single instance scores")
                logger.error(e)
                logger.error(e.__traceback__)
                instance_scores = None
        finally:
            self.remove_unitxt_files()
        return global_scores, instance_scores


def create_unitxt_yaml(yaml_file: str, unitxt_recipe: str, task_name: str) -> None:
    data = {"task": task_name, "include": "unitxt", "recipe": unitxt_recipe}
    with open(yaml_file, "w", encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False)
    logger.debug("task %s unitxt recipe written to %s", task_name, yaml_file)


def create_unitxt_pointer(tasks_dir):
    class_line = "class: !function " + task.__file__.replace("task.py", "task.Unitxt")
    output_file = os.path.join(tasks_dir, "unitxt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(class_line)
    logger.debug("Unitxt task pointer written to %s", output_file)
