# SPDX-License-Identifier: Apache-2.0


class Evaluator:
    """
    Parent class for Evaluators

    Atttributes:
        model_path   Path to the model to be evaluated
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
