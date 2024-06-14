# SPDX-License-Identifier: Apache-2.0


class Evaluator:
    """
    Parent class for Evaluators

    Atttributes:
        model   The model to be evaluated
    """

    def __init__(self, model: str) -> None:
        self.model = model
