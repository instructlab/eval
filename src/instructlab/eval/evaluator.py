# SPDX-License-Identifier: Apache-2.0


class Evaluator:
    """
    Parent class for Evaluators
    """

    def __init__(self, model) -> None:
        self.model = model

    def run(self) -> dict:
        return {}
