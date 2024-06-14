# SPDX-License-Identifier: Apache-2.0


class EvalError(Exception):
    """
    Parent class for all of instructlab-eval exceptions
    """


class ModelNotFoundError(EvalError):
    """
    Exception raised when model is not able to be found

    Attributes
        model   model that is being operated on
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.message = f"Model {self.model} could not be found"
