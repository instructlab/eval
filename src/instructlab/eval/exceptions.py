# SPDX-License-Identifier: Apache-2.0


class EvalError(Exception):
    """
    Parent class for all of instructlab-eval exceptions
    """


class ModelNotFoundError(EvalError):
    """
    Exception raised when model is not able to be found

    Attributes
        message     error message to be printed on raise
        model       model that is being operated on
        path        filepath of model location
    """

    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        self.model = path.rsplit("/")[-1]
        self.message = f"Model {self.model} could not be found at {self.path}"
