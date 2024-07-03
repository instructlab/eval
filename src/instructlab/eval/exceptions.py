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


class MMLUEvaluationError(EvalError):
    """
    Exception raised when MMLU evaluation of a model is not able to be completed

    Attributes
        message     error message to be printed on raise
        cause       root cause exception
        model       model being evaluated
    """

    def __init__(self, model, cause) -> None:
        super().__init__()
        self.model = model
        self.cause = cause
        self.message = f"Failed to run MMLU evaluation on {self.model}: {self.cause}"


class MMLUBranchEvaluationError(EvalError):
    """
    Exception raised when MMLUBranch evaluation of a model is not able to be completed

    Attributes
        message     error message to be printed on raise
        cause       root cause exception
        model       model being evaluated
        sdg_path    filepath of SDG data
    """

    def __init__(self, model, sdg_path, cause) -> None:
        super().__init__()
        self.model = model
        self.sdg_path = sdg_path
        self.cause = cause
        self.message = f"Failed to run MMLUBranch evaluation on {self.model} with {self.sdg_path}: {self.cause}"


class MTBenchEvaluationError(EvalError):
    """
    Exception raised when MT-Bench evaluation of a model is not able to be completed

    Attributes
        message     error message to be printed on raise
        cause       root cause exception
        model       model being evaluated
        judge_model model doing the evaluating
    """

    def __init__(self, model, judge_model, cause) -> None:
        super().__init__()
        self.model = model
        self.judge_model = judge_model
        self.cause = cause
        self.message = f"Failed to run MT-Bench evaluation on {self.model} with judge model {self.judge_model}: {self.cause}"


class MTBenchBranchEvaluationError(EvalError):
    """
    Exception raised when MT-Bench Branch evaluation of a model is not able to be completed

    Attributes
        message     error message to be printed on raise
        cause       root cause exception
        model       model being evaluated
        judge_model model doing the evaluating
    """

    def __init__(self, model, judge_model, cause) -> None:
        super().__init__()
        self.model = model
        self.judge_model = judge_model
        self.cause = cause
        self.message = f"Failed to run MT-Bench Branch evaluation on {self.model} with judge model {self.judge_model}: {self.cause}"
