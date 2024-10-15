# SPDX-License-Identifier: Apache-2.0


class EvalError(Exception):
    """
    Parent class for all of instructlab-eval exceptions
    """


class ModelNotFoundError(EvalError):
    """
    Error raised when model is not able to be found

    Attributes
        message     error message to be printed on raise
        path        filepath of model location
    """

    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        self.message = f"Model could not be found at {path}"


class InvalidModelError(EvalError):
    """
    Error raised when model can be found but is invalid

    Attributes
        message     error message to be printed on raise
        path        filepath of model location
        reason      root cause for model invalidity
    """

    def __init__(self, path, reason) -> None:
        super().__init__()
        self.path = path
        self.reason = reason
        self.message = f"Model found at {path} but was invalid due to: {reason}"


class InvalidMaxWorkersError(EvalError):
    """
    Error raised when max_workers isn't an int or "auto"

    Attributes
        message         error message to be printed on raise
        max_workers     max_workers specified
    """

    def __init__(self, max_workers) -> None:
        super().__init__()
        self.max_workers = max_workers
        self.message = f"Invalid max_workers '{max_workers}' specified. Valid values are positive integers or 'auto'."


class InvalidGitRepoError(EvalError):
    """
    Error raised when taxonomy dir provided isn't a valid git repo
    Attributes
        message         error message to be printed on raise
        taxonomy_dir    supplied taxonomy directory
    """

    def __init__(self, taxonomy_dir) -> None:
        super().__init__()
        self.taxonomy_dir = taxonomy_dir
        self.message = f"Invalid git repo: {taxonomy_dir}"


class GitRepoNotFoundError(EvalError):
    """
    Error raised when taxonomy dir provided does not exist
    Attributes
        message         error message to be printed on raise
        taxonomy_dir    supplied taxonomy directory
    """

    def __init__(self, taxonomy_dir) -> None:
        super().__init__()
        self.taxonomy_dir = taxonomy_dir
        self.message = f"Taxonomy git repo not found: {taxonomy_dir}"


class InvalidGitBranchError(EvalError):
    """
    Error raised when branch provided is invalid
    Attributes
        message         error message to be printed on raise
        branch          supplied branch
    """

    def __init__(self, branch) -> None:
        super().__init__()
        self.branch = branch
        self.message = f"Invalid git branch: {branch}"


class TasksDirNotFoundError(EvalError):
    """
    Error raised when the tasks dir doesn't exist
    Attributes
        message         error message to be printed on raise
        tasks_dir       tasks dir
    """

    def __init__(self, tasks_dir) -> None:
        super().__init__()
        self.tasks_dir = tasks_dir
        self.message = f"Tasks dir not found: {tasks_dir}"


class InvalidTasksDirError(EvalError):
    """
    Error raised when the tasks dir is invalid
    Attributes
        message         error message to be printed on raise
        tasks_dir       tasks dir
    """

    def __init__(self, tasks_dir) -> None:
        super().__init__()
        self.tasks_dir = tasks_dir
        self.message = f"Invalid Tasks Dir: {tasks_dir}"


class ModelServingAPIError(EvalError):
    """
    Error raised when reply retrieval from model serving fails.
    Attributes
        message              error message to be printed on raise
    """

    def __init__(self) -> None:
        super().__init__()
        self.message = "Failed to receive a reply from model serving API."


class EmptyTaxonomyError(EvalError):
    """
    Error raised when taxonomy doesn't contain any skill QNAs
    Attributes
        message              error message to be printed on raise
    """

    def __init__(self) -> None:
        super().__init__()
        self.message = "Provided taxonomy doesn't contain any skill qna.yaml files"
