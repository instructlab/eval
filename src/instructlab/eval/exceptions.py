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


class SDGPathNotFoundError(EvalError):
    """
    Error raised when the sdg path doesn't exist
    Attributes
        message         error message to be printed on raise
        sdg_path        sdg path
    """

    def __init__(self, sdg_path) -> None:
        super().__init__()
        self.sdg_path = sdg_path
        self.message = f"SDG Path not found: {sdg_path}"


class InvalidSDGPathError(EvalError):
    """
    Error raised when the sdg path is invalid
    Attributes
        message         error message to be printed on raise
        sdg_path        sdg path
    """

    def __init__(self, sdg_path) -> None:
        super().__init__()
        self.sdg_path = sdg_path
        self.message = f"Invalid SDG Path: {sdg_path}"
