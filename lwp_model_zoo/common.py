import os
from torch import hub


def get_torchhub_dir(
    repo_owner: str = "a-working-title", repo_branch: str = "main"
) -> str:
    lwp_repo_name: str = "lwp-model-zoo"

    normalized_br = repo_branch.replace("/", "_")
    return os.path.join(
        hub.get_dir(), "_".join([repo_owner, lwp_repo_name, normalized_br,]),
    )

