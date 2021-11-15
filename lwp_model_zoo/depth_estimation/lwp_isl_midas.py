from typing import Final, List
import torch
from torch import hub


class ModelInfo:
    REPO_OWNER: Final[str] = "intel-isl"
    REPO_NAME: Final[str] = "MiDaS"
    REPO_BRANCH: Final[str] = "master"
    ISL_MIDAS_MODELS: Final[List[str]] = ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]
    ISL_DEFAULT_MODEL: Final[str] = ISL_MIDAS_MODELS[0]

    def __init__(self, mdl: str = "", use_cuda: bool = True):
        super().__init__()
        try:
            if mdl == "":
                mdl = self.ISL_DEFAULT_MODEL
            if mdl not in self.ISL_MIDAS_MODELS:
                raise ValueError(f"Failed to find {mdl} in the published model names")
            self.__model_name = mdl
        except ValueError as _err:
            raise ValueError(f"invalid model name: {mdl}") from _err
        self.__midas = None
        self.__transforms = None
        self.__gh_repo = f"{self.REPO_OWNER}/{self.REPO_NAME}:{self.REPO_BRANCH}"
        self.__use_cuda = use_cuda

    @property
    def model_name(self):
        return self.__model_name

    @property
    def midas(self):
        if not self.__midas:
            return None
        self.__midas.eval()
        device = torch.device("cpu")
        if self.__use_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
        elif self.__use_cuda:
            print("Warn: CUDA is not available; Using CPU fallback")
        self.__midas.to(device)

        return self.__midas

    @property
    def transforms(self):
        if not self.__transforms:
            return None
        print(self.ISL_MIDAS_MODELS[-1])
        if self.__model_name == self.ISL_MIDAS_MODELS[-1]:
            return self.__transforms.small_transform
        return self.__transforms.dpt_transform

    @property
    def gh_repo(self):
        return self.__gh_repo

    def download_models(self):
        self.__midas = hub.load(self.gh_repo, self.__model_name)
        self.__transforms = hub.load(self.gh_repo, "transforms")


def isl_midas(model_name="", use_cuda=True, **kwargs):
    """
    Depth estimation models by Intel ISL
    model_type (string): Optional. Should be one of the following options:
        DPT_Large (defualt),
        DPT_Hybrid,
        MiDaS_small
    use_cuda (bool): Optional. If True (default), CUDA acceleration will be used.
    """

    model = ModelInfo(mdl=model_name, use_cuda=use_cuda)
    model.download_models()

    return model.midas, model.transforms
