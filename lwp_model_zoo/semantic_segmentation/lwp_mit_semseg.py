import csv
import errno
import os
from typing import List, Optional, Tuple
from urllib import error as urllib_err
from torch import nn
from torch.hub import get_dir, download_url_to_file
from torch.utils.model_zoo import load_url
from mit_semseg.config import cfg as default_cfg
from mit_semseg.models import ModelBuilder, SegmentationModule
from yacs.config import CfgNode
from scipy.io import loadmat

from lwp_model_zoo.common import get_torchhub_dir
from lwp_model_zoo.semantic_segmentation.lwp_mit_semseg_config import Config


class ModelInfo(Config):
    def __init__(self, mdl: str):
        super().__init__()
        self.__hub_dir = get_torchhub_dir()
        try:
            if os.path.exists(self.__hub_dir) and not os.path.isdir(self.__hub_dir):
                raise ValueError(f"{self.__hub_dir} exists but is not a directory")
            if mdl == "":
                mdl = super().DEFAULT_MODEL_NAME
            if mdl not in self.model_names:
                raise ValueError(f"Failed to find {mdl} in model_names")
            self.__model_name = mdl
            self.__epoch = super().get_epoch(mdl)
        except ValueError as _err:
            raise ValueError(f"invalid model name: {mdl}") from _err
        self.__decoder_name = super().DECODER_FMT.format(
            self.__model_name, self.__epoch
        )
        self.__encoder_name = super().ENCODER_FMT.format(
            self.__model_name, self.__epoch
        )
        self.__rsc_path = os.path.join(self.__hub_dir, "resources")
        os.makedirs(self.__rsc_path, exist_ok=True)

    @property
    def decoder_name(self):
        return self.__decoder_name

    @property
    def encoder_name(self):
        return self.__encoder_name

    @property
    def rsc_path(self):
        return self.__rsc_path

    def download_models(self) -> Tuple[str, ...]:
        models = [self.__decoder_name, self.__encoder_name]
        num_models = len(models)
        paths = []
        for model in models:
            if model.find("decoder") != -1:
                idx = model.find("decoder")
            else:
                idx = model.find("encoder")
            url = super().BASE_URL + model[: (idx - 1)] + "/" + model[idx:]
            load_url(url, file_name=model)
            path = os.path.join(get_dir(), "checkpoints", model)
            if os.path.exists(path):
                paths.append(path)

        if len(paths) != num_models:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

        return tuple(paths)

    def get_config(self) -> Optional[CfgNode]:
        try:
            cfg_url = super().get_cfg_url(self.__model_name)
        except ValueError as _err:
            print(f"{_err.__class__.__name__} :", str(_err))
            return None
        cfg_path = os.path.join(self.__rsc_path, self.__model_name + ".yaml")
        cfg = default_cfg.clone()
        if cfg_url is None:
            if self.__model_name != "ade20k-resnet18dilated-c1_deepsup":
                return None
            cfg.MODEL.arch_encoder = "resnet18dilated"
            cfg.MODEL.fc_dim = 512
            cfg.MODEL.arch_decoder = "c1_deepsup"
            return cfg

        if not os.path.exists(cfg_path):
            try:
                download_url_to_file(cfg_url, cfg_path)
            except (
                urllib_err.ContentTooShortError,
                urllib_err.HTTPError,
                urllib_err.URLError,
            ) as err:
                print(f"{err.__class__.__name__}: Failed to download {cfg_url}")
                return None

        cfg.merge_from_file(cfg_path)
        return cfg

    def get_color_map(self):
        path = os.path.join(self.__rsc_path, "color150.mat")
        if not os.path.exists(path):
            url = super().get_color_mat_url()
            try:
                download_url_to_file(url, path)
            except (
                urllib_err.ContentTooShortError,
                urllib_err.HTTPError,
                urllib_err.URLError,
            ) as err:
                print(f"{err.__class__.__name__}: Failed to download {path}")
                return None
        return loadmat(path)["colors"]

    def get_labels(self):
        path = os.path.join(self.__rsc_path, "object150_info.csv")
        if not os.path.exists(path):
            url = super().get_label_csv_url()
            try:
                download_url_to_file(url, path)
            except (
                urllib_err.ContentTooShortError,
                urllib_err.HTTPError,
                urllib_err.URLError,
            ) as err:
                print(f"{err.__class__.__name__}: Failed to download {path}")
                return None
        labels = {}
        with open(path, encoding="utf-8") as f_label:
            reader = csv.reader(f_label)
            next(reader)
            for row in reader:
                labels[int(row[0])] = row[5].split(";")[0]
        return labels

    def list_models(self) -> List[str]:
        return self.model_names


def mit_semseg(model_name="", use_cuda=True, **kwargs):
    """
    Semantic segmentation models on MIT ADE20K scene parsing dataset
    model_name (string): Optional. if it is given, load one of the following models;
        ade20k-hrnetv2-c1,
        ade20k-mobilenetv2dilated-c1_deepsup,
        ade20k-resnet18dilated-c1_deepsup,
        ade20k-resnet18dilated-ppm_deepsup,
        ade20k-resnet50-upernet,
        ade20k-resnet50dilated-ppm_deepsup,
        ade20k-resnet101-upernet,
        ade20k-resnet101dilated-ppm_deepsup (default).
    use_cuda (bool): Optional. If True (default), CUDA acceleration will be used.
    keyword arguments:
        sub_command (str): Optional. If given, do the given sub-command rather than returning fully loaded models.
            "list_models": Get the list of models in this model category
            "download_only": Just download and cache the given models
    """
    sub_cmd = ""
    if "sub_command" in kwargs:
        sub_cmd = kwargs.get("sub_command")

    if sub_cmd:
        if sub_cmd == "list_models":
            return ModelInfo(mdl=model_name).list_models()
        if sub_cmd == "download_only":
            if not model_name:
                print(
                    "Invalid argument:",
                    "download_only mode requires model_name explicitly given",
                )
                return None
            _model = ModelInfo(mdl=model_name)
            _ = _model.download_models()
            return None
        print(f"Err: {sub_cmd} is not a valid sub command")
        return None
    model = ModelInfo(model_name)
    decoder_path, encoder_path = model.download_models()
    cfg = model.get_config()  # type: Optional[CfgNode]
    decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=decoder_path,
        use_softmax=True,
    )

    encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder, fc_dim=cfg.MODEL.fc_dim, weights=encoder_path,
    )
    crit = nn.NLLLoss(ignore_index=-1)
    seg_module = SegmentationModule(encoder, decoder, crit)
    seg_module.eval()
    if use_cuda:
        seg_module.cuda()

    return seg_module
