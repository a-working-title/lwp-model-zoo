# -*- coding: utf-8 -*- #
#!/usr/bin/env python3

import csv
from genericpath import exists
import os
import tempfile
import torch
from torch import hub
from torch import nn
from torch.utils.model_zoo import load_url
from scipy.io import loadmat
from typing import Final, Tuple
from urllib import error as urllib_err
from urllib.parse import urlparse

dependencies = ["torch", "scipy"]
hub._validate_not_a_forked_repo = lambda a, b, c: True

MIT_SEMSEG_DEFAULT_MODEL_NAME: Final = "ade20k-resnet101dilated-ppm_deepsup"


def _download_mit_sem_seg(
    model_name=MIT_SEMSEG_DEFAULT_MODEL_NAME,
) -> Tuple[str, str, str]:
    BASE_URL: Final = "http://sceneparsing.csail.mit.edu/model/pytorch/"
    GH_BASE_URL: Final = "https://raw.githubusercontent.com/CSAILVision/semantic-segmentation-pytorch/master/"
    DECODER_FMT: Final = "{}_decoder_epoch_{}.pth"
    ENCODER_FMT: Final = "{}_encoder_epoch_{}.pth"
    CFG_URL_FMT: Final = GH_BASE_URL + "config/{}.yaml"
    COLOR_MAT_URL: Final = GH_BASE_URL + "data/color150.mat"
    LABEL_CSV_URL: Final = GH_BASE_URL + "data/object150_info.csv"
    MODEL_INFOS: Final = [
        ("ade20k-hrnetv2-c1", 30, "ade20k-hrnetv2"),
        (
            "ade20k-mobilenetv2dilated-c1_deepsup",
            20,
            "ade20k-mobilenetv2dilated-c1_deepsup",
        ),
        ("ade20k-resnet18dilated-c1_deepsup", 20, None),
        (
            "ade20k-resnet18dilated-ppm_deepsup",
            20,
            "ade20k-resnet18dilated-ppm_deepsup",
        ),
        ("ade20k-resnet50-upernet", 30, "ade20k-resnet50-upernet"),
        (
            "ade20k-resnet50dilated-ppm_deepsup",
            20,
            "ade20k-resnet50dilated-ppm_deepsup",
        ),
        ("ade20k-resnet101-upernet", 50, "ade20k-resnet101-upernet"),
        (
            "ade20k-resnet101dilated-ppm_deepsup",
            25,
            "ade20k-resnet101dilated-ppm_deepsup",
        ),
    ]

    found = False
    for _model_info in MODEL_INFOS:
        if model_name in _model_info:
            found = True
    if not found:
        return None, None, None

    model_pairs = {
        m: (
            DECODER_FMT.format(m, e),
            ENCODER_FMT.format(m, e),
            CFG_URL_FMT.format(c) if c is not None else None,
        )
        for m, e, c in MODEL_INFOS
    }

    decoder_name, encoder_name, cfg_url = model_pairs[model_name]
    load_url(
        BASE_URL + model_name + "/" + decoder_name[decoder_name.find("decoder") :],
        file_name=decoder_name,
    )
    load_url(
        BASE_URL + model_name + "/" + encoder_name[encoder_name.find("encoder") :],
        file_name=encoder_name,
    )
    cfg_path = os.path.join(tempfile.gettempdir(), model_name + ".yaml")
    try:
        os.remove(cfg_path)
    except:
        pass
    if cfg_url:
        try:
            hub.download_url_to_file(cfg_url, cfg_path)
        except (
            urllib_err.ContentTooShortError,
            urllib_err.HTTPError,
            urllib_err.URLError,
        ) as err:
            print(f"{err.__class__.__name__}: {err.msg}")
            cfg_path = None

    color_mat_path = os.path.join(
        tempfile.gettempdir(), os.path.basename(urlparse(COLOR_MAT_URL).path)
    )
    try:
        os.remove(color_mat_path)
    except:
        pass
    finally:
        try:
            hub.download_url_to_file(COLOR_MAT_URL, color_mat_path)
        except (
            urllib_err.ContentTooShortError,
            urllib_err.HTTPError,
            urllib_err.URLError,
        ) as err:
            print(f"{err.__class__.__name__}: {err.msg}")
            color_mat_path = None

    label_csv_path = os.path.join(
        tempfile.gettempdir(), os.path.basename(urlparse(LABEL_CSV_URL).path)
    )
    try:
        os.remove(label_csv_path)
    except:
        pass
    finally:
        try:
            hub.download_url_to_file(LABEL_CSV_URL, label_csv_path)
        except (
            urllib_err.ContentTooShortError,
            urllib_err.HTTPError,
            urllib_err.URLError,
        ) as err:
            print(f"{err.__class__.__name__}: {err.msg}")
            label_csv_path = None

    decoder_path = os.path.join(hub.get_dir(), "checkpoints", decoder_name)
    encoder_path = os.path.join(hub.get_dir(), "checkpoints", encoder_name)

    return (
        decoder_path if os.path.exists(decoder_path) else None,
        encoder_path if os.path.exists(encoder_path) else None,
        cfg_path if cfg_path is not None and os.path.exists(cfg_path) else None,
        color_mat_path
        if color_mat_path is not None and os.path.exists(color_mat_path)
        else None,
        label_csv_path
        if label_csv_path is not None and os.path.exists(label_csv_path)
        else None,
    )


def mit_semseg(model_name=MIT_SEMSEG_DEFAULT_MODEL_NAME, use_cuda=True, **kwargs):
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
    use_cuda (bool): Optional. If True, CUDA acceleration will be used (True by default).
    """
    try:
        from mit_semseg.config import cfg as default_cfg
        from mit_semseg.dataset import TestDataset
        from mit_semseg.models import ModelBuilder, SegmentationModule
    except (ModuleNotFoundError, ImportError) as err:
        print(f"{err.__class__.__name__} : {err.msg}")

    (
        decoder_path,
        encoder_path,
        cfg_path,
        color_mat_path,
        label_path,
    ) = _download_mit_sem_seg(model_name)
    if all(
        _path is None
        for _path in [decoder_path, encoder_path, color_mat_path, label_path]
    ):
        return None, None, None
    if cfg_path is None:
        if model_name != "ade20k-resnet18dilated-c1_deepsup":
            return None, None, None
        cfg = default_cfg.clone()
        cfg.MODEL.arch_encoder = "resnet18dilated"
        cfg.MODEL.fc_dim = 512
        cfg.MODEL.arch_decoder = "c1_deepsup"
    else:
        cfg = default_cfg.clone()
        cfg.merge_from_file(cfg_path)

    labels = {}
    with open(label_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            labels[int(row[0])] = row[5].split(";")[0]

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

    return seg_module, loadmat(color_mat_path)["colors"], labels


ISL_MIDAS_MODEL_TYPES: Final = ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]
ISL_MIDAS_DEFAULT_MODEL_TYPE: Final = ISL_MIDAS_MODEL_TYPES[0]


def isl_midas(model_type=ISL_MIDAS_DEFAULT_MODEL_TYPE, use_cuda=True, **kwargs):
    """
    Depth estimation models by Intel ISL
    model_type (string): Optional. Should be one of the following options:
        DPT_Large,
        DPT_Hybrid,
        MiDaS_small
    use_cuda (bool): Optional. If True, CUDA acceleration will be used (True by default).
    """
    try:
        import timm
    except (ModuleNotFoundError, ImportError) as err:
        print(f"{err.__class__.__name__} : {err.msg}")
    if model_type not in ISL_MIDAS_MODEL_TYPES:
        return None, None
    midas = hub.load("intel-isl/MiDaS", model_type)
    midas.eval()
    if use_cuda:
        if not torch.cuda.is_available():
            print("Warn: CUDA is not available; Using CPU fallback")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

    midas.to(device)
    _transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "MiDaS_small":
        midas_transforms = _transforms.small_transform
    else:
        midas_transforms = _transforms.dpt_transform

    return midas, midas_transforms
