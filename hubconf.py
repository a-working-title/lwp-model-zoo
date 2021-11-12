# -*- coding: utf-8 -*- #
# !/usr/bin/env python3

import torch
from torch import hub

from lwp_model_zoo.semantic_segmentation.lwp_mit_semseg import mit_semseg

dependencies = ["mit_semseg", "scipy", "torch", "yacs"]
hub._validate_not_a_forked_repo = lambda a, b, c: True


ISL_MIDAS_MODEL_TYPES = ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]
ISL_MIDAS_DEFAULT_MODEL_TYPE = ISL_MIDAS_MODEL_TYPES[0]


def isl_midas(model_type=ISL_MIDAS_DEFAULT_MODEL_TYPE, use_cuda=True, **kwargs):
    """
    Depth estimation models by Intel ISL
    model_type (string): Optional. Should be one of the following options:
        DPT_Large,
        DPT_Hybrid,
        MiDaS_small
    use_cuda (bool): Optional. If True, CUDA acceleration will be used (True by default).
    """
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
