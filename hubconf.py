# -*- coding: utf-8 -*- #
#!/usr/bin/env python3

import os
from shutil import rmtree
from torch import hub
from torch import nn
from torch.utils.model_zoo import load_url
from typing import Final, Tuple
from urllib import request

dependencies = ["torch"]
hub._validate_not_a_forked_repo = lambda a, b, c: True

MIT_SEMSEG_DEFAULT_MODEL_NAME: Final = "ade20k-resnet101dilated-ppm_deepsup"


def _download_mit_sem_seg(
    base_dir, model_name=MIT_SEMSEG_DEFAULT_MODEL_NAME, clean_slate=False
) -> Tuple[str, str, str]:
    BASE_URL: Final = "http://sceneparsing.csail.mit.edu/model/pytorch/"
    DECODER_FMT: Final = "decoder_epoch_{}.pth"
    ENCODER_FMT: Final = "encoder_epoch_{}.pth"
    CFG_URL_FMT: Final = "https://raw.githubusercontent.com/CSAILVision/semantic-segmentation-pytorch/master/config/{}.yaml"
    model_pairs: Final = {
        "ade20k-hrnetv2-c1": (
            DECODER_FMT.format(30),
            ENCODER_FMT.format(30),
            CFG_URL_FMT.format("ade20k-hrnetv2"),
        ),
        "ade20k-mobilenetv2dilated-c1_deepsup": (
            DECODER_FMT.format(20),
            ENCODER_FMT.format(20),
            CFG_URL_FMT.format("ade20k-mobilenetv2dilated-c1_deepsup"),
        ),
        "ade20k-resnet18dilated-c1_deepsup": (
            DECODER_FMT.format(20),
            ENCODER_FMT.format(20),
            CFG_URL_FMT.format("ade20k-resnet18dilated-ppm_deepsup"),
        ),
        "ade20k-resnet18dilated-ppm_deepsup": (
            DECODER_FMT.format(20),
            ENCODER_FMT.format(20),
            None,
        ),
        "ade20k-resnet50-upernet": (
            DECODER_FMT.format(30),
            ENCODER_FMT.format(30),
            CFG_URL_FMT.format("ade20k-resnet50-upernet"),
        ),
        "ade20k-resnet50dilated-ppm_deepsup": (
            DECODER_FMT.format(20),
            ENCODER_FMT.format(20),
            CFG_URL_FMT.format("ade20k-resnet50dilated-ppm_deepsup"),
        ),
        "ade20k-resnet101-upernet": (
            DECODER_FMT.format(50),
            ENCODER_FMT.format(50),
            CFG_URL_FMT.format("ade20k-resnet101-upernet"),
        ),
        "ade20k-resnet101dilated-ppm_deepsup": (
            DECODER_FMT.format(25),
            ENCODER_FMT.format(25),
            CFG_URL_FMT.format("ade20k-resnet101dilated-ppm_deepsup"),
        ),
    }

    if model_name not in model_pairs.keys():
        return None

    cache_dir = os.path.join(base_dir, model_name)
    if clean_slate and os.path.exists(cache_dir):
        rmtree(cache_dir)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    decoder, encoder, cfg = model_pairs[model_name]
    load_url(BASE_URL + model_name + "/" + decoder, model_dir=cache_dir)
    load_url(BASE_URL + model_name + "/" + encoder, model_dir=cache_dir)

    cfg_path = os.path.join(cache_dir, "{}.yaml".format(model_name))
    if cfg:
        request.urlretrieve(cfg, cfg_path)

    decoder_path = os.path.join(cache_dir, decoder)
    encoder_path = os.path.join(cache_dir, encoder)

    return (
        decoder_path if os.path.exists(decoder_path) else None,
        encoder_path if os.path.exists(encoder_path) else None,
        cfg_path if os.path.exists(cfg_path) else None,
    )


def mit_semseg(
    model_name=MIT_SEMSEG_DEFAULT_MODEL_NAME, clean_slate=False, use_cuda=True, **kwargs
):
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
    clean_state (bool): Optional. If True, the cached model files will be
        deleted and newly downloaded from the repository (True by default).
    use_cuda (bool): Optional. If True, CUDA acceleration will be used (True by default).
    """
    try:
        from mit_semseg.config import cfg as default_cfg
        from mit_semseg.dataset import TestDataset
        from mit_semseg.models import ModelBuilder, SegmentationModule
    except (ModuleNotFoundError, ImportError) as err:
        print(f"err.__class__.__name__ : {err.msg}")

    base_dir = os.path.join(hub.get_dir(), "lwp/mit_semseg/models")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    decoder_path, encoder_path, cfg_path = _download_mit_sem_seg(
        base_dir, model_name, clean_slate=clean_slate
    )

    if decoder_path is None or encoder_path is None:
        return None

    if cfg_path is None:
        if model_name != "ade20k-resnet18dilated-c1_deepsup":
            return None
        cfg = default_cfg
    else:
        cfg = default_cfg.clone()
        cfg.merge_from_file(cfg_path)

    decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=decoder_path,
        use_softmax=True,
    )

    encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=encoder_path,
    )

    crit = nn.NLLLoss(ignore_index=-1)
    seg_module = SegmentationModule(encoder, decoder, crit)
    seg_module.eval()
    if use_cuda:
        seg_module.cuda()

    return seg_module
