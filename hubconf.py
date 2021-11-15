# -*- coding: utf-8 -*- #
# !/usr/bin/env python3

from torch import hub

from lwp_model_zoo.semantic_segmentation.lwp_mit_semseg import mit_semseg
from lwp_model_zoo.depth_estimation.lwp_isl_midas import isl_midas

dependencies = ["mit_semseg", "scipy", "torch", "yacs"]
hub._validate_not_a_forked_repo = lambda a, b, c: True
