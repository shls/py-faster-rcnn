# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Fast R-CNN config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import sdha_cfg
sdha_cfg = __C

__C.two_category = True
__C.seven_category = False
__C.temporal = True
__C.spatial = False
__C.two_stream = False
__C.subdataset = 'mhi10'
__C.ptroot = 'temporal'
__C.channels = 1
__C.GTX780 = False
__C.GTX780_root = '/mnt/naruto/data_sets/sdha/rcnn'
__C.GTX980 = True
__C.GTX980_root = '/extra/ls/sdha/rcnn'
