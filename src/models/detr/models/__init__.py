from .backbone import *
from .detr import *
from .matcher import *
from .position_encoding import *
from .segmentation import *
from .transformer import *

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build


def build_model(args):
    return build(args)
