import sys

sys.path.append("src/models/detr/")

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .config import add_detr_config
from .detr import Detr
from .dataset_mapper import DetrDatasetMapper
from .trainer import TrainerDETR
