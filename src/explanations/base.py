from torch import nn
from typing import Dict, List
import torch

from detectron2.structures.instances import Instances


class BaseExplainer(object):
    def __init__(self, model: nn.Module, model_name: str = "faster_r_cnn"):
        self.model = model
        self.model_name = model_name
        self.model.eval()

    def __call__(
        self, inputs: List[Dict[str, torch.Tensor]], target_results: List[Instances]
    ):
        raise NotImplementedError
