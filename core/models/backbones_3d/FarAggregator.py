from mmdet3d.models.builder import BACKBONES
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule
from mmdet3d.models import builder
import torch.nn as nn
import torch.nn.functional as F


@BACKBONES.register_module()
class FarAggregator(BaseModule):
    def __init__(
        self,
        far_encoder_backbone=None,

    ):
        super().__init__()

        self.far_encoder_backbone = builder.build_backbone(far_encoder_backbone)
        
    
    def forward(self, x):

       x=self.far_encoder_backbone(x)
       
       return x
    