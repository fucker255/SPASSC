from mmdet3d.models.builder import BACKBONES
import torch
from mmcv.runner import BaseModule
from mmdet3d.models import builder
import torch.nn as nn
import torch.nn.functional as F

@BACKBONES.register_module()
class Fuser(BaseModule):
    def __init__(
        self,
        # embed_dims=128,
        global_aggregator=None,
        local_aggregator=None,
        far_aggregator=None,
        
    ):
        super().__init__()
        self.global_aggregator = builder.build_backbone(global_aggregator)
        self.local_aggregator = builder.build_backbone(local_aggregator)
        self.far_aggregator = builder.build_backbone(far_aggregator)
        # 可学习参数，初值为0（sigmoid后为0.5）
        self.ratio_param = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
    
    def forward(self, x):
        B, C, H, W, Z = x.shape
        ratio = torch.sigmoid(self.ratio_param)  # (0,1)
        z_range = torch.linspace(0, 1, Z, device=x.device).view(1, 1, 1, 1, Z)
        k = 40  # 控制过渡带宽，越大越接近硬切分
        mask = torch.sigmoid((z_range - ratio) * k)  # 可微分的soft mask
        mask = mask.expand(B, 1, 1, 1, Z)

        global_feats = self.global_aggregator(x)
        far_feats=self.far_aggregator(x)
        local_feats = self.local_aggregator(far_feats)
        out = mask * global_feats + (1 - mask) * local_feats
        return out