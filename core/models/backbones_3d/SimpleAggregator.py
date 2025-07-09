from mmdet3d.models.builder import BACKBONES
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule
from mmdet3d.models import builder
import torch.nn as nn
import torch.nn.functional as F


@BACKBONES.register_module()
class SimpleAggregator(BaseModule):
    def __init__(
        self,
        embed_dims=128,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
    ):
        super().__init__()

        self.conv3d_1 =  nn.Sequential(
                nn.Conv3d(embed_dims, embed_dims*2, kernel_size=(3, 3, 3), padding=(1,1,0),stride=(2,2,1)), 
                build_norm_layer(norm_cfg, embed_dims*2)[1], nn.ReLU(inplace=True))
        
        self.deconv3d_1 = nn.Sequential(
                nn.ConvTranspose3d(embed_dims*2, embed_dims, kernel_size=(3, 3, 3), padding=(1,1,0),stride=(2,2,1),output_padding=(1,1,0)), 
                build_norm_layer(norm_cfg, embed_dims)[1], nn.ReLU(inplace=True))
        
    
    def forward(self, x):

       x=self.conv3d_1(x)
       x=self.deconv3d_1(x)
       
       return x
    