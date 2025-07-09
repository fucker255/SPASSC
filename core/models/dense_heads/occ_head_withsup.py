import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from mmcv.cnn import build_conv_layer, build_norm_layer
from core.utils.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss

@HEADS.register_module()
class OccHeadWithSup(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channel,
        empty_idx=0,
        num_level=1,
        with_cp=True,
        occ_size=[256, 256, 32],
        loss_weight_cfg=None,
        balance_cls_weight=True,
        cls_weights=None,
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        class_frequencies=None,
        train_cfg=None,
        test_cfg=None
    ):
        super(OccHeadWithSup, self).__init__()
        
        if type(in_channels) is not list:
            in_channels = [in_channels]
        
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_level = num_level
        self.empty_idx = empty_idx

        self.with_cp = with_cp
        
        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        
        self.occ_size = occ_size
        # voxel losses
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)

        self.occ_convs = nn.ModuleList()
        for i in range(self.num_level):
            mid_channel = self.in_channels[i] // 2
            occ_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=self.in_channels[i],
                    out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, mid_channel)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=mid_channel, 
                    out_channels=out_channel, kernel_size=1, stride=1, padding=0),              
            )
            self.occ_convs.append(occ_conv)

        
        # loss functions
        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(np.array(class_frequencies) + 0.001))
        else:
            self.class_weights = torch.tensor(cls_weights)  
        
        self.supconv=nn.Conv1d(self.in_channels[0]+out_channel,out_channel,1)
    
    def forward(self, voxel_feats, img_metas=None, img_feats=None, gt_occ=None,proposal_padded=None):
        assert type(voxel_feats) is list and len(voxel_feats) == self.num_level

        output_occs = []
        for feats, occ_conv in zip(voxel_feats, self.occ_convs):
            bs, _, H, W, Z = feats.shape
            if self.with_cp:
                occ_output=torch.utils.checkpoint.checkpoint(occ_conv, feats)
                output_occs.append(occ_output)
            else:
                output_occs.append(occ_conv(feats))
            
            if self.training and self.supconv:
                supfeats_flatten=self.supconv(torch.cat((feats, output_occs[0]), dim=1).flatten(2))
            else:
                supfeats_flatten = None 
        
        result = {
            'output_voxels': F.interpolate(output_occs[0], size=self.occ_size, mode='trilinear', align_corners=False).contiguous(),
            'supfeats_flatten':supfeats_flatten
        }
        return result
    
    def loss(self, output_voxels, target_voxels):
        loss_dict = {}
        loss_dict['loss_voxel_ce'] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
        loss_dict['loss_voxel_sem_scal'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
        loss_dict['loss_voxel_geo_scal'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)

        return loss_dict
    
    def geo_loss(self, output_voxels, target_voxels):
        target_voxels = torch.where(target_voxels > 0, 1, target_voxels)
        loss_dict = {}
        loss_dict['loss_voxel_ce'] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, ignore_index=255)
        loss_dict['loss_voxel_geo_scal'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)

        return loss_dict