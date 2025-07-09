import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS
import torch.nn.functional as F
from .modules.Stereo_Depth_Net_modules import SimpleUnet, convbn_2d,DepthAggregation
from mmcv.runner import BaseModule, force_fp32
from .modules.Right_DepthNet_modules import DepthNet
from torch.cuda.amp.autocast_mode import autocast
from core.utils.gaussian import generate_guassian_depth_target
from torch.utils.checkpoint import checkpoint
import pdb


class StereoVolumeEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StereoVolumeEncoder, self).__init__()
        self.stem = convbn_2d(in_channels, out_channels, kernel_size=3, stride=1, pad=1)
        self.Unet = nn.Sequential(
            SimpleUnet(out_channels)
        )
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.Unet(x)
        x = self.conv_out(x)
        return x

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)




@NECKS.register_module()
class camera_fuse(BaseModule):
    def __init__(self, in_channels, mid_channels, context_channels, 
                 cam_channels=27,grid_config=None,downsample=8,loss_depth_weight=1.0,loss_depth_type='bce',with_cp=False):
        super(camera_fuse, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.loss_depth_weight = loss_depth_weight
        self.loss_depth_type = loss_depth_type
        self.with_cp=with_cp
        self.constant_std = 0.5
        self.downsample = downsample
        self.grid_config = grid_config
        self.cam_depth_range = self.grid_config['dbound']
        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1)
        D, _, _ = ds.shape
        self.D = D
        self.stereo_volume_encoder = StereoVolumeEncoder(
            in_channels=D, out_channels=D
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.depth_net = DepthNet(in_channels, in_channels, self.D, cam_channels=cam_channels)        

        self.bn = nn.BatchNorm1d(cam_channels)

        self.depth_aggregation = DepthAggregation(embed_dims=32, out_channels=1)
        self.context_mlp = Mlp(cam_channels, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        
        self.depth_context_mlp=Mlp(D, mid_channels, context_channels)
        
        self.mergeconv1=nn.Conv2d(context_channels,context_channels,3,padding=1)
        self.mergeconv2=nn.Sequential(
            nn.Conv2d(context_channels,
                      context_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.Conv2d(context_channels,
                      context_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(context_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(context_channels,
                      context_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )
        
        # self.fusion_conv = nn.Conv2d(context_channels+D, context_channels, kernel_size=3, padding=1)  # 用于将拼接后的通道数变回 C
        # self.transformer_layer1 = nn.Transformer(d_model=context_channels, nhead=1)
        # self.transformer_layer2 = nn.Transformer(d_model=context_channels, nhead=1)
        # self.weight_left = nn.Parameter(torch.ones(1, 1, 48, 160))
    @force_fp32()
    def get_bce_depth_loss(self, depth_labels, depth_preds):
        _, depth_labels = self.get_downsampled_gt_depth(depth_labels)
        # depth_labels = self._prepare_depth_gt(depth_labels)
        B,N,D,H,W=depth_preds.shape
        depth_preds=depth_preds.view(B*N,D,H,W)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(depth_preds, depth_labels, reduction='none').sum() / max(1.0, fg_mask.sum())
        
        return depth_loss
    
    @force_fp32()
    def get_klv_depth_loss(self, depth_labels, depth_preds):
        depth_gaussian_labels, depth_values = generate_guassian_depth_target(depth_labels,
            self.downsample, self.cam_depth_range, constant_std=self.constant_std)
        
        depth_values = depth_values.view(-1)
        fg_mask = (depth_values >= self.cam_depth_range[0]) & (depth_values <= (self.cam_depth_range[1] - self.cam_depth_range[2]))        
        
        depth_gaussian_labels = depth_gaussian_labels.view(-1, self.D)[fg_mask]
        B,N,D,H,W=depth_preds.shape
        depth_preds=depth_preds.view(B*N,D,H,W)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)[fg_mask]
        
        depth_loss = F.kl_div(torch.log(depth_preds + 1e-4), depth_gaussian_labels, reduction='batchmean', log_target=False)
        
        return depth_loss
    
    
    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        if self.loss_depth_type == 'bce':
            depth_loss = self.get_bce_depth_loss(depth_labels, depth_preds)
        
        elif self.loss_depth_type == 'kld':
            depth_loss = self.get_klv_depth_loss(depth_labels, depth_preds)

        else:
            pdb.set_trace()
        
        return self.loss_depth_weight * depth_loss
        
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)
        
        # [min - step / 2, min + step / 2] creates min depth
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / self.grid_config['dbound'][2]
        gt_depths_vals = gt_depths.clone()
        
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]
        
        return gt_depths_vals, gt_depths.float()
        
    def get_depth_dist(self, x):
        return x.softmax(dim=1)
    
    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda=None):
        B, N, _, _ = rot.shape
        
        if bda is None:
            bda = torch.eye(3).to(rot).view(1, 3, 3).repeat(B, 1, 1)
        
        bda = bda.view(B, 1, *bda.shape[-2:]).repeat(1, N, 1, 1)
        
        if intrin.shape[-1] == 4:
            # for KITTI, the intrin matrix is 3x4
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                intrin[:, :, 0, 3],
                intrin[:, :, 1, 3],
                intrin[:, :, 2, 3],
                post_rot[:, :, 0, 0],
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],
                bda[:, :, 0, 0],
                bda[:, :, 0, 1],
                bda[:, :, 1, 0],
                bda[:, :, 1, 1],
                bda[:, :, 2, 2],
            ], dim=-1)
            
            if bda.shape[-1] == 4:
                mlp_input = torch.cat((mlp_input, bda[:, :, :3, -1]), dim=2)
        else:
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                post_rot[:, :, 0, 0],
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],
                bda[:, :, 0, 0],
                bda[:, :, 0, 1],
                bda[:, :, 1, 0],
                bda[:, :, 1, 1],
                bda[:, :, 2, 2],
            ], dim=-1)
        
        sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)], dim=-1).reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        
        return mlp_input
    
    

    
    
    
    def forward(self, x, mlp_input,depth_result):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        B,N,C,H,W = x.shape
        x = self.reduce_conv(x.view(-1, C, H, W))
        
        if self.with_cp:
            depth_mono=checkpoint(self.depth_net,x, mlp_input)
        else:
            depth_mono=self.depth_net(x, mlp_input)
        depth_mono=self.get_depth_dist(depth_mono)
        
        
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        _,C,H,W=context.shape
        
        _, stereo_volume = self.get_downsampled_gt_depth(depth_result)
        stereo_volume = stereo_volume.view(B*N, H, W, -1).permute(0, 3, 1, 2)
        stereo_volume = self.stereo_volume_encoder(stereo_volume)
        stereo_volume = self.get_depth_dist(stereo_volume)
        
        if self.with_cp:
            depth_volume=checkpoint(self.depth_aggregation,stereo_volume, depth_mono)
        else:
            depth_volume = self.depth_aggregation(stereo_volume, depth_mono)
        depth_volume = self.get_depth_dist(depth_volume)

        if self.with_cp:
            depth_context=checkpoint(self.depth_context_mlp,depth_volume.permute(0,2,3,1).flatten(0,2)).view(B,N,H,W,C).permute(0, 1, 4, 2, 3).contiguous()
        else:   
            depth_context=self.depth_context_mlp(depth_volume.permute(0,2,3,1).flatten(0,2)).view(B,N,H,W,C).permute(0, 1, 4, 2, 3).contiguous()

        context = context.view(B,N,C,H,W)
        context_add=depth_context[:,0]+depth_context[:,1]+context[:,0]+context[:,1]
        if self.with_cp:
            context_merge=checkpoint(self.mergeconv1,context_add)+checkpoint(self.mergeconv2,context_add)
        else:
            context_merge=self.mergeconv1(context_add)+self.mergeconv2(context_add)


        return context_merge.view(B,1, -1, H, W),depth_volume.view(B,N, -1, H, W)


        # return context.view(B,N,C,H,W),stereo_volume.view(B,N,-1,H,W)
    
