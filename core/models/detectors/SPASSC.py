import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmcv.runner import force_fp32
import os
from PIL import Image
import numpy as np

@DETECTORS.register_module()
class SPASSC(BaseModule):
    def __init__(
        self,
        img_backbone,
        img_neck,
        
        img_view_transformer,
        camera_fuser=None,
        plugin_head=None,
        proposal_layer=None,
        VoxFormer_head=None,
        occ_encoder_backbone=None,
        occ_encoder_neck=None,
        pts_bbox_head=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None
    ):
        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)

        # if depth_net is not None:
        #     self.depth_net = builder.build_neck(depth_net)
        # else:
        #     self.depth_net = None
            
        if camera_fuser is not None:
            self.camera_fuser = builder.build_neck(camera_fuser)
        else:
            self.camera_fuser = None        
            
        if img_view_transformer is not None:
            self.img_view_transformer = builder.build_neck(img_view_transformer)
        else:
            self.img_view_transformer = None

        if plugin_head is not None:
            self.plugin_head = builder.build_head(plugin_head)
        else:
            self.plugin_head = None
            
        if proposal_layer is not None:
            self.proposal_layer = builder.build_head(proposal_layer)
        else:
            self.proposal_layer = None
        
        if VoxFormer_head is not None:
            self.VoxFormer_head = builder.build_head(VoxFormer_head)
        else:
            self.VoxFormer_head = None
        
        if occ_encoder_backbone is not None:
            self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        else:
            self.occ_encoder_backbone = None
        
        if occ_encoder_neck is not None:
            self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)
        else:
            self.occ_encoder_neck = None
        
        self.pts_bbox_head = builder.build_head(pts_bbox_head)
        
        self.init_cfg = init_cfg
        self.init_weights()
        
    
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape   
        imgs = imgs.view(B * N, C, imH, imW)
        
        x = self.img_backbone(imgs) 

        if self.img_neck is not None:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return x
    

    
    def extract_img_feat(self, img_inputs, img_metas,do_seg):
        img_enc_feats = self.image_encoder(img_inputs[0])
                   
        rots, trans, intrins, post_rots, post_trans, bda = img_inputs[1:7]# bda Bird's-eye-view Data Augmentation
        mlp_input = self.camera_fuser.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)

        #project left depth to right
        stereo_depth_right = self.project_depth_to_right(
            img_metas['stereo_depth'],
            img_inputs[1:7]
        )
        # self.visualize_depth(stereo_depth_right[0][0], title="Right Image Depth Map",filename=str(img_metas['sequence'])+str(img_metas['frame_id'])+'.png')

        stereo_depth=torch.cat((img_metas['stereo_depth'],stereo_depth_right),dim=1)

        context,depth= self.camera_fuser(img_enc_feats, mlp_input,stereo_depth)
        # context, depth = self.depth_net([img_enc_feats] + geo_inputs, img_metas, img)#此处+号是将两个list合并为一个list
        view_trans_inputs_left = [rots[:, 0:1, ...], trans[:, 0:1, ...], intrins[:, 0:1, ...], post_rots[:, 0:1, ...], post_trans[:, 0:1, ...], bda]
        # view_trans_inputs = [rots, trans, intrins, post_rots, post_trans, bda]

        # gaussian_depth=self.depth_net.get_gaussian_depth(depth)
        if self.img_view_transformer is not None:
            lss_volume = self.img_view_transformer(context, depth[:,0:1], view_trans_inputs_left)
        else:
            lss_volume = None
        #b, c, X, Y，Z
        if self.training==True:
            if do_seg:
                segmentation = self.plugin_head(context[:,0])
            else:
                segmentation = None
            
            if self.proposal_layer != None:
                query_proposal = self.proposal_layer(view_trans_inputs_left, img_metas['stereo_depth'])
                
                if query_proposal.shape[1] == 2:
                    proposal = torch.argmax(query_proposal, dim=1)
                else:
                    proposal = query_proposal # B 1 X Y Z
                proposal_list = []
                for i in range(proposal.shape[0]):
                    proposal_list.append(torch.nonzero(proposal[i].reshape(-1) > 0).view(-1))
                    
                l = len(proposal_list)
                max_n = max([idx.shape[0] for idx in proposal_list])

                # pad proposal_list 到 [B, max_n]，pad 值设为 -1
                proposal_padded = torch.full((l, max_n), -1, dtype=torch.long, device=proposal.device)
                for i, idx in enumerate(proposal_list):
                    proposal_padded[i, :idx.shape[0]] = idx
            else:
                proposal_padded = None
                proposal_list = None
        else:
            segmentation = None
            proposal_padded = None
            proposal_list = None
        return lss_volume,  depth, proposal_list, proposal_padded, segmentation
    
    def occ_encoder(self, x):
        if self.occ_encoder_backbone is not None:
            x = self.occ_encoder_backbone(x)

        if self.occ_encoder_neck is not None:
            x = self.occ_encoder_neck(x)
        
        return x

    def forward_train(self, data_dict):
        img_inputs = data_dict['img_inputs']#img_inputs[-1]是baseline，[-2]是焦距
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']
        gt_occ_2 = data_dict['gt_occ_2']
        do_seg=False
        if 'gt_semantics' in data_dict:
            target = data_dict['gt_semantics']
            do_seg=True
        else:
            target = None

        img_voxel_feats,  depth, proposal_list, proposal_padded, segmentation = self.extract_img_feat(img_inputs, img_metas, do_seg)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)
        # if len(voxel_feats_enc) > 1:
        #     voxel_feats_enc = [voxel_feats_enc[0]]
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]

        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            proposal_padded=proposal_padded,
            img_feats=img_voxel_feats
        )

        losses = dict()

        if depth is not None:
            losses['loss_depth'] = self.camera_fuser.get_depth_loss(img_metas['gt_depths'], depth)#只对左图像进行深度监督
            
            # losses['loss_depth'] += self.depth_net.get_l1_depth_loss_imgsize(img_metas['gt_depths'][:, 0:1, ...], depth_result)

        # losses_occupancy_geo = self.pts_bbox_head.geo_loss(
        #     output_voxels=output['output_geos'],
        #     target_voxels=gt_occ
        # )

        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
        )
        if output['supfeats_flatten'] != None:
            B, C, N_total = output['supfeats_flatten'].shape
            gt_occ_flat = gt_occ_2.flatten(1)  # [B, N_total]
            target_gathered = torch.full((B, proposal_padded.shape[-1]), 255, dtype=gt_occ_flat.dtype, device=gt_occ_flat.device)  # 255为ignore_index
            for i in range(B):
                valid = proposal_list[i].shape[0]
                if valid > 0:
                    target_gathered[i, :valid] = gt_occ_flat[i, proposal_list[i]]
            idx_expand = proposal_padded.unsqueeze(1).expand(-1, C, -1)
            loss_i = self.pts_bbox_head.loss(
                output_voxels=torch.gather(output['supfeats_flatten'], 2, idx_expand.clamp(min=0)),
                target_voxels=target_gathered
            )
            for key, value in loss_i.items():
                losses_occupancy[key] += value
                    
        
        if do_seg:
        
            losses_seg = self.plugin_head.loss(
                pred=segmentation,
                target=target[:, 0:1, ...],
                depth=img_metas['gt_depths'][:, 0:1, ...]
            )
                
            for key, value in losses_seg.items():
                if key in losses_occupancy:
                    losses_occupancy[key] += value
        
                
                
                        
        losses.update(losses_occupancy)

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        train_output = {
            'losses': losses,
            'pred': pred,
            'gt_occ': gt_occ
        }

        return train_output

    def forward_test(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        if 'gt_occ' in data_dict:
            gt_occ = data_dict['gt_occ']
        else:
            gt_occ = None

        img_voxel_feats, depth, proposal_list, proposal_padded,_ = self.extract_img_feat(img_inputs, img_metas,None)

        voxel_feats_enc = self.occ_encoder(img_voxel_feats)
        # if len(voxel_feats_enc) > 1:
        #     voxel_feats_enc = [voxel_feats_enc[0]]
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc
        )

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {
            'pred': pred,
            'gt_occ': gt_occ
        }

        return test_output
    
    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)
        
    def project_depth_to_right(self, stereo_depth, cam_params):
        b, C, h, w = stereo_depth.shape
        
        rots, trans, intrins, post_rots, post_trans, bda = cam_params

        xs = torch.linspace(0, h - 1, w, dtype=torch.float).view(1, 1, w).expand(1, h, w)
        ys = torch.linspace(0, h - 1, h, dtype=torch.float).view(1, h, 1).expand(1, h, w)

        grid = torch.stack((xs, ys), 1)
        image_grid=nn.Parameter(grid, requires_grad=False).to(stereo_depth.device)

        points = torch.cat([image_grid.repeat(b, 1, 1, 1), stereo_depth], dim=1) # [b, 3, h, w]
        points = points.view(b, 3, h * w).permute(0, 2, 1)

        # undo pos-transformation
        points = points - post_trans[:,0].view(b, 1, 3)
        points = torch.inverse(post_rots[:,0]).view(b, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam to ego
        points = torch.cat([points[:, :, 0:2, :] * points[:, :, 2:3, :], points[:, :, 2:3, :]], dim=2)
        
        if intrins.shape[3] == 4:
            shift = intrins[:, 0:1, :3, 3]
            points = points - shift.view(b, 1, 3, 1)
            intrins_l = intrins[:, 0:1, :3, :3]
        
        combine = rots[:,0].matmul(torch.inverse(intrins_l.squeeze(1)))
        points = torch.bmm(points.squeeze(-1), combine.transpose(1,2))  # [b, h*w, 3]s
        points += trans[:,0].view(b, 1, 3)
        # 使用左投影矩阵将像素坐标转换为相机坐标
        # cam_coords = torch.bmm(proj_matrix_left, cam_coords_hom)
        
        # 使用右投影矩阵将相机坐标转换为图像坐标
        # img_coords_right = torch.bmm(proj_matrix_right, cam_coords)
        # proj=torch.bmm(proj_matrix_left, proj_matrix_right)
        
        # img_coords_right = torch.bmm(proj, cam_coords_hom)
        # 将齐次坐标转换为非齐次坐标
        # img_coords_right = img_coords_right[:, :2, :] / img_coords_right[:, 2:3, :]
        # from lidar to camera
        points = points.view(-1, 1, 3) # N, 1, 3
        points = points - trans[:,1].view(1, -1, 3) # N, b, 3
        inv_rots = rots[:,0].inverse().unsqueeze(0) # 1, b, 3, 3
        points = (inv_rots @ points.unsqueeze(-1)) # N, b, 3, 1
        # the intrinsic matrix is [4, 4] for kitti and [3, 3] for nuscenes 
        if intrins.shape[-1] == 4:
            points = torch.cat((points, torch.ones((points.shape[0], points.shape[1], 1, 1)).to(points.device)), dim=2) # N, b, 4, 1
            points = (intrins[:,1].unsqueeze(0) @ points).squeeze(-1) # N, b, 4
        else:
            points = (intrins[:,1].unsqueeze(0) @ points).squeeze(-1)
        points_d = points[..., 2:3].view(-1,1) # N, b, 1
        points_uv = points[..., :2].view(-1,2) / points_d # N, b, 2
        # from raw pixel to transformed pixel

        points_uvd = torch.cat((points_uv, points_d), dim=1)
        
        valid_mask = (points_uvd[..., 0] >= 0) & \
                    (points_uvd[..., 1] >= 0) & \
                    (points_uvd[..., 0] <= w - 1) & \
                    (points_uvd[..., 1] <= h - 1) & \
                    (points_uvd[..., 2] > 0)
                    
        # 将图像坐标转换为深度图
        stereo_depth_right = torch.zeros_like(stereo_depth)
        
        valid_points = points_uvd[valid_mask]
        # sort
        depth_order = torch.argsort(valid_points[:, 2], descending=True)
        valid_points = valid_points[depth_order]
        # fill in
        stereo_depth_right[:,:,valid_points[:, 1].round().long(), 
                    valid_points[:, 0].round().long()] = valid_points[:, 2]
        
        return stereo_depth_right
    
    
    def visualize_depth(self,depth_map, title="Depth Map", filename="depth_map.png"):
        # 将深度图转换为CPU上的numpy数组
        depth_map_np = depth_map.squeeze().cpu().numpy()
        
        # 归一化深度图到0-255范围
        depth_map_np = (depth_map_np - np.min(depth_map_np)) / (np.max(depth_map_np) - np.min(depth_map_np)) * 255
        depth_map_np = depth_map_np.astype(np.uint8)
        
        # 将深度图转换为PIL图像
        depth_image = Image.fromarray(depth_map_np)
        
        # 保存图像
        depth_image.save(filename)