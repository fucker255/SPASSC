data_root = './dataset/SSCBenchKITTI360'
ann_file = './dataset/SSCBenchKITTI360/labels'
stereo_depth_root = './dataset/SSCBenchKITTI360/depth'
camera_used = ['left','right'] 

dataset_type = 'KITTI360Dataset'
point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
occ_size = [256, 256, 32]
lss_downsample = [2, 2, 2]

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_size = [voxel_x, voxel_y, voxel_z]

grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
}

empty_idx = 0

kitti360_class_frequencies = [
        2264087502, 20098728, 104972, 96297, 1149426, 
        4051087, 125103, 105540713, 16292249, 45297267,
        14454132, 110397082, 6766219, 295883213, 50037503,
        1561069, 406330, 30516166, 1950115,
]

class_names = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence',
    'vegetation', 'terrain', 'pole', 'traffic-sign', 'other-structure', 'other-object'
]

num_class = len(class_names)

# dataset config #
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.2,
    flip_dy_ratio=0.2,
    flip_dz_ratio=0
)

data_config={
    'input_size': (384, 1408),
    # 'resize': (-0.06, 0.11),
    # 'rot': (-5.4, 5.4),
    # 'flip': True,
    'resize': (0., 0.),
    'rot': (0.0, 0.0 ),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', data_config=data_config, load_stereo_depth=True,
         is_train=True, color_jitter=(0.4, 0.4, 0.4)),
    dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='kitti360', load_seg=False),
    dict(type='LoadAnnotationOcc', bda_aug_conf=bda_aug_conf, apply_bda=False,
            is_train=True, point_cloud_range=point_cloud_range),
    dict(type='CollectData', keys=['img_inputs', 'gt_occ','gt_occ_2'], 
            meta_keys=['pc_range', 'occ_size', 'raw_img', 'stereo_depth', 'sequence','frame_id', 'img_shape', 'gt_depths']),
]

trainset_config=dict(
    type=dataset_type,
    stereo_depth_root=stereo_depth_root,
    data_root=data_root,
    ann_file=ann_file,
    pipeline=train_pipeline,
    split='train',
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    test_mode=False,
)

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', data_config=data_config, load_stereo_depth=True,
         is_train=False, color_jitter=None),
    dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='kitti360'),
    dict(type='LoadAnnotationOcc', bda_aug_conf=bda_aug_conf, apply_bda=False,
            is_train=False, point_cloud_range=point_cloud_range),
    dict(type='CollectData', keys=['img_inputs', 'gt_occ'],  
            meta_keys=['pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img', 'stereo_depth', 'img_shape', 'gt_depths'])
]

testset_config=dict(
    type=dataset_type,
    stereo_depth_root=stereo_depth_root,
    data_root=data_root,
    ann_file=ann_file,
    pipeline=test_pipeline,
    split='test',
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range
)

valset_config=dict(
    type=dataset_type,
    stereo_depth_root=stereo_depth_root,
    data_root=data_root,
    ann_file=ann_file,
    pipeline=test_pipeline,
    split='test',
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range
)

data = dict(
    train=trainset_config,
    val=valset_config,
    test=testset_config
)

batch_size = 2

train_dataloader_config = dict(
    batch_size=batch_size,
    num_workers=16)

test_dataloader_config = dict(
    batch_size=batch_size,
    num_workers=16)


# model params #
numC_Trans = 128
voxel_channels = [128, 256, 512]
voxel_out_indices = (0, 1, 2)
voxel_out_channels = [128]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)


_dim_ = 128


# _split_=4


model = dict(
    type='SPASSC',
    img_backbone=dict(
        type='CustomEfficientNet',
        arch='b7',
        drop_path_rate=0.2,
        frozen_stages=0,
        norm_eval=False,
        out_indices=(2, 3, 4, 5, 6),
        with_cp=True,
        init_cfg=dict(type='Pretrained', prefix='backbone', 
        checkpoint='./pretrain/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth'),
    ),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[48, 80, 224, 640, 2560],
        upsample_strides=[0.5, 1, 2, 4, 4], 
        out_channels=[128, 128, 128, 128, 128]),
    camera_fuser=dict(
        type='camera_fuse',
        in_channels=640,
        mid_channels=640,
        context_channels=numC_Trans,
        cam_channels=33,
        grid_config=grid_config,
        downsample=8,
        loss_depth_type='kld',
        loss_depth_weight=0.0001,
        with_cp=False
    ),
    img_view_transformer=dict(
        type='ViewTransformerLSS',
        downsample=8,
        grid_config=grid_config,
        data_config=data_config,
    ),
    # plugin_head=dict(
    #     type='plugin_segmentation_head',
    #     in_channels=numC_Trans,
    #     out_channel_list=[128, 64, 32],
    #     num_class=num_class,
    # ),
    proposal_layer=dict(
        type='VoxelProposalLayer',
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
        input_dimensions=[128, 128, 16],
        data_config=data_config,
        init_cfg=None
    ),
    occ_encoder_backbone=dict(
        type='Fuser',
        # embed_dims=128,
        global_aggregator=dict(
            type='SimpleAggregator',
            embed_dims=_dim_,
        ),
        local_aggregator=dict(
            type='LocalAggregator',
            local_encoder_backbone=dict(
                type='CustomResNet3D',
                numC_input=128,
                num_layer=[2, 2, 2],
                num_channels=[128, 128, 128],
                stride=[1, 2, 2]
            ),
            local_encoder_neck=dict(
                type='GeneralizedLSSFPN',
                in_channels=[128, 128, 128],
                out_channels=_dim_,
                start_level=0,
                num_outs=3,
                norm_cfg=norm_cfg,
                conv_cfg=dict(type='Conv3d'),
                act_cfg=dict(
                    type='ReLU',
                    inplace=True),
                upsample_cfg=dict(
                    mode='trilinear',
                    align_corners=False
                )
            )
        ),
        far_aggregator=dict(
            type='FarAggregator',
            far_encoder_backbone=dict(
                type='Scan1dFormer',
                embed_dims=_dim_,
                num_layers=4,
                with_cp=False,
            ),
        ),
    ),
    pts_bbox_head=dict(
        type='OccHeadWithSup',
        in_channels=[sum(voxel_out_channels)],
        out_channel=num_class,
        empty_idx=0,
        num_level=1,
        with_cp=False,
        occ_size=occ_size,
        loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0
        },
        conv_cfg=dict(type='Conv3d', bias=False),
        balance_cls_weight=False,
        cls_weights=[
            1.0, 2.0, 5.0, 5.0, 3.0, 5.0, 5.0, 1.0, 3.0, 2.0, 5.0, 1.0, 5.0, 2.0, 2.0, 3.0, 3.0, 5.0, 5.0
            ],
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        class_frequencies=kitti360_class_frequencies
    )
)

"""Training params."""
learning_rate=3e-4
training_steps=25000//batch_size #一个epoch5000步，多卡会除以卡数 即 epoch25*5000/4=31250

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=learning_rate,
        weight_decay=0.0670113
    ),
    paramwise_cfg = dict(
        custom_keys={
            # 'backbone': dict(lr_mult=0.1) 这种配置貌似不适合数据量小的情况
        }
    ),
    # clip_grad=dict(max_norm=35, norm_type=2)  # 可选，梯度裁剪
)
lr_schedulers = [dict(
    type="OneCycleLR",
    max_lr=learning_rate,
    total_steps=training_steps + 10,
    pct_start=0.148316,
    cycle_momentum=False,
    anneal_strategy="cos",
    interval="step",
    frequency=1
)]
# lr_scheduler = [
#     dict(
#         type="LinearLR",
#         start_factor=0.001,  # 初始学习率为 max_lr 的 0.1%
#         total_iters=1000,    # 线性增长的总步数
#         interval="step",     # 每个 step 调整一次
#         frequency=1          # 调度器的调用频率
#     ),
#     dict(
#         type="CosineAnnealingLR",
#         T_max=13000,
#         eta_min=1e-6,
#         interval="step",
#         frequency=1
#     ),
# ]
# lr_scheduler = dict(
#     type="OneCycleLR",
#     max_lr=learning_rate,
#     total_steps=training_steps + 10,
#     pct_start=0.05,
#     cycle_momentum=False,
#     anneal_strategy="cos",
#     interval="step",
#     frequency=1
# )
# lr_scheduler = dict(
#     type="CosineAnnealingLR",
#     T_max=13000,  # 一个周期的步数
#     eta_min=1e-6, # 最小学习率
#     interval="step",
#     frequency=1
# )
# lr_scheduler = dict(
#     type="ReduceLROnPlateau",
#     mode="min",          # 监控指标的最小值
#     factor=0.1,          # 学习率衰减因子
#     patience=5,          # 等待多少个 epoch 后再衰减
#     threshold=1e-4,      # 性能提升的阈值
#     threshold_mode="rel",
#     cooldown=0,
#     min_lr=1e-6,
#     interval="epoch",
#     frequency=1
# )



# load_from = 'pretrain/pretrain.ckpt'