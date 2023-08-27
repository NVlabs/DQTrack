_base_ = [
    '../../../configs/_base_/datasets/nus-3d.py',
    '../../../configs/_base_/default_runtime.py'
]

plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
depth_bound = [2.0, 58.0, 0.5]
cam_sweep_num = 6
cam_sweep_list = [1]
fp16_enabled = True
bev_stride = 4
track_frame = 3
voxel_shape = [int(((point_cloud_range[3]-point_cloud_range[0])/voxel_size[0])//bev_stride),
               int(((point_cloud_range[4]-point_cloud_range[1])/voxel_size[1])//bev_stride),
               1]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

track_names = [
    'car', 'truck', 'bus', 'trailer',
    'motorcycle', 'bicycle', 'pedestrian',
]

velocity_error = {
  'car':6,
  'truck':6,
  'bus':6,
  'trailer':5,
  'pedestrian':5,
  'motorcycle':12,
  'bicycle':5,  
}
velocity_error = [velocity_error[_name] for _name in track_names]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
    cam_sweep_num=cam_sweep_num,
    cam_sweep_list=cam_sweep_list)

cam_sweep_num = min(cam_sweep_num, len(cam_sweep_list)+1)

model = dict(
    type='StereoTrackerDQ',
    # not use grid mask
    use_grid_mask=False,
    tracker_cfg=dict(
        track_frame=track_frame,
        num_track=300,
        num_query=300,
        num_cams=6,
        num_sweeps=cam_sweep_num,
        embed_dims=256,
        ema_decay=0.5,
        train_det_only=False,
        train_track_only=True,
        class_dict=dict(
            all_classes=class_names,
            track_classes=track_names,
        ),
        pos_cfg=dict(
            update_pos=True,
            update_all=True,
            detach_pos=True,
            detach_velo=True,
            pos_trans='ffn',
            fuse_type="sum",
            final_trans="linear",
        ),
        query_trans=dict(
            with_att=True,
            with_pos=True,
            min_channels=256,
            drop_rate=0.0,
        ),
        track_aug=dict(
            drop_prob=0,
            fp_ratio=0.2,
        ),
        # used for training
        ema_drop=0.0,
        # used for inference
        class_spec = True,
        eval_det_only=False,
        velo_error=velocity_error, 
        assign_method='hungarian',
        det_thres=0.2,
        new_born_thres=0.2,
        asso_thres=0.1,
        miss_thres=7,
    ),
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)
        ),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
        ),
    depth_head=dict(
        type='StereoDepth',
        model=dict(
            in_channels=512,
            mid_channels=512,
            out_channels=128,
            depth_bound=depth_bound,
            num_ranges=4,
            camera_aware=False,
        )),
    pts_bbox_head=dict(
        type='StereoTrackHead',
        trans_type='Stereo',
        multi_frame=dict(
            type='ema_update',
            ema_decay=0.7,
        ),
        view_cfg=dict(
            num_cams=6,
            num_sweeps=cam_sweep_num,
            em_iter=2,
            use_wnet=False, # whether use weight net
            use_temporal=True,
            depth_bound=depth_bound,
            range_list=[[2, 8], [8, 16], [16, 28], [28, 58]],
            downsample_factor=16,
            pc_range=point_cloud_range,
            voxel_size=[voxel_size[0]*bev_stride, 
                        voxel_size[1]*bev_stride, 
                        voxel_size[2]],
            voxel_shape=voxel_shape,
            fp16_enabled=fp16_enabled,
        ),
        # transformer_cfg
        num_classes=10,
        in_channels=128,
        unified_conv=dict(num_conv=3),
        with_box_refine=True,
        with_size_refine=False,
        transformer=dict(
            type='Uni3DTrackDETR',
            fp16_enabled=fp16_enabled,
            decoder=dict(
                type='UniTrackTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='UniCrossAtten',
                            num_points=1,
                            embed_dims=256,
                            num_sweeps=cam_sweep_num,
                            fp16_enabled=fp16_enabled)
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=512,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    norm_cfg=dict(type='LN'),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))
            )
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5)),
    bbox_coder=dict(
        type='DETRTrack3DCoder',
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=point_cloud_range,
        max_num=100), 
    loss_cfg=dict(
        type='DualQueryMatcher',
        num_classes=10,
        class_dict=dict(
            all_classes=class_names,
            track_classes=track_names),
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost.
            pc_range=point_cloud_range),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_asso=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0),
        loss_me=dict(
            type="CategoricalCrossEntropyLoss",
            loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=bev_stride)))

dataset_type = 'NuScenesTrackDataset'
data_root = 'data/nuscenes/'
info_root = 'data/infos/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewMultiSweepImageFromFiles', sweep_num=cam_sweep_num, to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ImageRandomResizeCropFlipRot', training=True, 
                                              resize_scales=(0.88, 1.0), 
                                              crop_sizes=(512, 1408),
                                              flip_ratio=0.5,
                                              rot_degree=(-5.4, 5.4)),
    dict(type='UnifiedRotScale', rot_range=[-0.3925, 0.3925],
                                 scale_ratio_range=[0.95, 1.05]),
    dict(type='UnifiedRandomFlip3D', sync_2d=False,
         flip_ratio_bev_horizontal=0.5,
         flip_ratio_bev_vertical=0.5),
    dict(type='TrackletRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
]
train_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(type='CollectUnified3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 'img', 'timestamp'])
]
test_pipeline = [
    dict(type='LoadMultiViewMultiSweepImageFromFiles', sweep_num=cam_sweep_num, to_float32=True),
    dict(type='ImageRandomResizeCropFlipRot', training=False,
                                              resize_scales=(0.88, 0.88), 
                                              crop_sizes=(512, 1408)),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
]

test_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(type='CollectUnified3D', keys=['img', 'timestamp'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=info_root + 'track_cat_10_infos_train.pkl', # please change to your own info file
        num_frames_per_sample=track_frame,  # number of frames for each clip in training.
        pipeline=train_pipeline,
        pipeline_post=train_pipeline_post,
        classes=class_names,
        track_classes=track_names,
        modality=input_modality,
        test_mode=False,
        force_continuous=True, # force to use continuous frame
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, pipeline=test_pipeline, pipeline_post=test_pipeline_post, 
             classes=class_names, track_classes=track_names, modality=input_modality, 
             ann_file=info_root + "track_cat_10_infos_val.pkl",
             num_frames_per_sample=1), # please change to your own info file
    test=dict(type=dataset_type, pipeline=test_pipeline, pipeline_post=test_pipeline_post, 
             classes=class_names, track_classes=track_names, modality=input_modality, 
             ann_file=info_root + "track_cat_10_infos_val.pkl",
             num_frames_per_sample=1)) # please change to your own info file

optimizer = dict(
    type='AdamW', 
    lr=1e-5,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=105, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 5
evaluation = dict(interval=24, pipeline=test_pipeline)
checkpoint_config = dict(max_keep_ckpts=5)

find_unused_parameters = True
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from='ckpts/pretrain/bevstero_r101_fb3_size_512_1408.pth' # please download the pretrained model from the our git
# fp16 setting
fp16 = dict(loss_scale=32.)