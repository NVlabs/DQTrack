_base_ = [
    '../../../configs/_base_/datasets/nus-3d.py',
    '../../../configs/_base_/default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
bev_stride = 4
track_frame = 3
fp16_enabled = True

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
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
    use_external=False)

model = dict(
    type='PETRTrackerDQ',
    # not use grid mask
    use_grid_mask=False,
    tracker_cfg=dict(
        track_frame=track_frame,
        num_track=300,
        num_query=900,
        num_cams=6,
        embed_dims=256,
        ema_decay=0.5,
        train_det_only=False,
        train_track_only=True,
        class_dict=dict(
            all_classes=class_names,
            track_classes=track_names,
        ),
        pos_cfg=dict(
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
        det_thres=0.3,
        new_born_thres=0.2,
        asso_thres=0.1,
        miss_thres=7,
    ),
    img_backbone=dict(
        type='VoVNet',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage4', 'stage5')),
    img_neck=dict(
        type='CPFPN',  ###remove unused parameters 
        in_channels=[768, 1024],
        out_channels=256,
        num_outs=2),
    pts_bbox_head=dict(
        type='PETRv2TrackHead',
        num_classes=10,
        in_channels=256,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        with_fpe=True,
        with_time=True,
        with_multi=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=False,  ###use checkpoint to save memory
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        positional_encoding=dict(
            type='SinePositionalEncoding3D', 
            num_feats=128, 
            normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
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
        out_size_factor=bev_stride,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=point_cloud_range))))

dataset_type = 'NuScenesDatasetPETRTrack'
data_root = 'data/nuscenes/'
info_root = 'data/infos/'
file_client_args = dict(backend='disk')

ida_aug_conf = {
        "resize_lim": (0.47, 0.625),
        "final_dim": (320, 800),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
    }
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewImageFromMultiSweepsFiles', sweeps_num=1, to_float32=True, pad_empty_sweeps=True, test_mode=False, sweep_range=[3,27]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='TrackletRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
            rot_range=[-0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[0.95, 1.05],
            reverse_angle=True,
            training=True
            ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
]
train_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(type='CollectUnified3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 'img', 'lidar_timestamp'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewImageFromMultiSweepsFiles', sweeps_num=1, to_float32=True, pad_empty_sweeps=True, sweep_range=[3,27]),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
]

test_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(type='CollectUnified3D', keys=['img', 'lidar_timestamp'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=info_root + 'mmdet3d_nuscenes_30f_infos_train.pkl',
        num_frames_per_sample=track_frame,  # number of frames for each 
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
             ann_file=info_root + 'mmdet3d_nuscenes_30f_infos_val.pkl', 
             num_frames_per_sample=1), # please change to your own info file
    test=dict(type=dataset_type, pipeline=test_pipeline, pipeline_post=test_pipeline_post, 
              classes=class_names, track_classes=track_names, modality=input_modality,
              ann_file=info_root + 'mmdet3d_nuscenes_30f_infos_val.pkl',
              num_frames_per_sample=1))

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

find_unused_parameters=True #### when use checkpoint, find_unused_parameters must be False
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from='ckpts/pretrain/petrv2_vovnet_320_800_train.pth' # download from https://drive.google.com/file/d/1tv_D8Ahp9tz5n4pFp4a64k-IrUZPu5Im/view?usp=sharing
fp16 = dict(loss_scale=32.)
