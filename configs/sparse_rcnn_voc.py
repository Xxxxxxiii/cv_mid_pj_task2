_base_ = [
    
    '/home/manxiafeng/codes/mmdetection-main/configs/_base_/default_runtime.py',
    '/home/manxiafeng/codes/mmdetection-main/configs/_base_/schedules/schedule_1x.py'
]

# custom_imports = dict(imports=['mmdet.models.task_modules.assigners.cost'], allow_failed_imports=False)


dataset_type = 'CocoDataset'
data_root = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012'

classes = (
    'airplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'dining table', 'dog', 'horse', 'motorcycle', 'person',
    'potted plant', 'sheep', 'couch', 'train', 'tv'
)

# classes = (
#     'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
#     'bus', 'car', 'cat', 'chair', 'cow',
#     'diningtable', 'dog', 'horse', 'motorbike', 'person',
#     'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
# )

# load_from = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/word_dirs_sparse/w1.pth'

metainfo = dict(classes=classes)

num_classes = 20
num_stages = 6
num_proposals = 100

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),   # 去掉mask开关
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/voc2012_train_instances_namesnew.json',
        data_prefix=dict(img='/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/JPEGImages_train'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/voc2012_val_instances_namesnew.json',
        data_prefix=dict(img='/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/JPEGImages_val'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/voc2012_val_instances_namesnew.json',
    metric=['bbox'],   # 这里不要segm
    format_only=False,
    backend_args=backend_args,
    classwise=True)
test_evaluator = val_evaluator

model = dict(
    type='SparseRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
        # init_cfg=None),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    rpn_head=dict(
        type='EmbeddingRPNHead',
        num_proposals=num_proposals,
        proposal_feature_channel=256),
    roi_head=dict(
        type='SparseRoIHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='DIIHead',
                num_classes=num_classes,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                # loss_cls=dict(
                #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)
        ]),
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        # dict(type='ClsCrossEntropyCost', weight=1.0),
                        dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.0)
                    ]),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1) for _ in range(num_stages)
        ]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_proposals)))

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, type='AdamW', lr=0.0000025, weight_decay=0.0001),   # 这里我多给了个0
    clip_grad=dict(max_norm=1, norm_type=2))

val_dataloader['dataset']['metainfo'] = metainfo
test_dataloader['dataset']['metainfo'] = metainfo

# load_from = '/home/manxiafeng/codes/mmdetection-main/checkpoints/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco_20201218_154234-7bc5c054.pth'
load_from = 'mmdetection-main/data/VOCdevkit/VOC2012/word_dirs_sparse_pre2/32epochs.pth'
work_dir = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/word_dirs_sparse_pre2'

print(num_classes)
print(classes)

print(train_dataloader['dataset']['metainfo']['classes'])

# python /home/manxiafeng/codes/mmdetection-main/tools/train.py /home/manxiafeng/codes/mmdetection-main/configs/sparse_rcnn/sparse-rcnn-voc.py --resume
