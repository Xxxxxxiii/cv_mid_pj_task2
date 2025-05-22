# _base_ = [
#     '../_base_/models/mask-rcnn_r50_fpn.py', 
#     '../_base_/datasets/voc_instance.py', 
#     '../_base_/schedules/schedule_1x.py',
#     '../_base_/default_runtime.py'
# ]

# _base_ = [
#     '../_base_/models/mask-rcnn_r50_fpn.py',
#     '../_base_/datasets/coco_instance.py',
#     '../_base_/default_runtime.py',
#     '../_base_/schedules/schedule_1x.py'
# ]



import os

_base_ = [
    '/home/manxiafeng/codes/mmdetection-main/configs/_base_/models/mask-rcnn_r50_fpn.py', 
    '/home/manxiafeng/codes/mmdetection-main/configs/_base_/datasets/coco_instance.py',
    '/home/manxiafeng/codes/mmdetection-main/configs/_base_/default_runtime.py',
    '/home/manxiafeng/codes/mmdetection-main/configs/_base_/schedules/schedule_1x.py'
]

dataset_type = 'CocoDataset'

classes = (
    'airplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'dining table', 'dog', 'horse', 'motorcycle', 'person',
    'potted plant', 'sheep', 'couch', 'train', 'tv'
)

data_root = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/'

# data = dict(
#     train=dict(
#         type=dataset_type,
#         ann_file=os.path.join(data_root, 'voc2012_train_instances.json'),
#         img_prefix=os.path.join(data_root, 'JPEGImages_train'),
#     ),
#     val=dict(
#         type=dataset_type,
#         ann_file=os.path.join(data_root, 'voc2012_val_instances.json'),
#         img_prefix=os.path.join(data_root, 'JPEGImages_val'),
#     ),
#     test=dict(
#         type=dataset_type,
#         ann_file=os.path.join(data_root, 'voc2012_val_instances.json'),
#         img_prefix=os.path.join(data_root, 'JPEGImages_val'),
#     )
# )

# load_from = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/work_dirs_MASK/mask_rcnn_voc2012_MASK-no_pretrain/39epochs.pth'
# load_from = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/work_dirs_MASK/mask_rcnn_voc2012_MASK-no_pretrain/39epochs.pth'
# load_from = 'mmdetection-main/data/VOCdevkit/VOC2012/work_dirs_MASK/mask_rcnn_voc2012_MASK-COCO_pretrain/40epochs.pth'
load_from = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/work_dirs_MASK/mask_rcnn_voc2012_MASK-COCO_pretrain_ver2/30epochs.pth'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
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
        data_root=data_root,
        ann_file=os.path.join(data_root, 'voc2012_train_instances_namesnew.json'),
        data_prefix=dict(img=os.path.join(data_root, 'JPEGImages_train/')),
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
        data_root=data_root,
        ann_file=os.path.join(data_root, 'voc2012_val_instances_namesnew.json'),
        data_prefix=dict(img=os.path.join(data_root, 'JPEGImages_val')),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root, 'voc2012_val_instances_namesnew.json'),
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args,
    classwise=True)
test_evaluator = val_evaluator



model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20),
        mask_head=dict(num_classes=20)
    )
)

# model = dict(
#     backbone=dict(
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint=load_from,
#             prefix='backbone'
#         )
#     ),
#     roi_head=dict(
#         bbox_head=dict(num_classes=20),
#         mask_head=dict(num_classes=20)
#     )
# )



# load_from = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/work_dirs_MASK/mask_rcnn_voc2012_MASK/epoch_.pth'
# 

# model = dict(
#     backbone=dict(
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint=load_from,
#             prefix='backbone'
#         )
#     ),
#     roi_head=dict(
#         bbox_head=dict(num_classes=20),
#         mask_head=dict(num_classes=20)
#     )
# )

# model = dict(
#     backbone=dict(
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint='/home/manxiafeng/codes/mmdetection-main/checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
#         )
#     ),
#     roi_head=dict(
#         bbox_head=dict(type='Shared2FCBBoxHead', num_classes=20),
#         mask_head=dict(type='FCNMaskHead', num_classes=20)
#     )
# )



# runner = dict(type='EpochBasedRunner', max_epochs=1)

# 可选: 使用 COCO-style 的 bbox 和 segm 评估
# evaluation = dict(interval=1, metric=['bbox', 'segm'])

# 输出目录
# work_dir = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/work_dirs_MASK/mask_rcnn_voc2012_MASK-no_pretrain'
work_dir = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/work_dirs_MASK/mask_rcnn_voc2012_MASK-COCO_pretrain_ver2'
# CUDA_LAUNCH_BLOCKING=1 python /home/manxiafeng/codes/mmdetection-main/tools/train.py /home/manxiafeng/codes/mmdetection-main/configs/mask_rcnn/mask_rcnn_voc_MASK.py
