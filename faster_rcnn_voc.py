_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/voc_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]


# VOC 类别
classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 数据集定义
dataset_type = 'VOCDataset'
data_root = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012/ImageSets/Main/train.txt',
        img_prefix=data_root + 'VOC2012/',
        classes=classes
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012/ImageSets/Main/val.txt',
        img_prefix=data_root + 'VOC2012/',
        classes=classes
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012/ImageSets/Main/val.txt',
        img_prefix=data_root + 'VOC2012/',
        classes=classes
    )
)

# 修改模型类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=20
        )
    )
)

# 加载预训练权重
load_from = '/home/manxiafeng/codes/mmdetection-main/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'


# VOC-style 评估方式
evaluation = dict(interval=1, metric='mAP')


runner = dict(type='EpochBasedRunner', max_epochs=1)
# 输出路径
work_dir = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/work_dirs_mbjc'


# python /home/manxiafeng/codes/mmdetection-main/tools/train.py /home/manxiafeng/codes/mmdetection-main/configs/faster_rcnn/faster_rcnn_voc.py
