
import os
import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine.visualization import Visualizer
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmdet.registry import VISUALIZERS
import torch
from mmdet.structures import DetDataSample
from mmcv.transforms import Compose
from mmengine.runner import load_checkpoint
from mmengine.dataset import default_collate
import inspect
from mmcv.visualization.image import imshow_bboxes
import numpy as np


img_dir = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/Gpt_generation_For_test/Gpt_generation_For_test' # 从哪里来

# # faster 目标检测
# config_file = '/home/manxiafeng/codes/mmdetection-main/configs/faster_rcnn/faster_rcnn_voc.py'
# checkpoint_file = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/work_dirs_MASK_mbjc/final_mbjc.pth'
# # img_dir = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/JPEGImages_val'
# save_dir = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/Gpt_generation_For_test-show-mbjc'

# 实例分割
config_file = '/home/manxiafeng/codes/mmdetection-main/configs/mask_rcnn/mask_rcnn_voc_MASK.py'
checkpoint_file = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/work_dirs_MASK/mask_rcnn_voc2012_MASK-COCO_pretrain_ver2/40epochs.pth'
save_dir = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/Gpt_generation_For_test-show-proposalboxes' # 保存
img_dir = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/TESTIMGS'

# # sparse
# config_file = '/home/manxiafeng/codes/mmdetection-main/configs/sparse_rcnn/sparse-rcnn-voc.py'
# # checkpoint_file = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/word_dirs_sparse/epoch_60.pth'
# # img_dir = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/JPEGImages_val'
# checkpoint_file = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/word_dirs_sparse_pre2/epoch_32.pth'
# save_dir = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/Gpt_generation_For_test-show-sparse2'

# 初始化模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

model.dataset_meta = dict(
     classes=[
         'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
     ]
 )

# 获取 visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta  # 设置类别信息

print(visualizer.dataset_meta)

# 创建输出文件夹
os.makedirs(save_dir, exist_ok=True)

# # 遍历可视化
# for img_file in os.listdir(img_dir):
#     if not img_file.endswith('.jpg'):
#         continue
#     img_path = os.path.join(img_dir, img_file)
#     result = inference_detector(model, img_path)

#     # 可视化并保存结果
#     visualizer.add_datasample(
#         name=img_file,
#         image=mmcv.imread(img_path, channel_order='rgb'),
#         data_sample=result,
#         draw_gt=False,
#         show=False,
#         out_file=os.path.join(save_dir, img_file),
#         # pred_score_thr=0.1
#     )

# mask情况生成候选框和最终框
test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

for img_file in os.listdir(img_dir):
    if not img_file.endswith('.jpg'):
        continue

    img_path = os.path.join(img_dir, img_file)

    # 预处理数据
    data = test_pipeline(dict(img_path=img_path))
    data = default_collate([data])
    data = model.data_preprocessor(data, False)
    batch_inputs = data['inputs'].to('cuda')
    batch_data_samples = data['data_samples']

    with torch.no_grad():
        features = model.backbone(batch_inputs)
        features = model.neck(features)

        rpn_results_list = model.rpn_head.predict(features, batch_data_samples, rescale=False)
        proposals = rpn_results_list[0].bboxes.detach().cpu()

        results = model.roi_head.predict(features, rpn_results_list, batch_data_samples, rescale=True)
        pred_sample = results[0]


    proposal_data = InstanceData(
        bboxes=proposals,
        scores=torch.ones((proposals.shape[0],), dtype=torch.float32),
        labels=torch.zeros((proposals.shape[0],), dtype=torch.long)
        )

    image = mmcv.imread(img_path, channel_order='rgb')

    bboxes_np = proposals.cpu().numpy()

    img_with_boxes = imshow_bboxes(
        img=image.copy(),
        bboxes=bboxes_np,
        top_k=-1,  # 显示所有框
        thickness=2,
        show=False
    )

    # 保存结果
    mmcv.imwrite(img_with_boxes, os.path.join(save_dir, img_file))


    
# python /home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/visualization.py
