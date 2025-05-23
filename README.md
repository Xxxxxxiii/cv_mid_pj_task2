**期中pj代码仓库**

*1. 数据处理*

1.1. 通过 ```data_reprocess/images_division.py ```分割数据集。
   
1.2. 通过 ```data_reprocess/mask_generation.py ```获得掩码数据。生成的mask可在```data/VOCdevikit/VOC2012/mask_examples```中查看示例。

1.3. 通过 ```data_reprocess/new_cocodict_genertate.py ```获得json形式的VOC标注数据集。生成的标注可在```data/VOCdevikit/VOC2012/annotations```中查看。

*2. 训练*
   
2.1. 在```configs/_base_/schedules/schedule_1x.py```中更改学习率、优化器等训练参数，也可在```config```文件中重写优化器。
   
2.2. 在```checkpoints```中上传预训练参数。

2.3. 各模型训练

2.3.1. Faster R-CNN 运行```configs/faster_rcnn_voc.py```.

```python tools/train.py configs/faster_rcnn_voc.py```

2.3.2. Mask R-CNN 运行```configs/mask_rcnn_voc.py```.

2.3.3. Sparse R-CNN 运行```configs/sparse_rcnn_voc.py```

*3. 可视化*

3.1. 运行```visulization/visulization.py```。包含三种R-CNN可视化方法和Mask R-CNN第一阶段候选框可视化方法。通过修改```img_dir, config_file, checkpoint_file, save_dir```加载配置和权重，可视化和保存图片。

3.2. 可视化结果示例

3.2.1 Faster R-CNN：```visulization/visulization.py```
