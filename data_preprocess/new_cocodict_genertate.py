import os
import numpy as np
from PIL import Image
import json
from pycocotools import mask as maskUtils
import cv2
import xml.etree.ElementTree as ET

def parse_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = []

    filename = root.find('filename').text

    for obj in root.findall('object'):
        name = obj.find('name').text
        objects.append(name)

    return filename, objects

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append({
            'label': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    return objects

def get_bbox_from_mask(mask):
    # 获取 mask 中所有 instance_id
    instance_ids = np.unique(mask)

    if set(instance_ids.tolist()) == {1, 255}:
        instance_ids = np.array([1]) 
    else:
        instance_ids = instance_ids[(instance_ids != 0) & (instance_ids != 255)]
    
    id_to_bbox = {}
    for inst_id in instance_ids:
        binary_mask = np.uint8(mask == inst_id)

        # 找所有轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # 合并所有轮廓点来求整体 bbox
        all_points = np.concatenate(contours, axis=0)
        x, y, w, h = cv2.boundingRect(all_points)

        id_to_bbox[inst_id] = [x, y, x + w, y + h]
    return id_to_bbox

def iou(boxA, boxB):
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    boxA_area = (xa2 - xa1) * (ya2 - ya1)
    boxB_area = (xb2 - xb1) * (yb2 - yb1)

    return inter_area / float(boxA_area + boxB_area - inter_area + 1e-5)

def match_instances_to_labels(xml_dir, mask_dir):
    image_instance_to_label = {}

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue
        image_name = xml_file.replace('.xml', '.jpg')
        mask_name = xml_file.replace('.xml', '_mask.png')

        xml_path = os.path.join(xml_dir, xml_file)
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            continue

        objects = parse_voc_xml(xml_path)
        mask = np.array(Image.open(mask_path))
        instance_bboxes = get_bbox_from_mask(mask)

        matched = {}
        for inst_id, inst_box in instance_bboxes.items():
            best_iou = 0
            best_label = None
            for obj in objects:
                iou_score = iou(inst_box, obj['bbox'])
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_label = obj['label']
            if best_label is not None and best_iou > 0.3:
                matched[inst_id] = best_label

        image_instance_to_label[image_name] = matched

    return image_instance_to_label

# 类别列表（VOC 20 类）
VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
category_mapping = {name: i+1 for i, name in enumerate(VOC_CLASSES)}

# 写入 COCO 格式数据
def create_coco_output():
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": id_, "name": name} for name, id_ in category_mapping.items()]
    }

coco_output_train = create_coco_output()
coco_output_val = create_coco_output()

image_id = 0
annotation_id = 0

image_folder = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/JPEGImages'
mask_folder = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/SegmentationObject_masks'
xml_dir = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/Annotations'
mask_dir=mask_folder

instance_label_map = match_instances_to_labels(xml_dir, mask_dir)
print(list(instance_label_map.keys())[:10])
for i in range(10):
    print(instance_label_map[list(instance_label_map.keys())[i]])
print("no error here")

def load_split(split_file):
    with open(split_file, 'r') as f:
        return set(line.strip() + '.jpg' for line in f.readlines())

train_ids = load_split('/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt')
val_ids = load_split('/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt')


for filename in os.listdir(mask_folder):
    if not filename.endswith('_mask.png'):
        continue
    mask_path = os.path.join(mask_folder, filename)
    original_filename = filename.replace('_mask.png', '.jpg')
    img_path = os.path.join(image_folder, original_filename)

    if original_filename in train_ids:
        coco_output = coco_output_train
    elif original_filename in val_ids:
        coco_output = coco_output_val
    else:
        print(f"跳过 {original_filename}，不在 train/val 中")
        continue

    image_id2 = original_filename  # eg: '2007_000027.jpg'
    instance_label_dict = instance_label_map.get(image_id2, {}) 

    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path)
    width, height = img.size

    coco_output["images"].append({
        "id": image_id,
        "file_name": original_filename,
        "width": width,
        "height": height
    })

    # 加载 mask 图像
    mask = np.array(Image.open(mask_path))
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids != 0]  # 去除背景

    for instance_id in instance_ids:
        if instance_id == 255:
            continue

        binary_mask = np.uint8(mask == instance_id)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []

        for contour in contours:
            contour = contour.flatten().tolist()
            if len(contour) >= 6:  # 至少3个点
                segmentation.append(contour)

        if not segmentation:
            continue

        rle = maskUtils.encode(np.asfortranarray(binary_mask))
        area = float(maskUtils.area(rle))
        bbox = maskUtils.toBbox(rle).tolist()
        
        label_name = instance_label_dict.get(instance_id)
        if label_name is None:
            print(f"跳过 {image_id2} 中的 instance {instance_id}，找不到标签")
            continue

        category_id = category_mapping[label_name]

        coco_output["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "area": area,
            "bbox": bbox,
            "iscrowd": 0
        })

        annotation_id += 1

    image_id += 1

with open("/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/voc2012_train_instances.json", "w") as f:
    json.dump(coco_output_train, f)

with open("/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/voc2012_val_instances.json", "w") as f:
    json.dump(coco_output_val, f)

# python /home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/new_cocodict_genertate.py
