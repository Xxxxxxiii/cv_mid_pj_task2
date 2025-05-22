

import os
import shutil

voc_root = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012'
jpeg_folder = os.path.join(voc_root, 'JPEGImages')
train_txt = os.path.join(voc_root, 'ImageSets/Segmentation/train.txt')
val_txt = os.path.join(voc_root, 'ImageSets/Segmentation/val.txt')

train_output = os.path.join(voc_root, 'JPEGImages_train')
val_output = os.path.join(voc_root, 'JPEGImages_val')

os.makedirs(train_output, exist_ok=True)
os.makedirs(val_output, exist_ok=True)

# 拷贝函数
def copy_images(txt_file, output_dir):
    with open(txt_file, 'r') as f:
        for line in f:
            img_name = line.strip() + '.jpg'
            src_path = os.path.join(jpeg_folder, img_name)
            dst_path = os.path.join(output_dir, img_name)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                print(f'找不到图片: {src_path}')

# 拷贝训练集和验证集图片
copy_images(train_txt, train_output)
copy_images(val_txt, val_output)

print('图像拷贝完成！')


# # python /home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/images_division.py
