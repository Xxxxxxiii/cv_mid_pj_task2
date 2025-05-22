import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# 映射颜色到物体类别的字典
def voc_colormap(N=256):
    def bitget(val, idx): return ((val & (1 << idx)) != 0)
 
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << 7 - j)
            g |= (bitget(c, 1) << 7 - j)
            b |= (bitget(c, 2) << 7 - j)
            c >>= 3
        # print([r, g, b])
        cmap[i, :] = [r, g, b]
    return cmap
 
color_map = voc_colormap()

def create_instance_masks(img_path):
    mask = np.array(Image.open(img_path))  # shape (H, W)
    instance_ids = np.unique(mask)
    print("非零实例 ID：", instance_ids[instance_ids != 0])  # 排除背景
    return mask

# 生成并查看掩码
mask = create_instance_masks('/home/manxiafeng/codes/mmdetection-main/ASSETS-TEST/2007_000033.png')
# print(mask)

# non_zero_values = mask[mask > 0]
# print("非零元素的值：")
# print(non_zero_values)

print(mask)
print("是否全为 0？", np.all(mask == 0))
plt.imshow(mask, cmap='nipy_spectral')  # 每个灰度值映射为不同颜色
plt.title("Instance Mask")
plt.colorbar()
plt.show()


def extract_instance_masks(mask):
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids != 0]  # 去掉背景 0
    masks = []

    for inst_id in instance_ids:
        binary_mask = (mask == inst_id).astype(np.uint8)
        masks.append(binary_mask)

    return masks

def save_mask_as_png(mask, save_path):
    mask_img = Image.fromarray(mask.astype(np.uint8))
    mask_img.save(save_path, format='PNG')


# 加载数据
segmentation_folder = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/SegmentationObject'  # 填写路径
save_path = '/home/manxiafeng/codes/mmdetection-main/data/VOCdevkit/VOC2012/SegmentationObject_masks'

# 遍历每张图像，生成掩码
for filename in os.listdir(segmentation_folder):
    if filename.endswith('.png'):
        img_path = os.path.join(segmentation_folder, filename)
        mask = create_instance_masks(img_path)

        mask_filename = os.path.splitext(filename)[0] + '_mask.png'
        mask_save_path = os.path.join(save_path, mask_filename)

        save_mask_as_png(mask, mask_save_path)


# python /home/manxiafeng/codes/mmdetection-main/mask_generation.py
