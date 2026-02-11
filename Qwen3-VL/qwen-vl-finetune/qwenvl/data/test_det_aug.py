from ocr_rec_augment.rec_aug import PARSeqAug
from PIL import Image
import cv2
import numpy as np
import json


image_path = "/mnt/public/lyl/evahan2026/dataset/EvaHan/train_data/Dataset_B_augmented/original_b_0001.jpg"


src_image = cv2.imread(image_path)


data = {
    'image': src_image,
}
aug = PARSeqAug()
aug_data = aug(data)
aug_image = aug_data['image']
cv2.imwrite('aug_det_image.jpg', aug_image)


