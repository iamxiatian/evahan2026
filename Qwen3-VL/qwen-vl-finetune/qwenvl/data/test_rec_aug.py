from ocr_rec_augment.rec_aug import PARSeqAug, SVTRAug
from PIL import Image
import cv2
import numpy as np
import ocr_rec_augment.auto_augment as auto_augment
from ocr_rec_augment.auto_augment import _LEVEL_DENOM, LEVEL_TO_ARG, NAME_TO_OP, _randomly_negate, rotate


image_path = "/mnt/public/lyl/evahan2026/dataset/EvaHan/train_data/Dataset_B/b_0001.jpg"

src_image = cv2.imread(image_path)
## rotate 90 degrees
src_image = cv2.rotate(src_image, cv2.ROTATE_90_CLOCKWISE)
data = {
    'image': src_image,
}
aug = SVTRAug(geometry_p=0.0, deterioration_p=0.0, colorjitter_p=1.0)
aug_data = aug(data)
aug_image = aug_data['image']
# rotate 90 degrees back
aug_image = cv2.rotate(aug_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imwrite('aug_image.jpg', aug_image)
# save src_image