import json
import os
import cv2
import numpy as np
import random

label_file = "dataset/EvaHan/train_data/Dataset_B_augmented.json"
data_dir = "dataset/EvaHan/train_data"
with open(label_file, "r") as f:
    items = json.load(f)

items = random.sample(items, 100)
for item in items:
    image_path = item["image_path"]
    image_path = os.path.join(data_dir, image_path)
    image = cv2.imread(image_path)
    regions = item["regions"]
    for region in regions:
        points = region["points"]
        points = np.array(points).reshape(-1, 2)
        cv2.polylines(image, [points], True, (0, 0, 255), 2)
    cv2.imwrite(f"local/draw_{image_path.split('/')[-1]}", image)