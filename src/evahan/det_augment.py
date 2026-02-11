from ocr_det_augment.iaa_augment import IaaAugment
import json
import os
import cv2

src_dir = "dataset/EvaHan/train_data/Dataset_B"
src_label_file = "dataset/EvaHan/train_data/Dataset_B.json"

dst_dir = "dataset/EvaHan/train_data/Dataset_B_augmented"
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
dst_label_file = "dataset/EvaHan/train_data/Dataset_B_augmented.json"


augmenter_args = [
    {
        'type': 'Fliplr',
        'args': {
            'p': 0.5
        }
    },
    {
        'type': 'Affine',
        'args': {
            'rotate': [-10, 10]
        }
    },
]

flip_augmenter_args = [
    {
        'type': 'Fliplr',
        'args': {
            'p': 1.0
        }
    },
]
flip_iaa_augmenter = IaaAugment(flip_augmenter_args)

rotate_augmenter_args = [
    {
        'type': 'Affine',
        'args': {
            'rotate': [-20, 20]
        }
    },
]
rotate_iaa_augmenter = IaaAugment(rotate_augmenter_args)

with open(src_label_file, "r") as f:
    items = json.load(f)

dst_items = []
for item in items:
    image_path = item["image_path"]
    image_name = image_path.split("/")[-1]
    assert os.path.exists(os.path.join(src_dir, image_name)), f"image path {image_path} not found"
    image = cv2.imread(os.path.join(src_dir, image_name))
    ## save original image
    cv2.imwrite(os.path.join(dst_dir, f"original_{image_name}"), image)
    dst_items.append({
        "image_path": f"{dst_dir.split('/')[-1]}/original_{image_name}",
        "regions": item["regions"],
    })
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    polys  = []
    regions = item["regions"]
    cls_labels = []
    for region in regions:
        polys.append(region["points"])
        cls_labels.append(region["label"])
    data = {
        "image": image,
        "polys": polys,
        "cls_labels": cls_labels,
    }

    ## flip
    augmented_data = flip_iaa_augmenter(data)
    augmented_image = augmented_data["image"]
    augmented_polys = augmented_data["polys"]
    # for poly in augmented_polys:
    #     poly = poly.astype(int).reshape(-1, 2)
    #     cv2.polylines(augmented_image, [poly], True, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(dst_dir, f"flip_{image_name}"), augmented_image)
    flip_regions = []
    for i, poly in enumerate(augmented_polys):
        poly = poly.astype(int).reshape(-1, 2).tolist()
        flip_regions.append({
            "points": poly,
            "label": cls_labels[i],
            "text": ""
        })
    dst_items.append({
        "image_path": f"{dst_dir.split('/')[-1]}/flip_{image_name}",
        "regions": flip_regions,
    })
    
    ## rotate   
    augmented_data = rotate_iaa_augmenter(data)
    augmented_image = augmented_data["image"]
    augmented_polys = augmented_data["polys"]
    # for poly in augmented_polys:
    #     poly = poly.astype(int).reshape(-1, 2)
    #     cv2.polylines(augmented_image, [poly], True, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(dst_dir, f"rotate_{image_name}"), augmented_image)

    rotate_regions = []
    for i, poly in enumerate(augmented_polys):
        poly = poly.astype(int).reshape(-1, 2).tolist()
        rotate_regions.append({
            "points": poly,
            "label": cls_labels[i],
            "text": ""
        })
    dst_items.append({
        "image_path": f"{dst_dir.split('/')[-1]}/rotate_{image_name}",
        "regions": rotate_regions,
    })
print("augmented image saved to:", dst_dir)
with open(dst_label_file, "w") as f:
    json.dump(dst_items, f, ensure_ascii=False, indent=2)