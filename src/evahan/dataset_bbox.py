import json
import jsonlines
from pathlib import Path
from typing import NamedTuple

from rich import print
import random
from PIL import Image
from evahan import config
from evahan.util import annotator

from qwen_vl_utils import smart_resize


class EvahanOcrItem(NamedTuple):
    image_path: str  # 图片路径
    text: str  # 图片中的文本内容


class EvahanRegion(NamedTuple):
    label: str  # 版面元素类别
    text: str  # 元素中的文本内容
    points: list[
        tuple[int, int]
    ]  # 元素的顶点坐标列表，顺序：左上、右上、右下、左下
    bbox: list[
        tuple[int, int, int, int]
    ]  # 元素的bbox坐标列表 格式：[x1, y1, x2, y2]

class EvahanLayoutItem(NamedTuple):
    image_path: str  # 图片路径
    regions: list[EvahanRegion]  # 版面元素列表


def load_evahan_ocr_dataset(dataset_path: Path) -> list[EvahanOcrItem]:
    """读取Evahan2026的OCR数据集，对应于Dataset_A和Dataset_C这两个数据集。"""
    items: list[EvahanOcrItem] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for item in json.load(f):
            items.append(
                EvahanOcrItem(
                    image_path=item["image_path"],
                    text=item["text"],
                )
            )
    return items


def load_evahan_layout_dataset(dataset_path: Path) -> list[EvahanLayoutItem]:
    """读取Evahan2026的版面数据集，对应于Dataset_B这个数据集。"""
    items: list[EvahanLayoutItem] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for item in json.load(f):
            regions = []
            for region in item["regions"]:
                regions.append(
                    EvahanRegion(
                        label=region["label"],
                        text=region["text"],
                        points=region["points"],
                        bbox=convert_points_to_bbox(region["points"]),
                    )
                )
            items.append(
                EvahanLayoutItem(
                    image_path=item["image_path"],
                    regions=regions,
                )
            )
    return items

def convert_points_to_bbox(points):
    # points format: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    # convert to bbox format: [x1, y1, x2, y2]
    x1 = min(points, key=lambda x: x[0])[0]
    y1 = min(points, key=lambda x: x[1])[1]
    x2 = max(points, key=lambda x: x[0])[0]
    y2 = max(points, key=lambda x: x[1])[1]
    return [x1, y1, x2, y2]


def validate() -> None:
    """验证数据集B是否有效: points是否符合要求等。"""
    ds_b = config.EVAHAN_TRAIN_PATH_B.parent / "Dataset_B.json"
    ds_b_items = load_evahan_layout_dataset(ds_b)
    element_count = 0  # 所有元素的数量
    irregular_count = 0  # 非正规矩形的元素数量
    for item in ds_b_items:
        regions = item.regions
        for region in regions:
            element_count += 1
            # points长度必须为4
            if len(region.points) != 4:
                print(f"{item.image_path} has invalid points in {region}")
                continue
            # 坐标必须是矩形，顺序为左上、右上、右下、左下
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = region.points
            if y1 != y2 or x1 != x4 or y3 != y4 or x2 != x3:
                irregular_count += 1
                print(f"{item.image_path} has invalid coordinates: {region}")
    print(f"Total elements: {element_count}")
    print(f"Irregular elements: {irregular_count}")


def draw_elements(item: EvahanLayoutItem, out_file: str) -> None:
    """在图片上绘制版面元素区域，用于可视化验证。
    Args:
        item (EvahanLayoutItem): 版面元素数据项
        out_file (str): 输出图片路径
    """
    image_file = config.EVAHAN_TRAIN_PATH_B.parent / item.image_path
    image = annotator.annotate(image_file, item.regions)
    image.save(out_file)

def draw_elements_bbox(item: EvahanLayoutItem, out_file: str) -> None:
    """在图片上绘制版面元素区域，用于可视化验证。
    Args:
        item (EvahanLayoutItem): 版面元素数据项
        out_file (str): 输出图片路径
    """
    image_file = config.EVAHAN_TRAIN_PATH_B.parent / item.image_path
    image = annotator.annotate_bbox(image_file, item.regions)
    image.save(out_file)


def __convert_ocr_item(item: EvahanOcrItem, root_dir: Path) -> dict:
    """将单个EvahanOcrItem转换为Swift LLM推理所需的格式。"""
    messages = [
        {
            "role": "user",
            "content": f"<image>{config.OCR_USER_QUERY}",
        },
        {
            "role": "assistant",
            "content": item.text,
        },
    ]
    images = [str(root_dir / item.image_path)]
    return {
        "messages": messages,
        "images": images,
    }


def __convert_layout_item(item: EvahanLayoutItem, root_dir: Path) -> dict:
    """将单个EvahanLayoutItem转换为Swift LLM推理所需的格式。"""
    ## rescale the bbox to the size of the image according to the scale factor 28
    image_path = item.image_path
    image = Image.open(config.EVAHAN_TRAIN_PATH_B.parent / image_path)
    width, height = image.size
    new_width, new_height = smart_resize(width, height, max_pixels=config.max_pixels, factor=28)
    scale_factor_w = new_width / width
    scale_factor_h = new_height / height
    response_text = ""
    for region in item.regions:
        bbox = region.bbox
        bbox = [bbox[0] * scale_factor_w, bbox[1] * scale_factor_h, bbox[2] * scale_factor_w, bbox[3] * scale_factor_h]
        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        response_text += f"""<div class="{region.label}" data-bbox="{bbox}">{region.text}</div>\n"""

    messages = [
        {
            "role": "user",
            "content": f"<image>{config.LAYOUT_USER_QUERY}",
        },
        {
            "role": "assistant",
            "content": response_text.strip(),
        },
    ]
    images = [str(root_dir / item.image_path)]
    return {
        "messages": messages,
        "images": images,
    }


def convert_to_swift():
    """
    将Evahan2026的数据集转换为Swift所需的格式。标准格式如下：
    
    ```json
    {
        "messages": [
            {"role": "user", "content": "<image>这是什么"},
            {"role": "assistant", "content": "这是一只小猫咪。"}
        ],
        "images": ["cat.png"],
        "rejected_messages": [
            {"role": "user", "content": "<image>这是什么"},
            {"role": "assistant", "content": "这是一只小猫咪。"}
        ],
        "rejected_images": ["dog.png"]
    }
    ```
    目前只有messages和images两个字段
    """

    base_folder = config.EVAHAN_TRAIN_PATH_A.parent # 原始训练集所在的父目录
    print("Convert Dataset_A to Swift format...")
    items = load_evahan_ocr_dataset(base_folder / "Dataset_A.json")
    items = [__convert_ocr_item(item, base_folder) for item in items]
    with (base_folder / "Swift_A.json").open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    with jsonlines.open(base_folder / "Swift_A.jsonl", mode="w") as writer:
        writer.write_all(items)

    print("Convert Dataset_B to Swift format...")
    items = load_evahan_layout_dataset(base_folder / "Dataset_B.json")
    items = [__convert_layout_item(item, base_folder) for item in items]
    with (base_folder / "Swift_B.json").open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    with jsonlines.open(base_folder / "Swift_B.jsonl", mode="w") as writer:
        writer.write_all(items)
        
    print("Convert Dataset_C to Swift format...")
    items = load_evahan_ocr_dataset(base_folder / "Dataset_C.json")
    items = [__convert_ocr_item(item, base_folder) for item in items]
    with (base_folder / "Swift_C.json").open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    with jsonlines.open(base_folder / "Swift_C.jsonl", mode="w") as writer:
        writer.write_all(items)

    print("All Done!")



# ds = MsDataset.load(
#     "AI-ModelScope/LaTeX_OCR", subset_name="small", split="train"
# )

# print(ds[0]["text"])
# ds[0]["image"].save("latex_ocr_sample_0.png")

if __name__ == "__main__":

    # ds_b = config.EVAHAN_TRAIN_PATH_B.parent / "Dataset_B.json"
    # ds_b_items = load_evahan_layout_dataset(ds_b)
    # # vis_path = "Dataset_B/b_4841.jpg"
    # item = random.choice(ds_b_items)
    # # print(item)
    # draw_elements(item, "dataset_b_sample_0.png")
    # image_path = config.EVAHAN_TRAIN_PATH_B.parent / item.image_path
    # image = Image.open(image_path)
    # width, height = image.size
    # new_width, new_height = smart_resize(width, height, max_pixels=config.max_pixels, factor=28)
    # resized_image = image.resize((new_width, new_height))
    # scale_factor_w = new_width / width
    # scale_factor_h = new_height / height
    # resized_image.save("dataset_b_sample_0_resized.png")
    # new_item = EvahanLayoutItem(image_path="dataset_b_sample_0_resized.png", regions=[])
    # for region in item.regions:
    #     bbox = region.bbox
    #     bbox = [bbox[0] * scale_factor_w, bbox[1] * scale_factor_h, bbox[2] * scale_factor_w, bbox[3] * scale_factor_h]
    #     region = EvahanRegion(label=region.label, text=region.text, points=region.points, bbox=bbox)
    #     new_item.regions.append(region)
    
    # out_file = "dataset_b_sample_0_bbox.png"
    # image_file =new_item.image_path
    # image = annotator.annotate_bbox(image_file, new_item.regions)
    # image.save(out_file)

    convert_to_swift()
