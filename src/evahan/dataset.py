import json
from pathlib import Path
from typing import NamedTuple

from rich import print

from evahan import config
from evahan.util import annotator


class EvahanOcrItem(NamedTuple):
    image_path: str  # 图片路径
    text: str  # 图片中的文本内容


class EvahanRegion(NamedTuple):
    label: str  # 版面元素类别
    text: str  # 元素中的文本内容
    points: list[
        tuple[int, int]
    ]  # 元素的顶点坐标列表，顺序：左上、右上、右下、左下


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
                    )
                )
            items.append(
                EvahanLayoutItem(
                    image_path=item["image_path"],
                    regions=regions,
                )
            )
    return items


def validate() -> None:
    """验证数据集B是否有效: points是否符合要求等。"""
    ds_b = config.EVAHAN_TRAIN_PATH_B.parent / "Dataset_B.json"
    ds_b_items = load_evahan_layout_dataset(ds_b)
    for item in ds_b_items:
        regions = item.regions
        for region in regions:
            # points长度必须为4
            if len(region.points) != 4:
                print(f"{item.image_path} has invalid points in {region}")
                continue
            # 坐标必须是矩形，顺序为左上、右上、右下、左下
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = region.points
            if y1 != y2 or x1 != x4 or y3 != y4 or x2 != x3:
                print(f"{item.image_path} has invalid coordinates: {region}")


def draw_elements(item: EvahanLayoutItem, out_file: str) -> None:
    """在图片上绘制版面元素区域，用于可视化验证。
    Args:
        item (EvahanLayoutItem): 版面元素数据项
        out_file (str): 输出图片路径
    """
    image_file = config.EVAHAN_TRAIN_PATH_B.parent / item.image_path
    image = annotator.annotate(image_file, item.regions)
    image.save(out_file)


def convert():
    pass


# ds = MsDataset.load(
#     "AI-ModelScope/LaTeX_OCR", subset_name="small", split="train"
# )

# print(ds[0]["text"])
# ds[0]["image"].save("latex_ocr_sample_0.png")

if __name__ == "__main__":
    # ds_a = config.EVAHAN_TRAIN_PATH_A.parent / "Dataset_A.json"
    ds_b = config.EVAHAN_TRAIN_PATH_B.parent / "Dataset_B.json"
    # ds_c = config.EVAHAN_TRAIN_PATH_C.parent / "Dataset_C.json"
    # ds_a_items = load_evahan_ocr_dataset(ds_a)
    ds_b_items = load_evahan_layout_dataset(ds_b)
    # ds_c_items = load_evahan_ocr_dataset(ds_c)
    validate()
    breakpoint()
