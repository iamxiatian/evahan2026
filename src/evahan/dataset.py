import json
from pathlib import Path

from rich import print

from evahan import config
from evahan.core import EvahanLayoutItem, EvahanOcrItem, EvahanRegion
from evahan.viz_layout import draw_layout


def annotate_dataset_b(name: str = "Dataset_B") -> None:
    """
    为数据集B生成可视化版面图像，方便查看标注是否正确
    Args:
        name: 数据集名称，默认 Dataset_B, 也可以是 Dataset_B_argument
    """

    base_folder = config.EVAHAN_TRAINSET_PATH
    json_file = base_folder / f"{name}.json"
    image_folder = base_folder / name
    save_folder = base_folder / f"{name}_annotated"

    if save_folder.exists():
        print(f"目录 {save_folder} 已存在，跳过可视化生成")
        return

    layout_items = load_evahan_layout_dataset(json_file)
    layout_dict: dict[str, list[EvahanRegion]] = {
        item.relative_image_path: item.regions for item in layout_items
    }

    draw_layout(
        image_folder=image_folder,
        save_folder=save_folder,
        layout_dict=layout_dict,
    )

    print(f"可视化版面图像已保存到 {save_folder}")


def load_evahan_ocr_dataset(dataset_path: Path) -> list[EvahanOcrItem]:
    """读取Evahan2026的OCR数据集的json文件，
    对应于Dataset_A和Dataset_C这两个数据集。"""
    items: list[EvahanOcrItem] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        items = [
            EvahanOcrItem(
                image_path=dataset_path.parent / item["image_path"],
                text=item["text"],
            )
            for item in json.load(f)
        ]
    return items


def load_evahan_layout_dataset(dataset_path: Path) -> list[EvahanLayoutItem]:
    """读取Evahan2026的版面数据集的json文件，对应于Dataset_B这个数据集。"""
    items: list[EvahanLayoutItem] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for item in json.load(f):
            regions = [
                EvahanRegion(
                    label=region["label"],
                    text=region["text"],
                    points=region["points"],
                )
                for region in item["regions"]
            ]

            items.append(
                EvahanLayoutItem(
                    image_path=dataset_path.parent / item["image_path"],
                    regions=regions,
                )
            )
    return items


def validate() -> None:
    """验证数据集B是否有效: points是否符合要求等。"""
    ds_b = config.EVAHAN_TRAINSET_B.parent / "Dataset_B.json"
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


if __name__ == "__main__":
    # ds_a = config.EVAHAN_TRAIN_PATH_A.parent / "Dataset_A.json"
    # ds_b = config.EVAHAN_TRAIN_PATH_B.parent / "Dataset_B.json"
    # ds_c = config.EVAHAN_TRAIN_PATH_C.parent / "Dataset_C.json"
    # ds_a_items = load_evahan_ocr_dataset(ds_a)
    # ds_b_items = load_evahan_layout_dataset(ds_b)
    # ds_c_items = load_evahan_ocr_dataset(ds_c)
    validate()
