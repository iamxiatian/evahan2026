"""数据增强处理:

1. 将数据集C中的图片转换为白底黑字或灰度图
2. 将数据集A中的图片与转换后的数据集C中的图片垂直合并，形成数据集D

"""

import json
from pathlib import Path
from typing import Literal

from rich.progress import track

from evahan import config
from evahan.core import EvahanOcrItem
from evahan.dataset import load_evahan_ocr_dataset
from evahan.util import file_util, image_util


def __argument_dataset(
    image_files_a: list[Path],
    image_files_c: list[Path],
    ocr_dict: dict[str, str],
    dataset_name: Literal["D", "E"],
):
    """
    合并数据集A和C，将A中的图片与C中的图片垂直合并
    Args:
        image_files_a: 数据集A中图片文件路径列表
        image_files_c: 数据集C中图片文件路径列表
        ocr_dict: 包含数据集A和C中图片OCR结果的字典，键是图片文件名称，值是OCR结果文本
        dataset_name: 合并后的数据集名称，可选值为"D"或"E", 如为"D"，则合并A和C，形成数据集D，如为"E"，则合并C和A，形成数据集E
    """

    # 合并后的数据集D，每个元素是一个EvahanOcrItem对象
    merged_items: list[EvahanOcrItem] = []

    dataset_path = (
        config.EVAHAN_TRAINSET_D
        if dataset_name == "D"
        else config.EVAHAN_TRAINSET_E
    )
    dataset_path.mkdir(parents=True, exist_ok=True)

    for f_a, f_b in track(
        zip(image_files_a, image_files_c, strict=True),
        total=len(image_files_a),
        description="合并数据集A和C...",
    ):
        seq_number = f_a.stem.split("_")[1]
        merged_file = dataset_path / f"{dataset_name.lower()}_{seq_number}.jpg"

        if dataset_name == "D":
            image_util.merge_file_vertically(f_a, f_b, merged_file, True)
            merged_text = f"{ocr_dict[f_a.name]}{ocr_dict[f_b.name]}"
        else:
            image_util.merge_file_vertically(f_b, f_a, merged_file, True)
            merged_text = f"{ocr_dict[f_b.name]}{ocr_dict[f_a.name]}"

        merged_items.append(
            EvahanOcrItem(
                image_path=merged_file,
                text=merged_text,
            )
        )

    # 保存合并后的数据集D到JSON文件，符合Evahan2026的OCR数据集格式
    with (config.EVAHAN_TRAINSET_PATH / f"Dataset_{dataset_name}.json").open(
        "w"
    ) as f:
        json.dump(
            [item.to_dict() for item in merged_items],
            f,
            ensure_ascii=False,
            indent=4,
        )


def argument_dataset_ac():
    """
    合并数据集A和C，将A中的图片与C中的图片垂直合并，形成数据集D和数据集E
    """
    image_files_a = file_util.list_image_files(config.EVAHAN_TRAINSET_A)
    image_files_c = file_util.list_image_files(config.EVAHAN_TRAINSET_C)

    # 从Dataset_A.json和Dataset_C.json中，分别读取出内容，保存到字典中
    items_a: list[EvahanOcrItem] = load_evahan_ocr_dataset(
        config.EVAHAN_TRAINSET_PATH / "Dataset_A.json"
    )

    items_c: list[EvahanOcrItem] = load_evahan_ocr_dataset(
        config.EVAHAN_TRAINSET_PATH / "Dataset_C.json"
    )

    # 字典的主键是图片的文件名称name，值是OCR的结果文本
    ocr_dict = {item.image_path.name: item.text for item in items_a + items_c}

    __argument_dataset(image_files_a, image_files_c, ocr_dict, "D")
    __argument_dataset(image_files_a, image_files_c, ocr_dict, "E")


if __name__ == "__main__":
    argument_dataset_ac()
    print("数据集增强完毕")
