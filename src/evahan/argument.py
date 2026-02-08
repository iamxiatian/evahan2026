"""数据增强处理:

1. 将数据集C中的图片转换为白底黑字或灰度图
2. 将数据集A中的图片与转换后的数据集C中的图片垂直合并，形成数据集D

"""

import json
from pathlib import Path
from typing import Literal

import cv2
from rich.progress import track

from evahan import config
from evahan.core import EvahanLayoutItem, EvahanOcrItem, EvahanRegion
from evahan.dataset import (
    load_evahan_layout_dataset,
    load_evahan_ocr_dataset,
    annotate_dataset_b,
)
from evahan.util import file_util, image_util
from evahan.util.image_resize import ImageProcessor, ResizedImage


def __argument_ocr_dataset(
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

    __argument_ocr_dataset(image_files_a, image_files_c, ocr_dict, "D")
    __argument_ocr_dataset(image_files_a, image_files_c, ocr_dict, "E")


def _adjust_regions(
    regions: list[EvahanRegion], resized_info: ResizedImage
) -> list[EvahanRegion]:
    """
    调整区域坐标以适应缩放后的图片

    Args:
        regions: 原始区域列表
        resized_info: 缩放后的图片信息

    Returns:
        调整后的区域列表
    """
    return [
        EvahanRegion(
            label=r.label,
            text=r.text,
            points=[
                (
                    int(p[0] * resized_info.scale + resized_info.x1),
                    int(p[1] * resized_info.scale + resized_info.y1),
                )
                for p in r.points
            ],
        )
        for r in regions
    ]


def argument_dataset_b() -> None:
    """
    增强训练集B中的图片，生成新的图片和布局信息，训练时使用新数据，而不再使用原来的训练集B
    """

    # 数据集B的增强版本的名称，目录和元数据文件的名称都采用这个名字
    argument_name = "Dataset_B_argument"
    processor = ImageProcessor(bg_width=924, bg_height=1232)
    out_dir = config.EVAHAN_TRAINSET_PATH / argument_name
    out_dir.mkdir(parents=True, exist_ok=True)

    resized_items: list[EvahanLayoutItem] = []
    raw_items = load_evahan_layout_dataset(
        config.EVAHAN_TRAINSET_PATH / "Dataset_B.json"
    )
    for raw_item in track(raw_items, description="增强数据集B..."):
        image_path = raw_item.image_path
        # print(f"处理图片: {image_path}")

        resized_info = processor.process_image(
            image_path.as_posix(), random_offset=False
        )
        out_file = out_dir / f"{image_path.stem}_argument_1.jpg"
        cv2.imwrite(out_file.as_posix(), resized_info.whole_image)

        resized_items.append(
            EvahanLayoutItem(
                image_path=out_file,
                regions=_adjust_regions(raw_item.regions, resized_info),
            )
        )

        # 随机移动偏移位置，再生成一张图片
        resized_info = processor.process_image(
            image_path.as_posix(), random_offset=True
        )
        out_file = out_dir / f"{image_path.stem}_argument_2.jpg"
        if resized_info.x1 > 0 or resized_info.y1 > 0:
            cv2.imwrite(out_file.as_posix(), resized_info.whole_image)

            resized_items.append(
                EvahanLayoutItem(
                    image_path=out_file,
                    regions=_adjust_regions(raw_item.regions, resized_info),
                )
            )

    # 保存处理后的布局信息
    out_file = config.EVAHAN_TRAINSET_PATH / f"{argument_name}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(
            [item.to_dict() for item in resized_items],
            f,
            indent=2,
            ensure_ascii=False,
        )

    # 可视化布局图
    annotate_dataset_b(name=argument_name)


def argument_task_b() -> None:
    """
    增强测试集B中的图片，生成新的图片和布局信息，测试时使用新数据，而不再使用原来的测试集B
    """
    # 数据集B的增强版本的名称，目录和元数据文件的名称都采用这个名字
    scale_factors:list[dict[str,str|float]] = [] # 缩放因子列表，每个元素是一个字典，包含缩放因子的文件名称和对应的值
    argument_name = "Task_B_argument"
    processor = ImageProcessor(bg_width=924, bg_height=1232, random_bg=False)
    out_dir = config.EVAHAN_TESTSET_PATH / argument_name
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_images = file_util.list_image_files(config.EVAHAN_TESTSET_PATH / "Task_B")
    for raw_image in track(raw_images, description="增强测试集B..."):
        resized_info = processor.process_image(
            raw_image.as_posix(), random_offset=False
        )
        out_file = out_dir / raw_image.name
        scale_factors.append(
            {
                "image_path": raw_image.name,
                "scale_factor": resized_info.scale,
            }
        )
        cv2.imwrite(out_file.as_posix(), resized_info.whole_image)

    # 保存缩放因子列表
    out_file = config.EVAHAN_TESTSET_PATH / f"{argument_name}_scale.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(
            scale_factors,
            f,
            indent=2,
            ensure_ascii=False,
        )



if __name__ == "__main__":
    # argument_dataset_ac()
    # print("OCR数据集增强完毕")
    #argument_dataset_b()
    # print("数据集B增强完毕")
    argument_task_b()
    print("任务B增强完毕")
