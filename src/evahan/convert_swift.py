"""
数据转换脚本，将默认的Evahan2026数据集转换为适合Swfit框架的模型微调数据集
"""

import json
from typing import Literal

from rich import print

from evahan import config
from evahan.dataset import (
    EvahanLayoutItem,
    EvahanOcrItem,
    load_evahan_layout_dataset,
    load_evahan_ocr_dataset,
)


CHAT_INSTANCE_TYPE = dict[str, list[dict[str, str]] | list[str]]


def __convert_ocr_item(
    item: EvahanOcrItem, use_abs_img_path: bool
) -> CHAT_INSTANCE_TYPE:
    """
    将单个EvahanOcrItem转换为Swift LLM推理所需的格式。
    Args:
        item (EvahanOcrItem): Evahan OCR数据项
        use_abs_img_path (bool): 是否使用绝对路径
    """
    img_path: str = item.relative_image_path
    if use_abs_img_path:
        img_path = item.image_path.as_posix()

    messages: list[dict[str, str]] = [
        {
            "role": "user",
            "content": f"<image>{config.OCR_USER_QUERY}",
        },
        {
            "role": "assistant",
            "content": item.text,
        },
    ]
    images: list[str] = [img_path]
    chat_instance: CHAT_INSTANCE_TYPE = {
        "messages": messages,
        "images": images,
    }
    return chat_instance


def __convert_layout_item(
    item: EvahanLayoutItem, use_abs_img_path: bool
) -> CHAT_INSTANCE_TYPE:
    """将单个EvahanLayoutItem转换为Swift LLM推理所需的格式。
    Args:
        item (EvahanLayoutItem): Evahan布局数据项
        use_abs_img_path (bool): 是否使用绝对路径
    """
    img_path = item.relative_image_path
    if use_abs_img_path:
        img_path = item.image_path.as_posix()

    response_text: str = "".join(
        [
            f"""<div class="{region.label}" data-bbox="{region.points}">{region.text}</div>\n"""
            for region in item.regions
        ]
    )

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
    images = [img_path]
    return {
        "messages": messages,
        "images": images,
    }


def __save(
    f, items: list[CHAT_INSTANCE_TYPE], format: Literal["json", "jsonl"]
) -> None:
    if format == "jsonl":
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        json.dump(items, f, ensure_ascii=False, indent=2)


def convert_to_swift(format: Literal["json", "jsonl"], use_abs_img_path: bool):
    """
    将Evahan2026的数据集转换为Swift所需的格式。Swift标准格式如下：

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

    Args:
        format (Literal["json", "jsonl"]): 保存格式
        use_abs_img_path (bool): 是否使用绝对路径
    """

    base_folder = config.EVAHAN_TRAINSET_A.parent  # 原始训练集所在的父目录
    print("Convert Dataset_A to Swift format...")
    items = load_evahan_ocr_dataset(base_folder / "Dataset_A.json")
    items = [__convert_ocr_item(item, use_abs_img_path) for item in items]
    with (base_folder / f"Swift_A.{format}").open("w", encoding="utf-8") as f:
        __save(f, items, format)

    print("Convert Dataset_B to Swift format...")
    items = load_evahan_layout_dataset(base_folder / "Dataset_B.json")
    items = [__convert_layout_item(item, use_abs_img_path) for item in items]
    with (base_folder / f"Swift_B.{format}").open("w", encoding="utf-8") as f:
        __save(f, items, format)

    print("Convert Dataset_C to Swift format...")
    items = load_evahan_ocr_dataset(base_folder / "Dataset_C.json")
    items = [__convert_ocr_item(item, use_abs_img_path) for item in items]
    with (base_folder / f"Swift_C.{format}").open("w", encoding="utf-8") as f:
        __save(f, items, format)

    print("All Done!")


if __name__ == "__main__":
    print("Converting Evahan2026 dataset to Swift format...")
    convert_to_swift(format="jsonl", use_abs_img_path=True)

    print("Converting Evahan2026 dataset to json array format...")
    convert_to_swift(format="json", use_abs_img_path=True)
