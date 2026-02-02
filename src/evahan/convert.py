import json

from evahan import config
from evahan.dataset import (
    EvahanLayoutItem,
    EvahanOcrItem,
    load_evahan_layout_dataset,
    load_evahan_ocr_dataset,
)


def __convert_ocr_item(item: EvahanOcrItem, use_abs_img_path: bool) -> dict:
    """
    将单个EvahanOcrItem转换为Swift LLM推理所需的格式。
    Args:
        item (EvahanOcrItem): Evahan OCR数据项
        use_abs_img_path (bool): 是否使用绝对路径
    """
    img_path = item.image_path
    if use_abs_img_path:
        img_path = f"{config.EVAHAN_DATA_PATH}/train_data/{item.image_path}"

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
    images = [img_path]
    return {
        "messages": messages,
        "images": images,
    }


def __convert_layout_item(
    item: EvahanLayoutItem, use_abs_img_path: bool
) -> dict:
    """将单个EvahanLayoutItem转换为Swift LLM推理所需的格式。
    Args:
        item (EvahanLayoutItem): Evahan布局数据项
        use_abs_img_path (bool): 是否使用绝对路径
    """
    img_path = item.image_path
    if use_abs_img_path:
        img_path = f"{config.EVAHAN_DATA_PATH}/train_data/{item.image_path}"

    response_text = ""
    for region in item.regions:
        response_text += f"""<div class="{region.label}" data-bbox="{region.points}">{region.text}</div>\n"""

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


def __save(f, items: list[dict], format: str) -> None:
    if format == "jsonl":
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        json.dump(items, f, ensure_ascii=False, indent=2)


def convert_to_swift(format: str, use_abs_img_path: bool):
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
        format (str): 文件存储格式，支持"jsonl"和"json"，如为json，则存储为数组，否则每一行为一个json对象
        use_abs_img_path (bool): 是否使用绝对路径
    """

    base_folder = config.EVAHAN_TRAIN_PATH_A.parent  # 原始训练集所在的父目录
    print("Convert Dataset_A to Swift format...")
    items = load_evahan_ocr_dataset(base_folder / "Dataset_A.json")
    items = [__convert_ocr_item(item, use_abs_img_path) for item in items]
    with (base_folder / "Swift_A.json").open("w", encoding="utf-8") as f:
        __save(f, items, format)

    print("Convert Dataset_B to Swift format...")
    items = load_evahan_layout_dataset(base_folder / "Dataset_B.json")
    items = [__convert_layout_item(item, use_abs_img_path) for item in items]
    with (base_folder / "Swift_B.json").open("w", encoding="utf-8") as f:
        __save(f, items, format)

    print("Convert Dataset_C to Swift format...")
    items = load_evahan_ocr_dataset(base_folder / "Dataset_C.json")
    items = [__convert_ocr_item(item, use_abs_img_path) for item in items]
    with (base_folder / "Swift_C.json").open("w", encoding="utf-8") as f:
        __save(f, items, format)

    print("All Done!")


if __name__ == "__main__":
    # ds_a = config.EVAHAN_TRAIN_PATH_A.parent / "Dataset_A.json"
    # ds_b = config.EVAHAN_TRAIN_PATH_B.parent / "Dataset_B.json"
    # ds_c = config.EVAHAN_TRAIN_PATH_C.parent / "Dataset_C.json"
    # ds_a_items = load_evahan_ocr_dataset(ds_a)
    # ds_b_items = load_evahan_layout_dataset(ds_b)
    # ds_c_items = load_evahan_ocr_dataset(ds_c)
    # validate()
    # breakpoint()
    convert_to_swift(format="jsonl", use_abs_img_path=True)
