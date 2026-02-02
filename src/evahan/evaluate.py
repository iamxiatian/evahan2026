import json
from pathlib import Path

import structlog
from rich.progress import track

from evahan import config
from evahan.client import query
from evahan.dataset import (
    EvahanLayoutItem,
    EvahanOcrItem,
    load_evahan_ocr_dataset,
)


logger = structlog.get_logger()


def predict_ocr_trainset(ds_json_file: Path, out_jsonl_file: str):
    """
    对dataset_path目录下的图片文件，执行OCR识别，返回一个jsonline文件。
    jsonline文件中每一行是一个对象：
    {"image_path":"file path", "response": "llm output text"}

    Args:
        ds_json_file (str): EvaHan原始数据集对应的json文件
        out_jsonl_file (str): 保存到的jsonline文件名称
    """
    items: list[EvahanOcrItem] = load_evahan_ocr_dataset(ds_json_file)

    with open(out_jsonl_file, "w", encoding="utf-8") as f:
        for item in track(items, ds_json_file.name):
            image_file = ds_json_file.parent / item.image_path
            response = query(image_file.as_posix(), config.OCR_USER_QUERY)
            f.write(
                json.dumps(
                    {
                        "image_path": item.image_path,
                        "text": item.text,
                        "predicted": response,
                    },
                    ensure_ascii=False,
                )
            )
            f.write("\n")
            f.flush()
    logger.info(f"predict: {ds_json_file}, save to:{out_jsonl_file}")


def predict_layout_trainset(ds_json_file: Path, out_jsonl_file: str) -> None:
    """对dataset_path目录下的图片文件，执行版面元素识别，返回一个jsonline文件。
    jsonline文件中每一行是一个对象：
    {"image_path":"file path", "response": "llm output text"}

    Args:
        ds_json_file (str): EvaHan原始数据集对应的json文件
        out_jsonl_file (str): 保存到的jsonline文件名称
    """
    items: list[EvahanLayoutItem] = load_evahan_ocr_dataset(ds_json_file)
    with open(out_jsonl_file, "w", encoding="utf-8") as f:
        for item in track(items, ds_json_file.name):
            image_file = ds_json_file.parent / item.image_path
            response = query(image_file.as_posix(), config.LAYOUT_USER_QUERY)
            f.write(
                json.dumps(
                    {
                        "image_path": item.image_path,
                        "regions": [
                            {
                                "label": region.label,
                                "text": region.text,
                                "points": region.points,
                            }
                            for region in item.regions
                        ],
                        "predicted": response,
                    },
                    ensure_ascii=False,
                )
            )
            f.write("\n")
    logger.info(f"predict: {ds_json_file}, save to:{out_jsonl_file}")


def predict_trainset():
    ds_a = config.EVAHAN_TRAIN_PATH_A.parent / "Dataset_A.json"
    logger.info(f"predict trainset: {ds_a}")
    predict_ocr_trainset(ds_a, "./dataset/A_QwenVL7B_Instruct.jsonl")

    # ds_b = config.EVAHAN_TRAIN_PATH_B.parent / "Dataset_B.json"
    ds_c = config.EVAHAN_TRAIN_PATH_C.parent / "Dataset_C.json"
    logger.info(f"predict trainset: {ds_c}")
    predict_ocr_trainset(ds_c, "./dataset/C_QwenVL7B_Instruct.jsonl")
    logger.info("All done!")


if __name__ == "__main__":
    predict_trainset()
