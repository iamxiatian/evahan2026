import json
from pathlib import Path
import random
import structlog
from rich.progress import track

from evahan import config
from evahan.dataset import EvahanRegion
from evahan.client import query, query_base64
from evahan.util.file import parser_html_to_json, pil_image_to_base64
from PIL import Image
from qwen_vl_utils import smart_resize
import base64
from evahan.dataset import (
    EvahanLayoutItem,
    EvahanOcrItem,
    load_evahan_ocr_dataset,
    load_evahan_layout_dataset
)
import re
from evahan.util import annotator


logger = structlog.get_logger()


def predict_ocr_trainset(ds_json_file: Path, out_json_file: str):
    """
    对dataset_path目录下的图片文件，执行OCR识别，返回一个jsonline文件。
    jsonline文件中每一行是一个对象：
    {"image_path":"file path", "response": "llm output text"}

    Args:
        ds_json_file (str): EvaHan原始数据集对应的json文件
        out_jsonl_file (str): 保存到的jsonline文件名称
    """
    items: list[EvahanOcrItem] = load_evahan_ocr_dataset(ds_json_file)

    
    with open(out_json_file, "w", encoding="utf-8") as f:
        results = []
        for item in track(items, ds_json_file.name):
            image_file = ds_json_file.parent / item.image_path
            image_pil = Image.open(image_file)
            width, height = image_pil.size
            new_width, new_height = smart_resize(width, height, max_pixels=config.max_pixels, factor=28)
            image_pil = image_pil.resize((new_width, new_height))
            image_base64 = pil_image_to_base64(image_pil)
            response = query_base64(image_base64, config.OCR_USER_QUERY)
            results.append({
                "image_path": item.image_path,
                "text": response,
            })
            print(f"predict: {item.image_path}, text: {response}")
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"predict: {ds_json_file}, save to:{out_json_file}")


def predict_layout_trainset(ds_json_file: Path, out_json_file: str) -> None:
    """对dataset_path目录下的图片文件，执行版面元素识别，返回一个jsonline文件。
    jsonline文件中每一行是一个对象：
    {"image_path":"file path", "response": "llm output text"}

    Args:
        ds_json_file (str): EvaHan原始数据集对应的json文件
        out_jsonl_file (str): 保存到的jsonline文件名称
    """
    items: list[EvahanLayoutItem] = load_evahan_layout_dataset(ds_json_file)
    with open(out_json_file, "w", encoding="utf-8") as f:
        results = []
        cnt = 0
        for item in track(items, ds_json_file.name):
            image_file = ds_json_file.parent / item.image_path
            image_pil = Image.open(image_file).convert("RGB")
            width, height = image_pil.size
            new_width, new_height = smart_resize(width, height, max_pixels=config.max_pixels, factor=28)
            image_pil = image_pil.resize((new_width, new_height))
            image_base64 = pil_image_to_base64(image_pil)
            response = query_base64(image_base64, config.LAYOUT_USER_QUERY)
            format_response = parser_html_to_json(response, new_width, new_height, width, height)
            regions = []
            for res in format_response:
                regions.append({
                    "label": res["label"],
                    "text": res["text"],
                    "points": res["points"],
                })
            results.append({
                "image_path": item.image_path,
                "regions": regions,
            })
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"predict: {ds_json_file}, save to:{out_json_file}")


def predict_OCR_trainset():
    ds_a = config.EVAHAN_TRAIN_PATH_A.parent / "Dataset_A.json"
    logger.info(f"predict trainset: {ds_a}")
    predict_ocr_trainset(ds_a, "./dataset/A_QwenVL7B_Instruct_trained_0206.json")

    # ds_b = config.EVAHAN_TRAIN_PATH_B.parent / "Dataset_B.json"
    ds_c = config.EVAHAN_TRAIN_PATH_C.parent / "Dataset_C.json"
    logger.info(f"predict trainset: {ds_c}")
    predict_ocr_trainset(ds_c, "./dataset/C_QwenVL7B_Instruct_trained_0206.json")
    logger.info("All done!")

def predict_LAYOUT_trainset():
    ds_b = config.EVAHAN_TRAIN_PATH_B.parent / "Dataset_B.json"
    logger.info(f"predict trainset: {ds_b}")
    predict_layout_trainset(ds_b, "./dataset/B_QwenVL7B_Instruct_trained_0206.json")

def predict_one():
    idx = random.randint(1, 5000)
    image_name = str(idx).zfill(4)
    image_path = config.EVAHAN_TRAIN_PATH_B / f"b_{image_name}.jpg"
    image_pil = Image.open(image_path).convert("RGB")

    width, height = image_pil.size
    new_width, new_height = smart_resize(width, height, max_pixels=config.max_pixels, factor=28)
    image_pil = image_pil.resize((new_width, new_height))
    image_base64 = pil_image_to_base64(image_pil)
    response = query_base64(image_base64, config.LAYOUT_USER_QUERY)
    format_response = parser_html_to_json(response, new_width, new_height, width, height)
    regions = []
    for res in format_response:
        regions.append(EvahanRegion(
            label=res["label"],
            text=res["text"],
            points=res["points"],
        ))
    image = annotator.annotate(image_path.as_posix(), regions)
    image.save("layout_annotated.png")

if __name__ == "__main__":
    predict_OCR_trainset()
    # predict_LAYOUT_trainset()

    # predict_one()