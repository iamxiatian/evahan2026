import json
from pathlib import Path
import random
import structlog
from rich.progress import track
import os
from evahan import config
from evahan.dataset import EvahanRegion
from evahan.client import query, query_base64
from evahan.util.file import parser_html_to_json, pil_image_to_base64, parser_html_to_json_v2
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


def predict_ocr_trainset(ds_dir: str, out_json_file: str):
    """
    对dataset_path目录下的图片文件，执行OCR识别，返回一个jsonline文件。
    jsonline文件中每一行是一个对象：
    {"image_path":"file path", "response": "llm output text"}

    Args:
        ds_json_file (str): EvaHan原始数据集对应的json文件
        out_jsonl_file (str): 保存到的jsonline文件名称
    """
    image_files = [os.path.join(ds_dir, f) for f in os.listdir(ds_dir)]
    results = []
    for image_path in track(image_files):
        image_file = image_path
        image_prefix = "/".join(image_path.split("/")[-2:])

        image_pil = Image.open(image_file)
        width, height = image_pil.size
        new_width, new_height = smart_resize(width, height, max_pixels=config.max_pixels, factor=28)
        image_pil = image_pil.resize((new_width, new_height))
        image_base64 = pil_image_to_base64(image_pil)
        response = query_base64(image_base64, config.OCR_USER_QUERY)
        results.append({
            "image_path": image_prefix,
            "text": response,
        })
        print(f"predict: {image_prefix}, text: {response}")
    with open(out_json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"predict: {ds_dir}, save to:{out_json_file}")


def predict_layout_trainset(ds_dir: str, out_json_file: str) -> None:
    """对dataset_path目录下的图片文件，执行版面元素识别，返回一个jsonline文件。
    jsonline文件中每一行是一个对象：
    {"image_path":"file path", "response": "llm output text"}

    Args:
        ds_json_file (str): EvaHan原始数据集对应的json文件
        out_jsonl_file (str): 保存到的jsonline文件名称
    """
    image_files = [os.path.join(ds_dir, f) for f in os.listdir(ds_dir)]
    results = []
    for image_path in track(image_files):
        image_file = image_path
        image_prefix = "/".join(image_path.split("/")[-2:])
        image_pil = Image.open(image_file).convert("RGB")
        width, height = image_pil.size
        new_width, new_height = smart_resize(width, height, max_pixels=config.max_pixels, factor=28)
        image_pil = image_pil.resize((new_width, new_height))
        image_base64 = pil_image_to_base64(image_pil)
        response = query_base64(image_base64, config.LAYOUT_USER_QUERY)
        format_response = parser_html_to_json_v2(response, new_width, new_height, width, height)
        regions = []
        for res in format_response:
                regions.append({
                    "label": res["label"],
                    "text": res["text"],
                    "points": res["points"],
                })
        results.append({
            "image_path": image_prefix,
            "regions": regions,
        })

    with open(out_json_file, "w", encoding="utf-8") as f:   
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"predict: {ds_dir}, save to:{out_json_file}")


def predict_OCR_trainset():
    ds_a = "dataset/TestData/Task_A"
    logger.info(f"predict trainset: {ds_a}")
    predict_ocr_trainset(ds_a, "dataset/TestData/Task_A_OCR.json")

    ds_c = "dataset/TestData/Task_C"
    logger.info(f"predict trainset: {ds_c}")
    predict_ocr_trainset(ds_c, "dataset/TestData/Task_C_OCR.json")
    logger.info("All done!")

def predict_LAYOUT_trainset():
    ds_b = "dataset/TestData/Task_B"
    logger.info(f"predict trainset: {ds_b}")
    predict_layout_trainset(ds_b, "dataset/TestData/Task_B_LAYOUT.json")

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
    format_response = parser_html_to_json_v2(response, new_width, new_height, width, height)
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
    # predict_OCR_trainset()
    predict_LAYOUT_trainset()

    # predict_one()