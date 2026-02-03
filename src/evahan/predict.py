import json
from pathlib import Path
from typing import Literal

import structlog
from rich.progress import track

from evahan import config
from evahan.client.swift_client import Client
from evahan.dataset import (
    EvahanLayoutItem,
    EvahanOcrItem,
    load_evahan_ocr_dataset,
)
from evahan.extract import extract_layout_item


logger = structlog.get_logger()

# 初始化需要访问vllm服务的客户端
client = Client(host="127.0.0.1", port=8000)


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
            response = client.query(
                item.image_path.as_posix(), config.OCR_USER_QUERY
            )
            f.write(
                json.dumps(
                    {
                        "image_path": item.relative_image_path,
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
            response = client.query(
                item.image_path.as_posix(), config.LAYOUT_USER_QUERY
            )
            f.write(
                json.dumps(
                    {
                        "image_path": item.relative_image_path,
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
            f.flush()
    logger.info(f"predict: {ds_json_file}, save to:{out_jsonl_file}")


def to_evaluate_format(
    input_jsonl: str, output_json: str, task: Literal["ocr", "layout"]
):
    """将VLLM的预测保存结果，转换为评测所需的格式。之所以分为两步，是为了先记录VLLM的输出，方便后续分析。
    Args:
        input_jsonl (str): 输入的预测结果jsonline文件
        output_json (str): 输出的评测格式json文件
        task (Literal["ocr","layout"]): 任务类型，ocr或layout
    """
    with (
        open(input_jsonl, encoding="utf-8") as f_in,
        open(output_json, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            item = json.loads(line)
            if task == "ocr":
                output_item = {
                    "image_path": item["image_path"],
                    "text": item["predicted"],
                }
            elif task == "layout":
                llm_output = item["predicted"]
                iamge_path = config.EVAHAN_TRAIN_PATH / item["image_path"]
                extracted_item = extract_layout_item(iamge_path, llm_output)
                output_item = {
                    "image_path": item["image_path"],
                    "regions": [
                        region.to_dict() for region in extracted_item.regions
                    ],
                }
            else:
                raise ValueError(f"Unknown task: {task}")
            json.dump(output_item, f_out, ensure_ascii=False, indent=2)
    logger.info(f"Convert {input_jsonl} to evaluation format: {output_json}")


def predict_trainset(
    model: Literal["QwenVL7B_Instruct", "Xunzi", "LayoutLora"],
):
    """使用模型model，对数据集进行预测。结果保存形式为：`./dataset/A_{model}.jsonl`

    Args:
        model (str): 采用的模型名称
    """
    ds_a = config.EVAHAN_TRAIN_PATH_A.parent / "Dataset_A.json"
    logger.info(f"predict trainset: {ds_a}")
    predict_ocr_trainset(ds_a, f"./dataset/A_{model}.jsonl")

    ds_b = config.EVAHAN_TRAIN_PATH_B.parent / "Dataset_B.json"
    logger.info(f"predict trainset: {ds_b}")
    predict_layout_trainset(ds_b, f"./dataset/B_{model}.jsonl")

    ds_c = config.EVAHAN_TRAIN_PATH_C.parent / "Dataset_C.json"
    logger.info(f"predict trainset: {ds_c}")
    predict_ocr_trainset(ds_c, f"./dataset/C_{model}.jsonl")

    # 转换为评测格式
    logger.info("Convert to evaluation format...")
    to_evaluate_format(
        f"./dataset/A_{model}.jsonl",
        f"./dataset/A_{model}_eval.json",
        task="ocr",
    )
    to_evaluate_format(
        f"./dataset/B_{model}.jsonl",
        f"./dataset/B_{model}_eval.json",
        task="layout",
    )
    to_evaluate_format(
        f"./dataset/C_{model}.jsonl",
        f"./dataset/C_{model}_eval.json",
        task="ocr",
    )

    logger.info("All done!")


if __name__ == "__main__":
    import typer

    typer.run(predict_trainset)
