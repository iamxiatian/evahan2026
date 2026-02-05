"""
对数据集数据进行预测，并保存预测结果。

保存文件时，会先保存到jsonline格式的文件中，每一行是一个json对象，保存了输入图像(image_path)以及大模型的输出结果(llm_response)，方便中断后可以在原来的基础上继续预测，也便于后续分析。

最后再将jsonline文件转换为评测所需的json格式文件。
"""

import json
from pathlib import Path
from typing import Literal

import structlog
from rich.progress import track

from evahan import config
from evahan.client.swift_client import Client
from evahan.extract import extract_layout_regions
from evahan.util import file_util
from evahan.viz_layout import draw_testset_results


logger = structlog.get_logger(__name__)


class Predictor:
    def __init__(self, client: Client):
        self.client = client

    @staticmethod
    def __read_predicted(out_file: Path) -> dict[str, str]:
        """读取已经预测过的图片，避免重复预测"""
        predicated_images: dict[str, str] = {}
        if out_file.exists():
            lines = out_file.read_text(encoding="utf-8").splitlines()
            lines = [line.strip() for line in lines if line.strip()]
            for line in lines:
                item = json.loads(line)
                image_path = item["image_path"]
                predicated_images[image_path] = line
        return predicated_images

    def __predict_folder(
        self,
        image_folder: Path,
        out_jsonl: Path,
        task_type: Literal["ocr", "layout"],
        resume: bool = True,
    ) -> None:
        """
        对image_folder目录下的图片文件，执行ocr或者layout识别，返回一个jsonline文件。
        {"image_path":"file path", "llm_response": "llm output text"}

        Args:
            image_folder (Path): 图片文件夹路径
            out_jsonl_file (Path): 保存到的jsonline文件名称
            task_type (Literal["ocr","layout"]): 任务类型，ocr或layout
            resume(bool): 是否从已有的out_jsonl_file中断点继续
        """
        # 读取已预测的图片，避免重复预测, 主键为image_path，值为整行json字符串
        predicated_images: dict[str, str] = {}
        if resume and out_jsonl.exists():
            predicated_images = Predictor.__read_predicted(out_jsonl)

        llm_query = (
            config.OCR_USER_QUERY
            if task_type == "ocr"
            else config.LAYOUT_USER_QUERY
        )
        image_files: list[Path] = file_util.list_image_files(image_folder)

        with out_jsonl.open("w", encoding="utf-8") as f:
            for image_path in track(
                image_files, f"Processing {image_folder.name}"
            ):
                # 先读取已预测的结果，避免重复预测
                relative_path = f"{image_path.parent.name}/{image_path.name}"
                if relative_path in predicated_images:
                    f.write(predicated_images[relative_path])
                    f.write("\n")
                    continue

                response = self.client.query(image_path.as_posix(), llm_query)
                f.write(
                    json.dumps(
                        {
                            "image_path": relative_path,
                            "llm_response": response,
                        },
                        ensure_ascii=False,
                    )
                )
                f.write("\n")
                f.flush()
        logger.info(f"predict: {image_folder}, save to:{out_jsonl}")

    @staticmethod
    def to_evahan_format(
        input_jsonl: Path, output_json: Path, task: Literal["ocr", "layout"]
    ):
        """将VLLM的预测保存结果，转换为评测所需的格式。之所以分为两步，是为了先记录VLLM的输出，方便后续分析。
        Args:
            input_jsonl (Path): 输入的预测结果jsonline文件
            output_json (Path): 输出的评测格式json文件
            task (Literal["ocr","layout"]): 任务类型，ocr或layout
        """
        with (
            input_jsonl.open("r", encoding="utf-8") as f_in,
            output_json.open("w", encoding="utf-8") as f_out,
        ):
            for line in f_in:
                item = json.loads(line)
                if task == "ocr":
                    output_item = {
                        "image_path": item["image_path"],
                        "text": item["llm_response"],
                    }
                elif task == "layout":
                    llm_output = item["llm_response"]
                    regions = extract_layout_regions(llm_output)
                    output_item = {
                        "image_path": item["image_path"],
                        "regions": [region.to_dict() for region in regions],
                    }
                else:
                    raise ValueError(f"Unknown task: {task}")
                json.dump(output_item, f_out, ensure_ascii=False, indent=2)
        logger.info(
            f"Convert {input_jsonl} to evaluation format: {output_json}"
        )

    def run(
        self,
        folders: list[Path],
        task_types: list[Literal["ocr", "layout"]],
        run_name: str,
        resume: bool = True,
    ) -> None:
        """执行预测，对folders中的每一个文件夹按照任务类型(task_types)，进行预测，并保存结果到run_name指定的文件夹中。

        Args:
            folders (list[Path]): 需要预测的文件夹列表
            task_types (list[Literal["ocr","layout"]]): 每个文件夹对应的任务类型列表
            run_name (str): 保存结果的文件夹名称
            resume (bool, optional): 是否从已有的预测结果中断点继续.
        """
        run_result_folder = config.EVAHAN_DATA_PATH / "run_results" / run_name
        run_result_folder.mkdir(parents=True, exist_ok=True)
        jsonl_files: list[Path] = []
        final_json_files: list[Path] = []
        for folder in folders:
            jsonl_files.append(run_result_folder / f"{folder.name}.jsonl")
            final_json_files.append(run_result_folder / f"{folder.name}.json")

        # 执行预测
        for folder, out_jsonl_file, task_type in zip(
            folders, jsonl_files, task_types
        ):
            logger.info(f"Evaluate {task_type} task: {folder}")
            self.__predict_folder(
                folder,
                out_jsonl_file,
                task_type,
                resume=resume,
            )

        # 转换为评测格式
        logger.info("Convert to evaluation format...")
        for in_jsonl, out_json, task_type in zip(
            jsonl_files, final_json_files, task_types
        ):
            Predictor.to_evahan_format(in_jsonl, out_json, task_type)

        logger.info("All done!")


def run_trainset(
    run_name: str, host: str = "127.0.0.1", port: int = 8000
) -> None:
    """对训练集进行预测，并保存结果到run_name指定的文件夹中。

    Args:
        run_name (str): 保存结果的文件夹名称
        host (str): vllm服务的主机地址
        port (int): vllm服务的端口号
    """
    # 初始化需要访问vllm服务的客户端
    client = Client(host=host, port=port)
    predictor = Predictor(client)

    folders = [
        config.EVAHAN_TRAINSET_A,
        config.EVAHAN_TRAINSET_B,
        config.EVAHAN_TRAINSET_C,
    ]
    task_types: list[Literal["ocr", "layout"]] = ["ocr", "layout", "ocr"]
    predictor.run(
        folders,
        task_types,
        f"train_{run_name}",
        resume=True,
    )


def run_testset(
    run_name: str, host: str = "127.0.0.1", port: int = 8000
) -> None:
    """对测试集进行预测，并保存结果到run_name指定的文件夹中。

    Args:
        run_name (str): 保存结果的文件夹名称
        host (str): vllm服务的主机地址
        port (int): vllm服务的端口号
    """
    # 初始化需要访问vllm服务的客户端
    logger.info(f"Connect to Swift server at {host}:{port}")
    client = Client(host=host, port=port)
    predictor = Predictor(client)

    test_folders = [
        config.EVAHAN_TESTSET_PATH / "Task_A",
        config.EVAHAN_TESTSET_PATH / "Task_B",
        config.EVAHAN_TESTSET_PATH / "Task_C",
    ]
    task_types: list[Literal["ocr", "layout"]] = ["ocr", "layout", "ocr"]
    predictor.run(
        test_folders,
        task_types,
        run_name,
        resume=True,
    )

    # 可视化版面的预测结果
    logger.info("visualize layout test result.")
    draw_testset_results(run_name)


if __name__ == "__main__":
    import typer

    typer.run(run_testset)
