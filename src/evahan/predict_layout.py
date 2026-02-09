"""
对数据集数据进行预测，并保存预测结果。

保存文件时，会先保存到jsonline格式的文件中，每一行是一个json对象，保存了输入图像(image_path)以及大模型的输出结果(llm_response)，方便中断后可以在原来的基础上继续预测，也便于后续分析。

最后再将jsonline文件转换为评测所需的json格式文件。
"""

import json
from pathlib import Path
from typing import cast

import structlog
from rich.progress import track

from evahan import config
from evahan.client.swift_client import Client
from evahan.core import REGION_DICT_TYPE
from evahan.extract import extract_layout_regions
from evahan.util import file_util
from evahan.viz_layout import draw_testset_results


logger = structlog.get_logger(__name__)
LAYOUT_ITEM_TYPE = dict[str, str | list[REGION_DICT_TYPE]]


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
        resume: bool = True,
    ) -> None:
        """
        对image_folder目录下的图片文件，执行ocr或者layout识别，返回一个jsonline文件。
        {"image_path":"file path", "llm_response": "llm output text"}

        Args:
            image_folder (Path): 图片文件夹路径
            out_jsonl_file (Path): 保存到的jsonline文件名称
            resume(bool): 是否从已有的out_jsonl_file中断点继续
        """
        # 读取已预测的图片，避免重复预测, 主键为image_path，值为整行json字符串
        predicated_images: dict[str, str] = {}
        if resume and out_jsonl.exists():
            predicated_images = Predictor.__read_predicted(out_jsonl)

        llm_query = config.LAYOUT_USER_QUERY
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
        input_jsonl: Path, output_json: Path
    ):
        """将VLLM的预测保存结果，转换为评测所需的格式。之所以分为两步，是为了先记录VLLM的输出，方便后续分析。
        Args:
            input_jsonl (Path): 输入的预测结果jsonline文件
            output_json (Path): 输出的评测格式json文件
        """
        with (
            input_jsonl.open("r", encoding="utf-8") as f_in,
            output_json.open("w", encoding="utf-8") as f_out,
        ):
            # 转换为Layout评测格式
            scale_factors: dict[str, float] = {} # 读取缩放因子
            with open(config.EVAHAN_TESTSET_PATH / "Task_B_argument_scale.json", "r", encoding="utf-8") as f:
                items = cast(list[dict[str, str|float]], json.load(f))
                for item in items:
                    image_path = str(item["image_path"])
                    scale_factors[image_path] = float(item["scale_factor"])

            layout_items: list[LAYOUT_ITEM_TYPE] = []
            for line in f_in:
                item = json.loads(line)
                llm_output = str(item["llm_response"])

                image_name = Path(item["image_path"]).name
                # 还原位置
                factor = scale_factors[image_name]
                regions: list[REGION_DICT_TYPE] = [
                    region.to_uncale_dict(factor)
                    for region in extract_layout_regions(llm_output)
                ]

                # 转换为最终的测试集路径
                layout_item: LAYOUT_ITEM_TYPE = {
                    "image_path":  f"Task_B/{image_name}",
                    "regions": regions,
                }
                layout_items.append(layout_item)
            json.dump(layout_items, f_out, ensure_ascii=False, indent=2)

        logger.info(
            f"Convert {input_jsonl} to evaluation format: {output_json}"
        )

    def run(
        self,
        folder: Path,
        run_name: str,
        resume: bool = True,
    ) -> None:
        """执行预测，对folders中的每一个文件夹，进行预测，并保存结果到run_name指定的文件夹中。

        Args:
            folders (list[Path]): 需要预测的文件夹列表
            task_types (list[Literal["ocr","layout"]]): 每个文件夹对应的任务类型列表
            run_name (str): 保存结果的文件夹名称
            resume (bool, optional): 是否从已有的预测结果中断点继续.
        """
        run_result_folder = config.EVAHAN_RUNTEST_PATH / run_name
        run_result_folder.mkdir(parents=True, exist_ok=True)
        jsonl_file: Path = run_result_folder / "Task_B.jsonl"
        json_file: Path = run_result_folder / "Task_B.json"
        
        logger.info(f"Evaluate layout task: {folder}")
        # self.__predict_folder(
        #     folder,
        #     jsonl_file,
        #     resume=resume,
        # )

        # 转换为评测格式
        logger.info("Convert to evaluation format...")
        Predictor.to_evahan_format(jsonl_file, json_file)



def run_layout_testset(
    run_name: str, host: str = "127.0.0.1", port: int = 8000
) -> None:
    """对版面测试集进行预测，并保存结果到run_name指定的文件夹中。

    Args:
        run_name (str): 保存结果的文件夹名称
        host (str): vllm服务的主机地址
        port (int): vllm服务的端口号
    """
    # 初始化需要访问vllm服务的客户端
    logger.info(f"Connect to Swift server at {host}:{port}")
    client = Client(host=host, port=port)
    predictor = Predictor(client)

    folder = config.EVAHAN_TESTSET_PATH / "Task_B_argument"
    predictor.run(
        folder,
        run_name,
        resume=True,
    )

    # 可视化版面的预测结果
    logger.info("visualize layout test result.")
    draw_testset_results(run_name)

    logger.info("All done!")



if __name__ == "__main__":
    import typer

    typer.run(run_layout_testset)
