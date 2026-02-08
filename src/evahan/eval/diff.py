"""对比两次预测结果的差异"""

import json
from pathlib import Path
from typing import cast

from evahan import config


run_name = "v0_qwen25vl7b"


def load_ocr_items(
    run_name: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    加载OCR两个任务的预测结果
    Args:
        run_name: 运行名称，用于指定运行结果目录
    Returns:
        包含两次OCR预测结果的元组，每个元素是一个字典列表，字典包含图片文件名和OCR结果文本
    """
    run_path: Path = config.EVAHAN_RUNTEST_PATH / run_name

    with open(run_path / "Task_A.json") as f:
        data = json.load(f)
        # breakpoint()
        itmes_a = cast(list[dict[str, str]], data)

    with open(run_path / "Task_C.json") as f:
        data = json.load(f)
        itmes_c = cast(list[dict[str, str]], data)

    return itmes_a, itmes_c


def compare_ocr_items(
    run_name1: str, run_name2: str, out_json_file: Path
) -> None:
    """
    对比两次OCR预测结果的差异
    Args:
        run_name1: 第一次运行名称，用于指定第一次运行结果目录
        run_name2: 第二次运行名称，用于指定第二次运行结果目录
    """
    a1, c1 = load_ocr_items(run_name1)
    a2, c2 = load_ocr_items(run_name2)

    diff_items: list[dict[str, str]] = []

    # 对比Task_A的差异
    for item1, item2 in zip(a1, a2, strict=True):
        if item1["text"] != item2["text"]:
            diff_items.append(
                {
                    "image_path": item1["image_path"],
                    run_name1: item1["text"],
                    run_name2: item2["text"],
                }
            )

    # 对比Task_C的差异
    for item1, item2 in zip(c1, c2, strict=True):
        diff_items.append(
            {
                "image_path": item1["image_path"],
                run_name1: item1["text"],
                run_name2: item2["text"],
            }
        )

    # 打印差异项
    print(f"共发现 {len(diff_items)} 项差异")
    with open(out_json_file, "w") as f:
        json.dump(diff_items, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    compare_ocr_items(
        run_name1="v0_qwen25vl7bunsft",
        run_name2="va_lora_a800_16000",
        out_json_file=config.EVAHAN_RUNTEST_PATH / "diff_v0_va.json",
    )
