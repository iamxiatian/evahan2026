"""如果目录下生成了满足条件的文件夹，就执行备份"""

import shutil
import time
from datetime import datetime
from pathlib import Path


processed: dict[str, str] = {}

targets = [
    "checkpoint-13300",
    "checkpoint-13500",
    "checkpoint-14000",
    "checkpoint-14500",
    "checkpoint-15000",
    "checkpoint-15500",
    "checkpoint-16000",
    "checkpoint-16500",
    "checkpoint-17000",
    "checkpoint-17500",
    "checkpoint-18000",
    "checkpoint-18500",
    "checkpoint-19000",
    "checkpoint-19500",
    "checkpoint-20000",
]


def watch_dir(dir_path: Path) -> None:
    """
    监控目录下是否有新生成的文件夹
    Args:
        dir_path: 要监控的目录路径
    """
    while True:
        for item in dir_path.glob("*"):
            if (
                item.is_dir()
                and item.name not in processed
                and item.name in targets
            ):
                print(f"发现需要处理的新文件夹：{item.name}")
                processed[item.name] = str(datetime.now())
                time.sleep(10)
                shutil.move(item, "output_ocr_v1/v0-backup/")
                print(
                    f"备份完成：{item.name} -> output_ocr_v1/v0-backup/{item.name}"
                )

            if item.name == "checkpoint-20000":
                print("checkpoint-20000 备份完成，退出监控")
                return

        time.sleep(30)


if __name__ == "__main__":
    watch_dir(Path("output_ocr_v1/v0-20260207-192519"))
