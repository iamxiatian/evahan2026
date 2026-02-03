import os
from pathlib import Path

import structlog
from dotenv import load_dotenv

from evahan import prompt


logger = structlog.get_logger()

load_dotenv()

QWEN_VL_7B_INSTRUCT = os.getenv("qwen_vl_7b_instruct")

# Evahan数据集的存放路径
EVAHAN_DATA_PATH: Path = Path(
    os.getenv("evahan_dataset_parent_path", default="./dataset")
)

# 训练集的父路径
EVAHAN_TRAIN_PATH: Path = EVAHAN_DATA_PATH / "train_data"

# 三个数据集的路径
EVAHAN_TRAIN_PATH_A: Path = EVAHAN_TRAIN_PATH / "Dataset_A"
EVAHAN_TRAIN_PATH_B: Path = EVAHAN_TRAIN_PATH / "Dataset_B"
EVAHAN_TRAIN_PATH_C: Path = EVAHAN_TRAIN_PATH / "Dataset_C"

if EVAHAN_DATA_PATH.exists():
    logger.info(f"EVAHAN_DATA_PATH: {EVAHAN_DATA_PATH}")
else:
    logger.error(f"EVAHAN_DATA_PATH: {EVAHAN_DATA_PATH} not exists!")

# OCR的默认提取提示语
OCR_USER_QUERY = prompt.OCR_USER_QUERY_V1

# 版面默认的提示
LAYOUT_USER_QUERY = prompt.LAYOUT_USER_QUERY_V1
