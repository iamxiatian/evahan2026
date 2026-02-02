import json 
from tqdm import tqdm
from pathlib import Path
from modelscope import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
)
from evahan import config

model_path = config.QWEN_VL_7B_INSTRUCT
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_path, torch_dtype="auto", device_map="auto"
# )

## tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

text = "汧第卅五"
# encode
tokens = tokenizer.encode(text)
print(tokens)
# decode
decoded_text = tokenizer.decode(tokens)
print(decoded_text)

    

# EVAHAN_TRAIN_PATH_A: Path = config.EVAHAN_TRAIN_PATH_A.parent / "Dataset_C.json"
# with EVAHAN_TRAIN_PATH_A.open("r", encoding="utf-8") as f:
#     items = json.load(f)
#     for item in tqdm(items):
#         text = item["text"]
#         # encode
#         tokens = tokenizer.encode(text)
#         # decode
#         decoded_text = tokenizer.decode(tokens)
#         if not decoded_text == text:
#             print(text)


# EVAHAN_TRAIN_PATH_B: Path = config.EVAHAN_TRAIN_PATH_B.parent / "Dataset_B.json"
# with EVAHAN_TRAIN_PATH_B.open("r", encoding="utf-8") as f:
#     items = json.load(f)
#     for item in tqdm(items):
#         regions = item["regions"]
#         for region in regions:
#             text = region["text"]
#             # encode
#             tokens = tokenizer.encode(text)
#             # decode
#             decoded_text = tokenizer.decode(tokens)
#             if not decoded_text == text:
#                 print(text)

