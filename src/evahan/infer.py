import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MAX_PIXELS"] = "1003520"
os.environ["VIDEO_MAX_PIXELS"] = "50176"
os.environ["FPS_MAX_FRAMES"] = "12"

from swift.llm import PtEngine, RequestConfig, InferRequest
from evahan import config

model = "/data/app/workspace/models/Qwen2.5-VL-7B-Instruct"

def main():
    # 加载推理引擎
    engine = PtEngine(model, max_batch_size=2)
    request_config = RequestConfig(max_tokens=512, temperature=0)

    # 这里使用了1个infer_request来展示batch推理
    infer_requests = [
        InferRequest(
            messages=[
                {
                    "role": "user",
                    "content": f"<image>{config.OCR_USER_QUERY}",
                }
            ],
            images=[
                "/data/app/workspace/evahan2026/dataset/train_data/Dataset_A/a_0001.jpg",
            ],
        ),
    ]
    resp_list = engine.infer(infer_requests, request_config)
    # query0 = infer_requests[0].messages[0]["content"]
    print(f"response0: {resp_list[0].choices[0].message.content}")

if __name__ == "__main__":
    main()
