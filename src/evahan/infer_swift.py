"""
推理Qwen2.5-VL-7B-Instruct模型的示例代码，使用Swift LLM框架进行多模态推理。
"""

import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MAX_PIXELS"] = "1003520"
os.environ["VIDEO_MAX_PIXELS"] = "50176"
os.environ["FPS_MAX_FRAMES"] = "12"

from swift.llm import InferRequest, PtEngine, RequestConfig

from evahan import config


model_path = "/data/app/workspace/models/Qwen2.5-VL-7B-Instruct-AWQ"


def main(image_path):
    # 加载推理引擎
    engine = PtEngine(model_path, max_batch_size=2)
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
                image_path,
            ],
        ),
        InferRequest(
            messages=[
                {
                    "role": "system",
                    "content": config.LAYOUT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"<image>{config.LAYOUT_USER_PROMPT}",
                },
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
    test_image_path = (
        "/data/app/workspace/evahan2026/dataset/train_data/Dataset_A/a_0001.jpg"
    )
    main(test_image_path)
