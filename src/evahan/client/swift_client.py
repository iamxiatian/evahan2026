"""
采用Swift的方式，调用本地部署的模型服务进行图文问答。

需要提前运行：./scripts/deploy.sh

"""

from pathlib import Path
from typing import cast

from rich import print
from swift.llm import InferClient, InferRequest, Messages, RequestConfig
from swift.llm.infer.protocol import ChatCompletionResponse

from evahan import config


class Client:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.engine = InferClient(host=host, port=port)
        # model_path = "/data/app/workspace/models/Qwen2.5-VL-7B-Instruct"
        # self.engine = PtEngine(model_path, max_batch_size=2)

    def query(
        self,
        image_url: str,
        query: str,
        system: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0,
    ) -> str:
        messages: Messages = [
            {
                "role": "user",
                "content": f"<image>{query}",
            }
        ]

        # 插入system消息
        if system:
            messages.insert(
                0,
                {"role": "system", "content": system},
            )

        request_config = RequestConfig(
            max_tokens=max_tokens, temperature=temperature
        )
        infer_requests = [
            InferRequest(
                messages=messages,
                images=[
                    image_url,
                ],
            )
        ]
        resp_list = cast(
            list[ChatCompletionResponse],
            self.engine.infer(infer_requests, request_config),
        )
        response: str = cast(str, resp_list[0].choices[0].message.content)
        return response


if __name__ == "__main__":
    client = Client(host="127.0.0.1", port=8000)
    # image_path = config.EVAHAN_TRAIN_PATH_A / "a_0001.jpg"
    # response = client.query(image_path.as_posix(), config.OCR_USER_QUERY)
    # print("OCR:\n", response)

    image_path: Path = config.EVAHAN_TRAINSET_B / "b_0001.jpg"
    response = client.query(image_path.as_posix(), config.LAYOUT_USER_QUERY)
    print("Layout Response:\n", response)
    from evahan.extract import extract_layout_regions

    regions = extract_layout_regions(response)

    print("Layout Regions:\n", regions)
    from evahan.util.annotate import visualize_layout

    visualize_layout(image_path, regions, save_path=Path("./layout_viz.png"))
