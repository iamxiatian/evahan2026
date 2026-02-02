from rich import print


def check_env():
    import torch

    print("PyTorch版本：", torch.__version__)
    print("PyTorch绑定的CUDA版本：", torch.version.cuda)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"BF16支持: {torch.cuda.is_bf16_supported()}")


if __name__ == "__main__":
    check_env()
