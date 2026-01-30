# EVAHAN2026

## 开发环境设置

1. 安装uv

    ```shell
    curl -fsSL https://get.uv.dev | bash
    # 或者通过pip安装
    pip install uv
    ```

2. 可编辑安装本项目

    ```shell
    #创建虚拟环境
    uv venv
    uv pip install -e .
    ```

3. 通过uv运行脚本main.py示例

    ```shell
    uv run -m evahan.main
    ```

4. 代码格式化

    ```shell
    uv run ruff check --fix src
    ```

## TODO