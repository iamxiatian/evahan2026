.PHONY: help install test lint fmt build clean
help:  ## 显示可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:   ## 安装虚拟环境+依赖
	uv sync

test:      ## 跑测试
	uv run pytest

lint:      ## 代码检查
	uv run ruff check --fix src/
	uv run pyright src/

fmt:       ## 格式化
	uv run ruff format src/

diff:      ## 对比OCR不同预测结果的差异
	uv run -m evahan.eval.diff

build:     ## 构建 wheel
	uv build

run:	   ## 启动HTTP Server
	uv run -m evahan.main

clean:     ## 删缓存+临时文件
	uv cache clean
	rm -rf dist/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

deploy:
	./build.sh
