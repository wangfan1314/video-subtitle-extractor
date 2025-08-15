#!/bin/bash

# 视频字幕提取API服务启动脚本

echo "==================================="
echo "视频字幕提取API服务"
echo "==================================="

# 激活虚拟环境
echo "激活虚拟环境..."
source /Users/wangzhizhong/workspace/virturlEnv/videoEnv/bin/activate

# 检查虚拟环境是否激活成功
if [ -z "$VIRTUAL_ENV" ]; then
    echo "错误: 虚拟环境激活失败"
    exit 1
fi

echo "✓ 虚拟环境已激活: $VIRTUAL_ENV"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到python命令"
    exit 1
fi

# 进入项目目录
cd "$(dirname "$0")"

# 检查依赖
echo "检查依赖..."
python -c "import fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装依赖..."
    pip install -r requirements.txt
fi

# 启动服务
echo "启动API服务..."
echo "服务地址: http://0.0.0.0:8000"
echo "API文档: http://0.0.0.0:8000/docs"
echo "按 Ctrl+C 停止服务"
echo "==================================="
python start_service.py --host 0.0.0.0 --port 8000

echo "服务已停止"