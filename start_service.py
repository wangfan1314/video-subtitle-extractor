#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author  : API Service
@Time    : 2024
@FileName: start_service.py
@desc: 启动视频字幕提取API服务
"""
import os
import sys
import argparse
import uvicorn
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='启动视频字幕提取API服务')
    parser.add_argument('--host', default='0.0.0.0', help='服务主机地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='服务端口 (默认: 8000)')
    parser.add_argument('--workers', type=int, default=1, help='工作进程数 (默认: 1)')
    parser.add_argument('--reload', action='store_true', help='启用自动重载')
    parser.add_argument('--log-level', default='info', choices=['critical', 'error', 'warning', 'info', 'debug', 'trace'], help='日志级别')
    
    args = parser.parse_args()
    
    # 确保在正确的目录运行
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print(f"正在启动视频字幕提取API服务...")
    print(f"服务地址: http://{args.host}:{args.port}")
    print(f"API文档: http://{args.host}:{args.port}/docs")
    print(f"工作目录: {project_root}")
    print(f"工作进程数: {args.workers}")
    print(f"日志级别: {args.log_level}")
    
    # 检查虚拟环境
    virtual_env = os.environ.get('VIRTUAL_ENV')
    if virtual_env:
        print(f"✓ 虚拟环境: {virtual_env}")
    else:
        print("⚠ 警告: 未检测到虚拟环境")
        print("建议激活虚拟环境: source /Users/wangzhizhong/workspace/virturlEnv/videoEnv/bin/activate")
    
    # 检查依赖
    try:
        import fastapi
        import uvicorn
        print("✓ API依赖检查通过")
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        sys.exit(1)
    
    # 检查模型文件
    models_dir = project_root / 'backend' / 'models'
    if not models_dir.exists():
        print(f"⚠ 警告: 模型目录不存在 {models_dir}")
    else:
        print("✓ 模型目录检查通过")
    
    try:
        # 启动服务
        uvicorn.run(
            "api_service:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n服务已停止")
    except Exception as e:
        print(f"启动服务失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()