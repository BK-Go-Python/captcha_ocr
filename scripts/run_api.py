#!/usr/bin/env python
"""
API服务启动脚本
"""

import os
import sys
import argparse
import uvicorn

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from src.api.main import app
from src.config.settings import config

def main():
    parser = argparse.ArgumentParser(description='启动验证码识别API服务')
    parser.add_argument('--host', type=str, default=config.API_HOST, help='主机地址')
    parser.add_argument('--port', type=int, default=config.API_PORT, help='端口号')
    parser.add_argument('--reload', action='store_true', default=config.API_RELOAD, help='是否启用热重载')
    
    args = parser.parse_args()
    
    print(f"启动API服务...")
    print(f"访问地址: http://{args.host}:{args.port}")
    print(f"API文档: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()