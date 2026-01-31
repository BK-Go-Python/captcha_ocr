"""
配置模块

包含应用程序的配置设置。
"""

from .settings import config, Config
from .env_config import get_config, env_config

__all__ = ['config', 'Config', 'get_config', 'env_config']