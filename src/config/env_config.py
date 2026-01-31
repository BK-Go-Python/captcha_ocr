import os
from typing import Dict, Any, Optional
from .settings import Config

class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    API_RELOAD = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    API_RELOAD = False
    LOG_LEVEL = "INFO"

class TestingConfig(Config):
    """测试环境配置"""
    DEBUG = True
    API_RELOAD = False
    LOG_LEVEL = "DEBUG"
    BATCH_SIZE = 4  # 测试时使用较小的批量大小
    NUM_WORKERS = 0  # 测试时不使用多进程

# 配置映射
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(env: Optional[str] = None) -> Config:
    """
    根据环境获取配置
    
    Args:
        env: 环境名称，如果为None则从环境变量ENV获取
        
    Returns:
        配置实例
    """
    if env is None:
        env = os.getenv('ENV', 'default')
    
    config_class = config_map.get(env.lower(), config_map['default'])
    return config_class()

# 创建全局配置实例
env_config = get_config()