import torch
import os
from pathlib import Path

class Config:
    """应用程序配置类"""
    
    # 基础路径配置
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # 数据集配置
    IMG_HEIGHT = 64           # 图片统一高度
    IMG_WIDTH = None          # 宽度自适应
    CHANNELS = 3              # 彩色图像
    NUM_CLASSES = 63          # 62个字符 + 1个blank（CTC用）
    MAX_LABEL_LENGTH = 6      # 验证码最大长度
    
    # 训练配置
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    
    # 模型配置
    HIDDEN_SIZE = 256
    NUM_LSTM_LAYERS = 2
    
    # 路径配置
    TRAIN_DATA_DIR = str(DATA_DIR / "train")
    TEST_DATA_DIR = str(DATA_DIR / "test")
    LABEL_FILE = str(DATA_DIR / "labels.txt")
    MODEL_SAVE_DIR = str(MODELS_DIR)
    LOG_DIR = str(LOGS_DIR)
    
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 字符集
    CHARSET = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
        'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
        'u', 'v', 'w', 'x', 'y', 'z'
    ]
    CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARSET)}
    IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARSET)}
    
    # 添加blank字符用于CTC
    BLANK_IDX = len(CHARSET)
    
    # API配置
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_RELOAD = True
    
    # 确保目录存在
    @classmethod
    def ensure_directories(cls):
        """确保必要的目录存在"""
        directories = [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            Path(cls.TRAIN_DATA_DIR),
            Path(cls.TEST_DATA_DIR)
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# 创建全局配置实例
config = Config()

# 确保目录存在
config.ensure_directories()