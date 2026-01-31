import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from ..config.settings import config

def resize_with_aspect_ratio(image, target_height=64):
    """
    保持宽高比调整图片大小
    Args:
        image: numpy array (H, W, C) or (H, W)
        target_height: 目标高度
    Returns:
        resized_image: 调整后的图片
    """
    if len(image.shape) == 3:
        h, w, c = image.shape
    else:
        h, w = image.shape
        c = 1
    
    # 计算新宽度（保持比例）
    ratio = target_height / h
    new_w = int(w * ratio)
    
    # 调整大小
    if c == 1:
        resized = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
        resized = np.expand_dims(resized, axis=-1)  # 保持3维
        resized = np.repeat(resized, 3, axis=-1)    # 转换为3通道
    else:
        resized = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
    
    return resized

def load_image(image_path, target_height=64):
    """
    加载并预处理图片
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 转换为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 调整大小（保持宽高比）
    image = resize_with_aspect_ratio(image, target_height)
    
    # 归一化到 [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # 转换为PyTorch格式 (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    
    return image

def preprocess_image(image):
    """
    预处理单张图片
    """
    if isinstance(image, str):
        # 文件路径
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"无法读取图片: {image}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        # numpy数组
        if len(image.shape) == 2:
            # 灰度图转RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # RGBA转RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif isinstance(image, Image.Image):
        # PIL Image
        image = np.array(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError("不支持的图像格式")
    
    # 调整大小
    h, w = image.shape[:2]
    ratio = config.IMG_HEIGHT / h
    new_w = int(w * ratio)
    resized = cv2.resize(image, (new_w, config.IMG_HEIGHT))
    
    # 归一化
    normalized = resized.astype(np.float32) / 255.0
    
    # 转换为PyTorch格式
    tensor = torch.FloatTensor(normalized).permute(2, 0, 1)  # [C, H, W]
    tensor = tensor.unsqueeze(0)  # 添加batch维度
    
    return tensor