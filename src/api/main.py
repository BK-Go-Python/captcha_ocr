from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io
import torch
import os
from typing import Dict, Any

from ..models import create_model
from ..utils.image_utils import preprocess_image
from ..utils.text_utils import ctc_greedy_decode
from ..config.settings import config

# 导入响应模型
from .schemas import PredictionResponse, HealthResponse, InfoResponse

# 创建FastAPI应用
app = FastAPI(
    title="验证码识别API",
    description="基于CRNN+CTC的验证码识别服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储模型
model = None
device = config.DEVICE

@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    global model
    try:
        model_path = str(config.MODELS_DIR / "best_model.pth")
        model = create_model('crnn')
        
        # 加载模型权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型加载成功: {model_path}")
        else:
            print(f"警告: 模型文件不存在: {model_path}")
        
        model.eval()
        model = model.to(device)
        print(f"模型加载成功，使用设备: {device}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise e

@app.get("/", response_model=InfoResponse)
async def root():
    """根路径，返回API信息"""
    return {
        "message": "验证码识别API",
        "version": "1.0.0",
        "device": str(device)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    预测验证码图片
    
    Args:
        file: 上传的验证码图片文件
        
    Returns:
        PredictionResponse: 预测结果
    """
    # 检查文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")
    
    try:
        # 读取图片
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # 预处理图片
        processed_image = preprocess_image(image)
        
        # 预测验证码
        result = predict_captcha(model, processed_image, device)
        
        return PredictionResponse(
            captcha=result["captcha"],
            confidence=result["confidence"],
            processing_time=result["processing_time"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return {"status": "healthy", "model_loaded": model is not None}

def predict_captcha(model, image_tensor, device):
    """
    使用模型预测验证码
    
    Args:
        model: 加载的模型
        image_tensor: 预处理后的图像张量
        device: 计算设备
        
    Returns:
        预测结果字典
    """
    import time
    start_time = time.time()
    
    # 将图像移动到指定设备
    image_tensor = image_tensor.to(device)
    
    # 模型预测
    with torch.no_grad():
        outputs = model(image_tensor)  # [B, T, C]
        outputs = torch.nn.functional.log_softmax(outputs, dim=2)
        
        # CTC解码
        predictions = ctc_greedy_decode(outputs, config.BLANK_IDX)
        
        # 转换为字符串
        if predictions and len(predictions) > 0:
            captcha = ''.join([config.IDX_TO_CHAR[idx] for idx in predictions[0]])
        else:
            captcha = ""
        
        # 计算置信度（使用最大概率的平均值）
        probs = torch.exp(outputs)
        max_probs, _ = torch.max(probs, dim=2)
        confidence = torch.mean(max_probs).item()
    
    processing_time = time.time() - start_time
    
    return {
        "captcha": captcha,
        "confidence": confidence,
        "processing_time": processing_time
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host=config.API_HOST, 
        port=config.API_PORT, 
        reload=config.API_RELOAD
    )