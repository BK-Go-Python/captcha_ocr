from pydantic import BaseModel, Field
from typing import Optional

class PredictionResponse(BaseModel):
    """预测结果响应模型"""
    captcha: str = Field(..., description="识别的验证码文本")
    confidence: float = Field(..., ge=0.0, le=1.0, description="预测置信度")
    processing_time: float = Field(..., ge=0.0, description="处理时间（秒）")
    
    class Config:
        schema_extra = {
            "example": {
                "captcha": "ABCD12",
                "confidence": 0.95,
                "processing_time": 0.05
            }
        }

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    model_loaded: bool = Field(..., description="模型是否已加载")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True
            }
        }

class InfoResponse(BaseModel):
    """API信息响应模型"""
    message: str = Field(..., description="API描述信息")
    version: str = Field(..., description="API版本")
    device: str = Field(..., description="使用的计算设备")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "验证码识别API",
                "version": "1.0.0",
                "device": "cpu"
            }
        }