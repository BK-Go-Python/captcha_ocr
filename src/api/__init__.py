"""
API模块

包含FastAPI应用和相关的API端点。
"""

from .main import app, predict_captcha
from .schemas import PredictionResponse, HealthResponse, InfoResponse

__all__ = ['app', 'predict_captcha', 'PredictionResponse', 'HealthResponse', 'InfoResponse']