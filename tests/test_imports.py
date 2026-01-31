"""
测试导入是否正常工作
"""
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_import_config():
    """测试配置模块导入"""
    try:
        from src.config import config, Config, get_config, env_config
        assert config is not None
        assert Config is not None
        assert get_config is not None
        assert env_config is not None
        print("[OK] 配置模块导入成功")
    except ImportError as e:
        print(f"[ERROR] 配置模块导入失败: {e}")
        raise

def test_import_models():
    """测试模型模块导入"""
    try:
        from src.models import create_model, model_summary, count_parameters
        assert create_model is not None
        assert model_summary is not None
        assert count_parameters is not None
        print("[OK] 模型模块导入成功")
    except ImportError as e:
        print(f"[ERROR] 模型模块导入失败: {e}")
        raise

def test_import_utils():
    """测试工具模块导入"""
    try:
        from src.utils import (
            resize_with_aspect_ratio, load_image, preprocess_image,
            label_to_indices, indices_to_label, ctc_greedy_decode, calculate_accuracy,
            visualize_batch, save_checkpoint, load_checkpoint, plot_training_curve
        )
        assert resize_with_aspect_ratio is not None
        assert load_image is not None
        assert preprocess_image is not None
        assert label_to_indices is not None
        assert indices_to_label is not None
        assert ctc_greedy_decode is not None
        assert calculate_accuracy is not None
        assert visualize_batch is not None
        assert save_checkpoint is not None
        assert load_checkpoint is not None
        assert plot_training_curve is not None
        print("[OK] 工具模块导入成功")
    except ImportError as e:
        print(f"[ERROR] 工具模块导入失败: {e}")
        raise

def test_import_data():
    """测试数据模块导入"""
    try:
        from src.data import CaptchaDataset, get_data_loaders
        assert CaptchaDataset is not None
        assert get_data_loaders is not None
        print("[OK] 数据模块导入成功")
    except ImportError as e:
        print(f"[ERROR] 数据模块导入失败: {e}")
        raise

def test_import_api():
    """测试API模块导入"""
    try:
        from src.api import app, predict_captcha, PredictionResponse, HealthResponse, InfoResponse
        assert app is not None
        assert predict_captcha is not None
        assert PredictionResponse is not None
        assert HealthResponse is not None
        assert InfoResponse is not None
        print("[OK] API模块导入成功")
    except ImportError as e:
        print(f"[ERROR] API模块导入失败: {e}")
        raise

def test_all():
    """运行所有导入测试"""
    print("开始测试模块导入...")
    
    test_import_config()
    test_import_models()
    test_import_utils()
    test_import_data()
    test_import_api()
    
    print("所有模块导入测试通过！")

if __name__ == "__main__":
    test_all()