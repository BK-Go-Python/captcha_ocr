# 验证码识别系统 (CRNN + CTC + FastAPI)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于深度学习的验证码识别系统，使用CRNN+CTC模型和FastAPI框架提供服务。

## 🌟 功能特点

- 🔤 **高精度识别**：基于CRNN+CTC深度学习模型，支持多种验证码类型
- 🚀 **高性能API**：使用FastAPI构建，支持异步处理和自动文档生成
- 🔍 **模型可视化**：提供CNN特征图、梯度流和CTC对齐路径的可视化工具
- 📊 **训练监控**：实时监控训练过程，自动保存最佳模型
- 🧪 **完整测试**：提供API测试脚本和模型评估工具
- 🏗️ **规范化结构**：采用标准的项目结构，便于维护和扩展

## 📁 项目结构

```
captcha_ocr/
├── src/                     # 源代码目录
│   ├── api/                # API模块
│   │   ├── main.py         # FastAPI应用主文件
│   │   └── schemas.py      # 请求和响应模型
│   ├── config/             # 配置模块
│   │   ├── settings.py     # 基础配置
│   │   └── env_config.py   # 环境配置
│   ├── data/               # 数据处理模块
│   │   └── dataset.py      # 数据集处理
│   ├── models/             # 模型模块
│   │   └── crnn_model.py   # CRNN模型定义
│   └── utils/              # 工具模块
│       ├── image_utils.py  # 图像处理工具
│       ├── text_utils.py   # 文本处理工具
│       └── model_utils.py  # 模型工具
├── scripts/                # 脚本目录
│   ├── train.py            # 训练脚本
│   ├── predict.py          # 预测脚本
│   └── run_api.py          # API启动脚本
├── tests/                  # 测试目录
├── docs/                   # 文档目录
├── data/                   # 数据集目录
│   ├── train/              # 训练数据
│   └── test/               # 测试数据
├── models/                 # 模型文件
│   └── best_model.pth      # 最佳模型
├── logs/                   # 日志和可视化结果
├── requirements.txt        # 依赖项列表
├── setup.py               # 安装脚本
└── README.md              # 项目说明
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.3+ (可选，用于GPU加速)

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/yourusername/captcha_ocr.git
cd captcha_ocr

# 安装依赖
pip install -r requirements.txt

# 或者作为包安装
pip install -e .
```

### 训练模型

```bash
# 开始训练
python scripts/train.py
```

训练过程会自动：
- 加载数据集并进行预处理
- 创建CRNN+CTC模型
- 训练并验证模型
- 保存最佳模型到 `models/best_model.pth`
- 生成训练曲线图到 `logs/training_curve.png`

### 启动API服务

```bash
# 启动FastAPI服务
python scripts/run_api.py
```

或者使用uvicorn：

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后，可以访问：
- API文档：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/health`

### 使用API

#### 上传图片进行识别

```bash
curl -X POST -F "file=@/path/to/captcha.png" http://localhost:8000/predict
```

响应示例：
```json
{
  "captcha": "ABCD12",
  "confidence": 0.95,
  "processing_time": 0.05
}
```

#### Python客户端示例

```python
import requests

# 读取图片
with open('captcha.png', 'rb') as f:
    files = {'file': f}
    
# 发送请求
response = requests.post('http://localhost:8000/predict', files=files)

# 获取结果
result = response.json()
print(f"验证码: {result['captcha']}")
print(f"置信度: {result['confidence']:.2f}")
```

### 使用预测脚本

```bash
# 预测单张图片
python scripts/predict.py --image /path/to/captcha.png

# 批量预测
python scripts/predict.py --batch --test_dir /path/to/images

# 在测试集上评估
python scripts/predict.py --test_dir /path/to/test --label_file /path/to/labels.txt
```

## 🔧 配置

### 环境配置

项目支持多种环境配置：

- `development`: 开发环境
- `production`: 生产环境
- `testing`: 测试环境

通过环境变量 `ENV` 指定：

```bash
export ENV=production
python scripts/run_api.py
```

### 自定义配置

编辑 `src/config/settings.py` 文件：

```python
class Config:
    # 数据集配置
    IMG_HEIGHT = 64           # 图片统一高度
    CHANNELS = 3              # 彩色图像
    NUM_CLASSES = 63          # 62个字符 + 1个blank（CTC用）
    MAX_LABEL_LENGTH = 6      # 验证码最大长度
    
    # 训练配置
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # 模型配置
    HIDDEN_SIZE = 256
    NUM_LSTM_LAYERS = 2
```

### 添加新字符

1. 更新 `src/config/settings.py` 中的 `CHARSET` 列表
2. 重新训练模型

## 📊 模型架构

### CRNN+CTC模型

本系统使用CRNN（卷积循环神经网络）结合CTC（连接主义时间分类）的架构：

1. **CNN特征提取器**：
   - 7层卷积网络
   - 批量归一化和ReLU激活
   - 最大池化层减少空间维度

2. **序列建模**：
   - 双向LSTM层
   - 捕获序列中的上下文信息

3. **CTC解码器**：
   - 处理可变长度序列
   - 无需对齐标签即可训练

### 模型性能

在标准验证码数据集上的性能：

| 指标 | 值 |
|------|-----|
| 字符准确率 | 95%+ |
| 序列准确率 | 85%+ |
| 平均推理时间 | < 50ms |

## 🧪 测试

### API测试

```bash
# 运行API测试
python -m pytest tests/api/
```

### 模型测试

```bash
# 运行模型测试
python -m pytest tests/models/
```

### 完整测试

```bash
# 运行所有测试
python -m pytest
```

## 🛠️ 开发

### 代码风格

项目使用以下工具保证代码质量：

- **Black**: 代码格式化
- **Flake8**: 代码风格检查
- **MyPy**: 类型检查

运行代码检查：

```bash
# 格式化代码
black src/ scripts/ tests/

# 检查代码风格
flake8 src/ scripts/ tests/

# 类型检查
mypy src/
```

### 添加新功能

1. 创建功能分支
2. 编写代码和测试
3. 运行测试确保通过
4. 提交Pull Request

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [FastAPI](https://fastapi.tiangolo.com/) - 现代、快速的Web框架
- [CTC](https://distill.pub/2017/ctc/) - 连接主义时间分类

## 📞 联系

如有问题或建议，请通过以下方式联系：

- 创建 Issue
- 发送邮件至：your.email@example.com

---

⭐ 如果这个项目对您有帮助，请考虑给个星标！