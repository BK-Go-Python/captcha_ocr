import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config.settings import config

class CRNN_CTC(nn.Module):
    """
    CRNN + CTC 模型
    """
    def __init__(self, num_classes, hidden_size=256, num_lstm_layers=2):
        super(CRNN_CTC, self).__init__()
        
        # CNN特征提取器
        self.cnn = nn.Sequential(
            # 第一层
            nn.Conv2d(config.CHANNELS, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2
            
            # 第二层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4
            
            # 第三层
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 第四层
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/8
            
            # 第五层
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 第六层
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/16
            
            # 第七层
            nn.Conv2d(512, 512, kernel_size=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)  # 输出: [B, 512, 1, W']
        )
        
        # LSTM序列建模
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if num_lstm_layers > 1 else 0
        )
        
        # 分类层
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向LSTM
        
    def forward(self, x):
        # CNN提取特征
        conv_features = self.cnn(x)  # [B, C, H, W] -> [B, 512, H', W']
        
        # 调整维度: [B, C, H, W] -> [B, W, C] (序列格式)
        batch_size, channels, height, width = conv_features.size()
        
        # 如果高度不是1，使用平均池化将其压缩为1
        if height > 1:
            conv_features = F.avg_pool2d(conv_features, kernel_size=(height, 1))
        
        # 现在形状应该是 [B, C, 1, W']
        conv_features = conv_features.squeeze(2)  # [B, C, W']
        conv_features = conv_features.permute(0, 2, 1)  # [B, W', C]
        
        # LSTM处理序列
        lstm_out, _ = self.lstm(conv_features)  # [B, W', hidden_size*2]
        
        # 分类
        output = self.fc(lstm_out)  # [B, W', num_classes]
        
        return output

class CRNN_CTC_ResNet(nn.Module):
    """
    使用ResNet作为特征提取器的CRNN
    """
    def __init__(self, num_classes, hidden_size=256, num_lstm_layers=2):
        super(CRNN_CTC_ResNet, self).__init__()
        
        # 使用预训练的ResNet18（去掉最后的全连接层）
        import torchvision.models as models
        resnet = models.resnet18(pretrained=True)
        
        # 移除最后的全连接层和平均池化层
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        
        # 调整通道数
        self.conv_adjust = nn.Conv2d(512, 512, kernel_size=1)
        self.bn_adjust = nn.BatchNorm2d(512)
        self.relu_adjust = nn.ReLU(inplace=True)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if num_lstm_layers > 1 else 0
        )
        
        # 分类层
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # CNN特征提取
        features = self.cnn(x)  # [B, 512, H/32, W/32]
        features = self.conv_adjust(features)
        features = self.bn_adjust(features)
        features = self.relu_adjust(features)
        
        # 调整维度
        features = features.squeeze(2)  # 如果高度不是1，需要平均池化
        if features.dim() == 4:
            features = F.avg_pool2d(features, kernel_size=(features.size(2), 1))
            features = features.squeeze(2)
        
        features = features.permute(0, 2, 1)  # [B, W', 512]
        
        # LSTM
        lstm_out, _ = self.lstm(features)
        
        # 分类
        output = self.fc(lstm_out)
        
        return output

def create_model(model_type='crnn'):
    """
    创建模型
    """
    if model_type == 'crnn':
        model = CRNN_CTC(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            num_lstm_layers=config.NUM_LSTM_LAYERS
        )
    elif model_type == 'resnet_crnn':
        model = CRNN_CTC_ResNet(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            num_lstm_layers=config.NUM_LSTM_LAYERS
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    
    return model.to(config.DEVICE)

def count_parameters(model):
    """
    计算模型参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model):
    """
    打印模型摘要
    """
    print("=" * 80)
    print("模型结构:")
    print(model)
    print("=" * 80)
    print(f"可训练参数数量: {count_parameters(model):,}")
    print("=" * 80)