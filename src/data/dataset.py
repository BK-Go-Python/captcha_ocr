import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from ..config.settings import config
from ..utils.image_utils import load_image
from ..utils.text_utils import label_to_indices

class CaptchaDataset(Dataset):
    """
    验证码数据集类
    """
    def __init__(self, data_dir, label_file, transform=None, is_train=True):
        """
        Args:
            data_dir: 图片目录
            label_file: 标签文件路径（每行: 图片名,标签）
            transform: 数据增强
            is_train: 是否为训练集
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.char_to_idx = config.CHAR_TO_IDX
        self.blank_idx = config.BLANK_IDX
        
        # 读取标签文件
        self.samples = []
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) >= 2:
                    img_name = parts[0].strip()
                    label = parts[1].strip()
                    
                    # 检查图片是否存在
                    img_path = os.path.join(data_dir, img_name)
                    if os.path.exists(img_path):
                        self.samples.append((img_name, label))
                    else:
                        print(f"警告: 图片不存在 {img_path}")
        
        print(f"加载 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        try:
            # 加载并预处理图片
            image = load_image(img_path, target_height=config.IMG_HEIGHT)
            
            # 数据增强（仅在训练时）
            if self.transform and self.is_train:
                # 转换为PIL Image进行增强
                image_pil = Image.fromarray((image * 255).astype(np.uint8).transpose(1, 2, 0))
                image_pil = self.transform(image_pil)
                image = np.array(image_pil).transpose(2, 0, 1).astype(np.float32) / 255.0
            
            # 转换为Tensor
            image_tensor = torch.FloatTensor(image)
            
            # 将标签转换为索引
            label_indices = label_to_indices(
                label, 
                self.char_to_idx, 
                config.MAX_LABEL_LENGTH
            )
            label_tensor = torch.LongTensor(label_indices)
            
            return image_tensor, label_tensor
            
        except Exception as e:
            print(f"错误处理图片 {img_path}: {e}")
            # 返回一个虚拟样本
            dummy_image = torch.zeros((3, config.IMG_HEIGHT, 100))
            dummy_label = torch.zeros(config.MAX_LABEL_LENGTH, dtype=torch.long)
            return dummy_image, dummy_label
    
    @staticmethod
    def collate_fn(batch):
        """
        自定义批处理函数，处理变长序列
        """
        images, labels = zip(*batch)
        
        # 找到最大宽度
        max_width = max(img.size(2) for img in images)
        
        # 填充图片到相同宽度
        padded_images = []
        for img in images:
            c, h, w = img.size()
            if w < max_width:
                # 右侧填充
                padding = torch.zeros((c, h, max_width - w))
                padded_img = torch.cat([img, padding], dim=2)
            else:
                padded_img = img
            padded_images.append(padded_img)
        
        # 堆叠
        images_batch = torch.stack(padded_images)
        labels_batch = torch.stack(labels)
        
        return images_batch, labels_batch

def get_data_loaders():
    """
    获取训练和验证数据加载器
    """
    from torchvision import transforms
    
    # 数据增强（仅训练时）
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
    ])
    
    # 创建数据集
    train_dataset = CaptchaDataset(
        config.TRAIN_DATA_DIR,
        config.LABEL_FILE,
        transform=train_transform,
        is_train=True
    )
    
    # 划分训练集和验证集 (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=CaptchaDataset.collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=CaptchaDataset.collate_fn,
        pin_memory=True
    )
    
    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本")
    
    return train_loader, val_loader