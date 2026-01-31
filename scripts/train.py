import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import config
from src.data import get_data_loaders
from src.models import create_model, model_summary
from src.utils import calculate_accuracy, visualize_batch, save_checkpoint, load_checkpoint

class Trainer:
    def __init__(self, model_type='crnn'):
        # 创建目录
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        
        # 创建模型
        self.model = create_model(model_type)
        model_summary(self.model)
        
        # 数据加载器
        self.train_loader, self.val_loader = get_data_loaders()
        
        # 损失函数（CTC Loss）
        self.criterion = nn.CTCLoss(
            blank=config.BLANK_IDX,
            reduction='mean',
            zero_infinity=True
        )
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_accuracy = 0.0
        self.start_epoch = 0
        
        print(f"使用设备: {config.DEVICE}")
        print(f"批量大小: {config.BATCH_SIZE}")
        print(f"学习率: {config.LEARNING_RATE}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_char_acc = 0.0
        total_seq_acc = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [训练]')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # 移动到设备
            images = images.to(config.DEVICE, non_blocking=True)
            targets = targets.to(config.DEVICE, non_blocking=True)
            
            # 前向传播
            outputs = self.model(images)  # [B, T, C]
            outputs = F.log_softmax(outputs, dim=2)
            
            # 准备CTC输入
            batch_size = images.size(0)
            input_lengths = torch.full(
                size=(batch_size,),
                fill_value=outputs.size(1),
                dtype=torch.long,
                device=config.DEVICE
            )
            
            target_lengths = torch.full(
                size=(batch_size,),
                fill_value=config.MAX_LABEL_LENGTH,
                dtype=torch.long,
                device=config.DEVICE
            )
            
            # 计算CTC损失
            loss = self.criterion(
                outputs.permute(1, 0, 2),  # [T, B, C] (CTC要求)
                targets,
                input_lengths,
                target_lengths
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            # 计算准确率
            with torch.no_grad():
                char_acc, seq_acc = calculate_accuracy(
                    outputs, 
                    targets, 
                    config.BLANK_IDX
                )
            
            # 更新统计
            total_loss += loss.item()
            total_char_acc += char_acc
            total_seq_acc += seq_acc
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'char_acc': f'{char_acc:.2%}',
                'seq_acc': f'{seq_acc:.2%}'
            })
        
        # 计算epoch平均
        avg_loss = total_loss / num_batches
        avg_char_acc = total_char_acc / num_batches
        avg_seq_acc = total_seq_acc / num_batches
        
        self.train_losses.append(avg_loss)
        
        return avg_loss, avg_char_acc, avg_seq_acc
    
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        total_char_acc = 0.0
        total_seq_acc = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [验证]')
            
            for batch_idx, (images, targets) in enumerate(pbar):
                # 移动到设备
                images = images.to(config.DEVICE)
                targets = targets.to(config.DEVICE)
                
                # 前向传播
                outputs = self.model(images)
                outputs = F.log_softmax(outputs, dim=2)
                
                # 准备CTC输入
                batch_size = images.size(0)
                input_lengths = torch.full(
                    size=(batch_size,),
                    fill_value=outputs.size(1),
                    dtype=torch.long,
                    device=config.DEVICE
                )
                
                target_lengths = torch.full(
                    size=(batch_size,),
                    fill_value=config.MAX_LABEL_LENGTH,
                    dtype=torch.long,
                    device=config.DEVICE
                )
                
                # 计算CTC损失
                loss = self.criterion(
                    outputs.permute(1, 0, 2),
                    targets,
                    input_lengths,
                    target_lengths
                )
                
                # 计算准确率
                char_acc, seq_acc = calculate_accuracy(
                    outputs, 
                    targets, 
                    config.BLANK_IDX
                )
                
                # 更新统计
                total_loss += loss.item()
                total_char_acc += char_acc
                total_seq_acc += seq_acc
                num_batches += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'char_acc': f'{char_acc:.2%}',
                    'seq_acc': f'{seq_acc:.2%}'
                })
                
                # 可视化第一批结果
                if batch_idx == 0 and epoch % 5 == 0:
                    visualize_batch(
                        images[:4], 
                        outputs[:4], 
                        targets[:4], 
                        config.IDX_TO_CHAR,
                        config.BLANK_IDX
                    )
        
        # 计算epoch平均
        avg_loss = total_loss / num_batches
        avg_char_acc = total_char_acc / num_batches
        avg_seq_acc = total_seq_acc / num_batches
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_seq_acc)
        
        return avg_loss, avg_char_acc, avg_seq_acc
    
    def save_model(self, epoch, accuracy, is_best=False):
        """保存模型"""
        if is_best:
            model_path = os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth')
        else:
            model_path = os.path.join(config.MODEL_SAVE_DIR, f'model_epoch_{epoch+1}.pth')
        
        save_checkpoint(
            self.model,
            self.optimizer,
            epoch,
            accuracy,
            model_path
        )
    
    def load_model(self, model_path):
        """加载模型"""
        self.start_epoch = load_checkpoint(
            model_path,
            self.model,
            self.optimizer
        )
    
    def train(self, resume_from=None):
        """主训练循环"""
        # 恢复训练（如果指定）
        if resume_from:
            self.load_model(resume_from)
            print(f"从 epoch {self.start_epoch} 恢复训练")
        
        print("开始训练...")
        
        for epoch in range(self.start_epoch, config.EPOCHS):
            start_time = time.time()
            
            # 训练
            train_loss, train_char_acc, train_seq_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_char_acc, val_seq_acc = self.validate(epoch)
            
            epoch_time = time.time() - start_time
            
            # 打印epoch总结
            print(f"\\nEpoch {epoch+1}/{config.EPOCHS} 完成 ({epoch_time:.1f}s)")
            print(f"训练 - 损失: {train_loss:.4f}, 字符准确率: {train_char_acc:.2%}, 序列准确率: {train_seq_acc:.2%}")
            print(f"验证 - 损失: {val_loss:.4f}, 字符准确率: {val_char_acc:.2%}, 序列准确率: {val_seq_acc:.2%}")
            
            # 更新学习率
            self.scheduler.step(val_seq_acc)
            
            # 保存最佳模型
            if val_seq_acc > self.best_accuracy:
                self.best_accuracy = val_seq_acc
                self.save_model(epoch, val_seq_acc, is_best=True)
                print(f"新的最佳模型! 序列准确率: {val_seq_acc:.2%}")
            
            # 每5个epoch保存一次检查点
            if (epoch + 1) % 5 == 0:
                self.save_model(epoch, val_seq_acc, is_best=False)
            
            # 提前停止条件
            if epoch > 20 and val_seq_acc < 0.01:
                print("准确率太低，提前停止")
                break
        
        print(f"训练完成! 最佳验证准确率: {self.best_accuracy:.2%}")
        
        # 绘制训练曲线
        self.plot_training_curve()
    
    def plot_training_curve(self):
        """绘制训练曲线"""
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(epochs, self.train_losses, 'b-', label='训练损失')
        ax1.plot(epochs, self.val_losses, 'r-', label='验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.set_title('训练和验证损失')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(epochs, self.val_accuracies, 'g-', label='验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('准确率')
        ax2.set_title('验证准确率')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.LOG_DIR, 'training_curve.png'))
        plt.show()

def main():
    """主函数"""
    # 创建训练器
    trainer = Trainer(model_type='crnn')
    
    # 开始训练
    trainer.train()
    
    # 也可以从检查点恢复训练
    # trainer.train(resume_from='./models/best_model.pth')

if __name__ == "__main__":
    main()