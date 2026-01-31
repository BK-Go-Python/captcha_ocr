import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from ..config.settings import config

def visualize_batch(images, predictions, targets, idx_to_char=None, blank_idx=None, num_samples=4):
    """
    可视化批处理结果
    """
    if idx_to_char is None:
        idx_to_char = config.IDX_TO_CHAR
    if blank_idx is None:
        blank_idx = config.BLANK_IDX
    
    batch_size = images.size(0)
    num_samples = min(num_samples, batch_size)
    
    # 解码预测结果
    from .text_utils import ctc_greedy_decode
    decoded_preds = ctc_greedy_decode(predictions, blank_idx)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # 获取图片
        img = images[i].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        
        # 获取预测标签
        pred_indices = decoded_preds[i]
        pred_label = ''.join([idx_to_char[idx] for idx in pred_indices])
        
        # 获取真实标签
        target_seq = targets[i].cpu().tolist()
        target_seq = [idx for idx in target_seq if idx != blank_idx]
        target_label = ''.join([idx_to_char[idx] for idx in target_seq])
        
        # 显示图片
        axes[i].imshow(img)
        axes[i].set_title(f'Pred: {pred_label}\\nTrue: {target_label}', fontsize=12)
        axes[i].axis('off')
        
        # 标记错误
        if pred_label != target_label:
            axes[i].set_facecolor('#FFE4E1')  # 浅红色背景表示错误
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(config.LOG_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.LOG_DIR, 'batch_visualization.png'))
    plt.show()

def save_checkpoint(model, optimizer, epoch, accuracy, path):
    """
    保存模型检查点
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'config': config.__dict__
    }
    torch.save(checkpoint, path)
    print(f"检查点已保存到 {path}")

def load_checkpoint(path, model, optimizer=None):
    """
    加载模型检查点
    """
    checkpoint = torch.load(path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"从 epoch {checkpoint['epoch']} 加载检查点，准确率: {checkpoint['accuracy']:.4f}")
    return checkpoint['epoch']

def plot_training_curve(train_losses, val_losses, val_accuracies, save_path=None):
    """
    绘制训练曲线
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='训练损失')
    ax1.plot(epochs, val_losses, 'r-', label='验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.set_title('训练和验证损失')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(epochs, val_accuracies, 'g-', label='验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率')
    ax2.set_title('验证准确率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        plt.savefig(os.path.join(config.LOG_DIR, 'training_curve.png'))
    
    plt.show()