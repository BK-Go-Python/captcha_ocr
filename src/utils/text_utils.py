import torch
from ..config.settings import config

def label_to_indices(label, char_to_idx=None, max_length=None):
    """
    将标签字符串转换为索引列表
    """
    if char_to_idx is None:
        char_to_idx = config.CHAR_TO_IDX
    if max_length is None:
        max_length = config.MAX_LABEL_LENGTH
    
    indices = []
    for char in label:
        if char in char_to_idx:
            indices.append(char_to_idx[char])
        else:
            # 未知字符用最后一个索引（blank）代替
            indices.append(len(char_to_idx))
    
    # 填充到固定长度（如果需要）
    if len(indices) < max_length:
        indices.extend([len(char_to_idx)] * (max_length - len(indices)))
    
    return indices[:max_length]

def indices_to_label(indices, idx_to_char=None, blank_idx=None):
    """
    将索引列表转换回标签字符串（移除重复和blank）
    """
    if idx_to_char is None:
        idx_to_char = config.IDX_TO_CHAR
    if blank_idx is None:
        blank_idx = config.BLANK_IDX
    
    chars = []
    prev_idx = blank_idx
    
    for idx in indices:
        if idx != blank_idx and idx != prev_idx:
            chars.append(idx_to_char[idx])
        prev_idx = idx
    
    return ''.join(chars)

def ctc_greedy_decode(predictions, blank_idx=None):
    """
    CTC贪婪解码
    Args:
        predictions: (T, C) 或 (B, T, C)
    Returns:
        decoded_labels: 解码后的标签列表
    """
    if blank_idx is None:
        blank_idx = config.BLANK_IDX
    
    if predictions.dim() == 3:
        # 批量解码
        decoded_labels = []
        for i in range(predictions.size(0)):
            pred = predictions[i]  # (T, C)
            _, indices = torch.max(pred, dim=1)  # (T,)
            
            # 移除重复和blank
            decoded = []
            prev_idx = blank_idx
            for idx in indices:
                if idx != blank_idx and idx != prev_idx:
                    decoded.append(idx.item())
                prev_idx = idx
            
            decoded_labels.append(decoded)
        return decoded_labels
    else:
        # 单样本解码
        pred = predictions  # (T, C)
        _, indices = torch.max(pred, dim=1)  # (T,)
        
        # 移除重复和blank
        decoded = []
        prev_idx = blank_idx
        for idx in indices:
            if idx != blank_idx and idx != prev_idx:
                decoded.append(idx.item())
            prev_idx = idx
        
        return [decoded]

def calculate_accuracy(predictions, targets, blank_idx=None):
    """
    计算准确率（字符级别和序列级别）
    """
    if blank_idx is None:
        blank_idx = config.BLANK_IDX
    
    # 贪婪解码
    decoded_preds = ctc_greedy_decode(predictions, blank_idx)
    
    # 将targets转换为列表形式
    if targets.dim() == 2:
        targets_list = []
        for i in range(targets.size(0)):
            target_seq = targets[i].cpu().tolist()
            # 移除padding（blank_idx）
            target_seq = [idx for idx in target_seq if idx != blank_idx]
            targets_list.append(target_seq)
    else:
        targets_list = [targets.cpu().tolist()]
    
    char_correct = 0
    char_total = 0
    seq_correct = 0
    
    for pred_seq, target_seq in zip(decoded_preds, targets_list):
        # 字符级别准确率
        min_len = min(len(pred_seq), len(target_seq))
        for i in range(min_len):
            if pred_seq[i] == target_seq[i]:
                char_correct += 1
        char_total += len(target_seq)
        
        # 序列级别准确率
        if pred_seq == target_seq:
            seq_correct += 1
    
    char_accuracy = char_correct / char_total if char_total > 0 else 0
    seq_accuracy = seq_correct / len(targets_list)
    
    return char_accuracy, seq_accuracy