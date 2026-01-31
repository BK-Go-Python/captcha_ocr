import os
import torch
import numpy as np
from PIL import Image
import cv2
import glob
import argparse
import sys

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from src.config.settings import config
from src.models import create_model
from src.utils.image_utils import preprocess_image
from src.utils.text_utils import ctc_greedy_decode

class CaptchaRecognizer:
    """验证码识别器"""
    def __init__(self, model_path=None):
        # 创建模型
        self.model = create_model('crnn')
        self.model.eval()
        
        # 加载权重
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=config.DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型已加载: {model_path}")
        else:
            # 如果没有指定路径，尝试加载最佳模型
            best_model = os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth')
            if os.path.exists(best_model):
                checkpoint = torch.load(best_model, map_location=config.DEVICE)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"已加载最佳模型: {best_model}")
            else:
                print("警告: 未找到模型权重，使用随机初始化")
        
        # 移动到设备
        self.model.to(config.DEVICE)
    
    def predict(self, image):
        """
        预测单张图片
        """
        # 预处理
        tensor = preprocess_image(image)
        tensor = tensor.to(config.DEVICE)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(tensor)
            outputs = torch.nn.functional.log_softmax(outputs, dim=2)
        
        # 解码
        decoded_indices = ctc_greedy_decode(outputs, config.BLANK_IDX)[0]
        
        # 转换为字符串
        predicted_chars = []
        for idx in decoded_indices:
            if idx in config.IDX_TO_CHAR:
                predicted_chars.append(config.IDX_TO_CHAR[idx])
        
        predicted_label = ''.join(predicted_chars)
        
        # 计算置信度
        if len(decoded_indices) > 0:
            probs = torch.exp(outputs[0])  # 转换为概率
            confidence = 1.0
            for i, idx in enumerate(decoded_indices):
                if i < probs.size(0) and idx < probs.size(1):
                    confidence *= probs[i, idx].item()
            confidence = confidence ** (1.0 / max(1, len(decoded_indices)))
        else:
            confidence = 0.0
        
        return predicted_label, confidence
    
    def predict_batch(self, image_paths):
        """
        批量预测
        """
        results = []
        
        for img_path in image_paths:
            try:
                label, confidence = self.predict(img_path)
                results.append({
                    'file': os.path.basename(img_path),
                    'prediction': label,
                    'confidence': confidence
                })
            except Exception as e:
                results.append({
                    'file': os.path.basename(img_path),
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        return results
    
    def evaluate_on_testset(self, test_dir, label_file=None):
        """
        在测试集上评估模型
        """
        # 如果没有标签文件，只进行预测
        if label_file is None:
            image_files = glob.glob(os.path.join(test_dir, '*.jpg')) + \
                         glob.glob(os.path.join(test_dir, '*.png'))
            results = self.predict_batch(image_files)
            return results
        
        # 如果有标签文件，计算准确率
        from src.utils.text_utils import label_to_indices
        
        # 读取标签
        labels_dict = {}
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    img_name = parts[0].strip()
                    label = parts[1].strip()
                    labels_dict[img_name] = label
        
        # 预测所有图片
        image_files = []
        true_labels = []
        
        for img_name, true_label in labels_dict.items():
            img_path = os.path.join(test_dir, img_name)
            if os.path.exists(img_path):
                image_files.append(img_path)
                true_labels.append(true_label)
        
        # 批量预测
        predictions = self.predict_batch(image_files)
        
        # 计算准确率
        correct = 0
        char_correct = 0
        char_total = 0
        
        for pred, true_label in zip(predictions, true_labels):
            if pred['prediction'] == true_label:
                correct += 1
            
            # 字符级别准确率
            min_len = min(len(pred['prediction']), len(true_label))
            for i in range(min_len):
                if pred['prediction'][i] == true_label[i]:
                    char_correct += 1
            char_total += len(true_label)
        
        seq_accuracy = correct / len(true_labels) if true_labels else 0
        char_accuracy = char_correct / char_total if char_total > 0 else 0
        
        print("=" * 50)
        print(f"测试集评估结果:")
        print(f"序列准确率: {seq_accuracy:.2%} ({correct}/{len(true_labels)})")
        print(f"字符准确率: {char_accuracy:.2%} ({char_correct}/{char_total})")
        print("=" * 50)
        
        # 打印错误样本
        print("\\n错误样本:")
        error_count = 0
        for pred, true_label, img_file in zip(predictions, true_labels, image_files):
            if pred['prediction'] != true_label and error_count < 10:
                print(f"图片: {os.path.basename(img_file)}")
                print(f"预测: {pred['prediction']} (置信度: {pred['confidence']:.2f})")
                print(f"真实: {true_label}")
                print("-" * 30)
                error_count += 1
        
        return {
            'seq_accuracy': seq_accuracy,
            'char_accuracy': char_accuracy,
            'predictions': predictions,
            'true_labels': true_labels
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='验证码识别推理')
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--image', type=str, help='单张图片路径')
    parser.add_argument('--test_dir', type=str, help='测试集目录')
    parser.add_argument('--label_file', type=str, help='测试集标签文件')
    parser.add_argument('--batch', action='store_true', help='批量预测目录下的所有图片')
    
    args = parser.parse_args()
    
    # 创建识别器
    recognizer = CaptchaRecognizer(args.model)
    
    # 单张图片预测
    if args.image:
        if os.path.exists(args.image):
            prediction, confidence = recognizer.predict(args.image)
            print(f"图片: {os.path.basename(args.image)}")
            print(f"预测结果: {prediction}")
            print(f"置信度: {confidence:.4f}")
        else:
            print(f"图片不存在: {args.image}")
    
    # 批量预测
    elif args.batch and args.test_dir:
        if not os.path.exists(args.test_dir):
            print(f"目录不存在: {args.test_dir}")
            return
        
        # 获取所有图片
        image_files = glob.glob(os.path.join(args.test_dir, '*.jpg')) + \
                     glob.glob(os.path.join(args.test_dir, '*.png')) + \
                     glob.glob(os.path.join(args.test_dir, '*.jpeg'))
        
        if not image_files:
            print(f"目录中没有找到图片: {args.test_dir}")
            return
        
        print(f"找到 {len(image_files)} 张图片")
        results = recognizer.predict_batch(image_files)
        
        # 打印结果
        for result in results:
            print(f"{result['file']}: {result['prediction']} (置信度: {result['confidence']:.2f})")
    
    # 测试集评估
    elif args.test_dir and args.label_file:
        if not os.path.exists(args.test_dir):
            print(f"测试集目录不存在: {args.test_dir}")
            return
        
        if not os.path.exists(args.label_file):
            print(f"标签文件不存在: {args.label_file}")
            return
        
        recognizer.evaluate_on_testset(args.test_dir, args.label_file)
    
    else:
        # 交互模式
        print("验证码识别系统 (交互模式)")
        print("输入图片路径或输入 'quit' 退出")
        
        while True:
            user_input = input("\\n请输入图片路径: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not os.path.exists(user_input):
                print(f"文件不存在: {user_input}")
                continue
            
            try:
                prediction, confidence = recognizer.predict(user_input)
                print(f"预测结果: {prediction}")
                print(f"置信度: {confidence:.4f}")
                
                # 显示图片
                try:
                    import matplotlib.pyplot as plt
                    img = cv2.imread(user_input)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.imshow(img)
                    plt.title(f"预测: {prediction}")
                    plt.axis('off')
                    plt.show()
                except:
                    pass
                    
            except Exception as e:
                print(f"预测失败: {e}")

if __name__ == "__main__":
    main()