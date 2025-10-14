import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

ROOT = "/root/wzhdesign/Igeood"
sys.path.append(f"{ROOT}/")
from models.my_resnet.inc_model import IncModel
from utils.data_and_nn_loader import *

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """绘制混淆矩阵"""
    plt.figure(figsize=(12, 10))
    
    # 创建DataFrame以便更好地显示
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # 使用seaborn绘制热力图
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()
    logger.info(f"已保存混淆矩阵图像: {title.replace(' ', '_')}.png")

def evaluate_model(model, dataloader, device):
    """评估模型在给定数据集上的准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="评估中"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 获取预测结果
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs
                
            _, preds = torch.max(logits, 1)
            
            # 统计正确预测数
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    
    accuracy = 100 * correct / total
    return accuracy

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 1. 加载模型
    model_path = args['model_path']
    logger.info(f"加载模型: {model_path}")
    model = IncModel(model_path)
    model.model.to(device)
    model.model.eval()
    
    # 2. 准备数据集
    dataset_name = "ustc_task_0_in"
    transform_name = "ustc_transform"
    
    logger.info(f"准备数据集: {dataset_name}")
    
    # 获取训练集和测试集
    logger.info("创建数据加载器...")
    train_loader = train_dataloader(dataset_name, transform_name, batch_size=128)
    test_loader = test_dataloader(dataset_name, transform_name, batch_size=128)
    
    # 3. 获取当前任务的类别信息
    task_class_labels = train_loader.dataset.classes
    logger.info(f"当前任务类别标签: {task_class_labels}")
    num_classes = len(task_class_labels)
    
    # 4. 评估模型
    logger.info("\n评估模型性能:")
    logger.info("=" * 50)
    
    # 评估训练集准确率
    train_acc = evaluate_model(model.model, train_loader, device)
    logger.info(f"训练集准确率: {train_acc:.2f}%")
    
    # 评估测试集准确率
    test_acc = evaluate_model(model.model, test_loader, device)
    logger.info(f"测试集准确率: {test_acc:.2f}%")
    
    # 5. 评估每个类别的准确率
    logger.info("\n评估每个类别的准确率:")
    logger.info("=" * 50)
    
    # 初始化统计变量
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    model.model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="评估类别准确率"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model.model(inputs)
            
            # 获取预测结果
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs
                
            _, preds = torch.max(logits, 1)
            
            # 统计每个类别的正确预测数
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += (preds[i] == targets[i]).item()
                class_total[label] += 1
    
    # 打印每个类别的准确率
    logger.info("\n类别准确率:")
    logger.info("=" * 50)
    logger.info(f"{'类别':<15} | {'样本数':<8} | {'正确数':<8} | {'准确率':<8}")
    logger.info("-" * 50)
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
        else:
            acc = 0.0
        logger.info(f"{task_class_labels[i]:<15} | {class_total[i]:<8} | {class_correct[i]:<8} | {acc:.2f}%")
    
    # 6. 计算并打印混淆矩阵
    logger.info("\n计算混淆矩阵...")
    
    # 收集所有预测和目标
    all_targets = []
    all_preds = []
    
    model.model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="收集混淆矩阵数据"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model.model(inputs)
            
            # 获取预测结果
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs
                
            _, preds = torch.max(logits, 1)
            
            # 收集所有预测和目标
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
    
    # 打印数值混淆矩阵
    logger.info("\n混淆矩阵（数值）:")
    logger.info("=" * 50)
    logger.info(cm)
    
    # 打印归一化混淆矩阵（百分比）
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    logger.info("\n混淆矩阵（归一化百分比）:")
    logger.info("=" * 50)
    logger.info(np.round(cm_norm * 100, 2))
    
    # 绘制并保存混淆矩阵图像
    plot_confusion_matrix(cm, task_class_labels, title=f'Confusion Matrix for {dataset_name}')
    
    # 7. 打印每个类别的精确率、召回率和F1分数
    logger.info("\n详细分类报告:")
    logger.info("=" * 50)
    logger.info(f"{'类别':<15} | {'精确率':<8} | {'召回率':<8} | {'F1分数':<8}")
    logger.info("-" * 50)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info(f"{task_class_labels[i]:<15} | {precision:.4f} | {recall:.4f} | {f1:.4f}")

if __name__ == '__main__':
    args = {
        'model_path': '/root/wzhdesign/Igeood/pre_trained/task_0_model.pth'
    }
    main(args)