from sklearn import logger
import torch
import numpy as np
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed
import time

def kmeans_process_class(c, samples, max_clusters, device='cuda'):
    """处理单个类别（使用GPU加速）"""
    # 转换数据格式为PyTorch张量并移到GPU
    if isinstance(samples, torch.Tensor):
        data = samples.to(device)
    else:
        data = torch.tensor(samples, dtype=torch.float32, device=device)
    # 确定k范围（从1到max_clusters）
    k_range = list(range(2, max_clusters + 1))
    
    # 计算所有k值的聚类结果
    results = []
    for k in k_range:
        # 计算聚类
        _, centers, score = compute_kmeans_gpu(data, k)
        results.append((k, centers, score))
    
    # 选择轮廓系数最高的结果
    best_score = -1
    best_centers = None
    for k, centers, score in results:
        # print(f"[debug]k={k}, 轮廓系数={score:.4f}")
        if score > best_score:
            best_score = score
            best_centers = centers
    # print(f"[debug]选择k={best_centers.shape[0]}, 轮廓系数={best_score:.4f}")
    return c, best_centers.cpu().numpy()

def compute_kmeans_gpu(data, k):
    """使用GPU加速的KMeans实现"""
    n_samples, n_features = data.shape
    
    # 对于k=1的特殊情况
    if k == 1:
        centers = torch.mean(data, dim=0, keepdim=True)
        return k, centers, 0.0  # k=1时轮廓系数设为0
    
    # 随机初始化质心
    indices = torch.randperm(n_samples)[:k]
    centroids = data[indices].clone()
    
    # KMeans迭代
    max_iter = 100
    prev_centroids = centroids.clone()
    
    for _ in range(max_iter):
        # 计算距离矩阵
        distances = torch.cdist(data, centroids)
        
        # 分配样本到最近的质心
        labels = torch.argmin(distances, dim=1)
        
        # 更新质心
        new_centroids = centroids.clone()
        for i in range(k):
            mask = (labels == i)
            if mask.sum() > 0:  # 确保有样本分配到该质心
                new_centroids[i] = data[mask].mean(dim=0)
        
        # 检查收敛
        if torch.allclose(new_centroids, centroids, atol=1e-4):
            break
        
        centroids = new_centroids
    
    # 计算轮廓系数（使用子采样）
    sample_size = min(1000, n_samples)
    sample_indices = torch.randperm(n_samples)[:sample_size]
    sample_data = data[sample_indices]
    
    # 计算样本距离
    sample_distances = torch.cdist(sample_data, centroids)
    sample_labels = torch.argmin(sample_distances, dim=1)
    
    # 转换为numpy计算轮廓系数
    try:
        score = silhouette_score(
            sample_data.cpu().numpy(), 
            sample_labels.cpu().numpy()
        )
    except:
        score = -1
    
    return k, centroids, score

# 修改聚类质量评估函数
def evaluate_clustering_quality(sample_class_mean):
    """评估聚类质量并生成报告"""
    quality_report = {}
    
    for layer_idx, class_dict in sample_class_mean.items():
        layer_report = {}
        for class_id, centers in class_dict.items():
            # 正确计算聚类数
            if isinstance(centers, np.ndarray):
                n_clusters = centers.shape[0]  # NumPy数组的第一维是聚类数
            elif isinstance(centers, torch.Tensor):
                n_clusters = centers.size(0)  # PyTorch张量的第一维是聚类数
            elif isinstance(centers, list):
                n_clusters = len(centers)
            else:
                n_clusters = 1
            
            layer_report[class_id] = {
                'n_clusters': n_clusters,
                'quality': 'N/A'  # 后续添加
            }
        
        # 计算平均聚类数
        avg_clusters = np.mean([c['n_clusters'] for c in layer_report.values()])
        
        quality_report[layer_idx] = {
            'class_report': layer_report,
            'avg_clusters': avg_clusters
        }
    
    return quality_report
def print_clustering_report(report):
    """打印聚类质量报告"""
    print("\n===== 聚类质量报告 =====")
    
    for layer_idx, layer_data in report.items():
        print(f"\n=== 层 {layer_idx} ===")
        print(f"平均聚类数: {layer_data['avg_clusters']:.2f}")
        
        # 统计聚类数分布
        cluster_counts = {}
        for class_data in layer_data['class_report'].values():
            n = class_data['n_clusters']
            cluster_counts[n] = cluster_counts.get(n, 0) + 1
        
        print("聚类数分布:")
        for n, count in sorted(cluster_counts.items()):
            print(f"  {n}个聚类: {count}个类别")
        
        # 打印每个类别的聚类情况
        print("\n类别详细聚类情况:")
        for class_id, class_data in layer_data['class_report'].items():
            print(f"  类别 {class_id}: {class_data['n_clusters']}个聚类")