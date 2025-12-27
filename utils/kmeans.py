import torch
import numpy as np
from joblib import Parallel, delayed
import time
from collections import defaultdict

# === 全局缓存：用于存储每个类别的轮廓系数 ===
GLOBAL_SCORE_CACHE = defaultdict(list)

def silhouette_score_gpu(X, labels):
    """纯 PyTorch 实现的轮廓系数计算 (完全在GPU上运行)"""
    unique_labels = torch.unique(labels)
    num_samples = X.shape[0]
    
    if len(unique_labels) <= 1:
        return torch.tensor(0.0, device=X.device)

    distances = torch.cdist(X, X)

    a = torch.zeros(num_samples, device=X.device)
    b = torch.full((num_samples,), float('inf'), device=X.device)

    for label in unique_labels:
        mask = (labels == label)
        cluster_size = mask.sum()
        
        if cluster_size > 1:
            d_cluster = distances[mask][:, mask]
            a[mask] = d_cluster.sum(dim=1) / (cluster_size - 1)
        else:
            a[mask] = 0.0
            
        for other_label in unique_labels:
            if label == other_label: continue
            other_mask = (labels == other_label)
            if other_mask.sum() == 0: continue
                
            d_other = distances[mask][:, other_mask]
            mean_d_other = d_other.mean(dim=1)
            b[mask] = torch.minimum(b[mask], mean_d_other)

    max_ab = torch.maximum(a, b)
    s = (b - a) / max_ab
    s[max_ab == 0] = 0
    return s.mean()

def compute_kmeans_gpu(data, k):
    """使用GPU加速的KMeans实现"""
    n_samples, n_features = data.shape
    
    if k == 1:
        return k, torch.mean(data, dim=0, keepdim=True), 0.0
    
    indices = torch.randperm(n_samples, device=data.device)[:k]
    centroids = data[indices].clone()
    
    for _ in range(100): 
        distances = torch.cdist(data, centroids)
        labels = torch.argmin(distances, dim=1)
        new_centroids = centroids.clone()
        
        for i in range(k):
            mask = (labels == i)
            if mask.any():
                new_centroids[i] = data[mask].mean(dim=0)
        
        if torch.allclose(new_centroids, centroids, atol=1e-4):
            break
        centroids = new_centroids
    
    sample_size = min(1000, n_samples)
    if n_samples > sample_size:
        indices = torch.randperm(n_samples, device=data.device)[:sample_size]
        sample_data = data[indices]
    else:
        sample_data = data
    
    sample_dists = torch.cdist(sample_data, centroids)
    sample_labels = torch.argmin(sample_dists, dim=1)
    
    try:
        score_tensor = silhouette_score_gpu(sample_data, sample_labels)
        score = score_tensor.item()
    except:
        score = -1
    
    return k, centroids, score

def kmeans_process_class(c, samples, max_clusters, device='cuda:0'):
    """
    处理单个类别
    【修改点】：增加阈值判断，小于0.5则不聚类
    """
    if isinstance(samples, torch.Tensor):
        data = samples.to(device)
    else:
        data = torch.as_tensor(samples, dtype=torch.float32, device=device)
    
    best_score = -1.0
    best_centers = None
    
    # 寻找最佳 K
    for k in range(2, max_clusters + 1):
        _, centers, score = compute_kmeans_gpu(data, k)
        # 寻找最高分
        if score > best_score:
            best_score = score
            best_centers = centers
            
    # logic modification start ===========
    # 1. 如果没有找到合适的中心 (best_centers is None)
    # 2. 或者 最佳分数小于 0.5 (best_score < 0.5)
    # 则强制使用 k=1 (即所有数据的均值)
    if best_centers is None or best_score < 0.5:
        best_centers = torch.mean(data, dim=0, keepdim=True)
        best_score = 0.0 # k=1 时分数记为 0.0，表示未聚类
    # logic modification end =============

    # 记录分数到全局缓存
    global GLOBAL_SCORE_CACHE
    GLOBAL_SCORE_CACHE[c].append(best_score)
    
    # 返回结果
    return c, best_centers.cpu().numpy()

def evaluate_clustering_quality(sample_class_mean):
    """评估聚类质量并生成报告"""
    quality_report = {}
    score_consume_index = defaultdict(int)
    sorted_layers = sorted(sample_class_mean.keys())
    
    for layer_idx in sorted_layers:
        class_dict = sample_class_mean[layer_idx]
        layer_report = {}
        
        for class_id, centers in class_dict.items():
            if isinstance(centers, np.ndarray):
                n_clusters = centers.shape[0]
            elif isinstance(centers, torch.Tensor):
                n_clusters = centers.size(0)
            elif isinstance(centers, list):
                n_clusters = len(centers)
            else:
                n_clusters = 1
            
            score = 0.0
            idx = score_consume_index[class_id]
            if class_id in GLOBAL_SCORE_CACHE and idx < len(GLOBAL_SCORE_CACHE[class_id]):
                score = GLOBAL_SCORE_CACHE[class_id][idx]
                score_consume_index[class_id] += 1
            
            layer_report[class_id] = {
                'n_clusters': n_clusters,
                'score': score 
            }
        
        all_n = [v['n_clusters'] for v in layer_report.values()]
        all_s = [v['score'] for v in layer_report.values()]
        
        quality_report[layer_idx] = {
            'class_report': layer_report,
            'avg_clusters': np.mean(all_n) if all_n else 0,
            'avg_score': np.mean(all_s) if all_s else 0
        }
    
    return quality_report

def print_clustering_report(report):
    """打印详细的聚类质量报告"""
    print("\n" + "="*30)
    print("      聚类质量详细报告")
    print("="*30)
    
    for layer_idx, layer_data in report.items():
        print(f"\n>>> Layer {layer_idx}")
        print(f"  - 平均聚类数: {layer_data['avg_clusters']:.2f}")
        print(f"  - 平均轮廓系数: {layer_data['avg_score']:.4f}")
        
        print(f"  - 类别详细数据:")
        sorted_classes = sorted(layer_data['class_report'].items(), key=lambda x: int(x[0]) if isinstance(x[0], (int, str)) and str(x[0]).isdigit() else x[0])
        
        print(f"    {'Class ID':<10} | {'Clusters':<10} | {'Score':<10}")
        print(f"    {'-'*10}-+-{'-'*10}-+-{'-'*10}")
        for class_id, info in sorted_classes:
            print(f"    {str(class_id):<10} | {info['n_clusters']:<10} | {info['score']:.4f}")