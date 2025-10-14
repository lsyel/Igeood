import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score
import torch

def kmeans_precess_class(c, samples, max_clusters):
    """处理单个类别"""
    # 转换数据格式
    if isinstance(samples, torch.Tensor):
        data = samples.cpu().numpy()
    else:
        data = np.array(samples)
    
    n_samples = len(data)
    
    # 自适应聚类策略
    if n_samples < 10:
        return c, np.mean(data, axis=0, keepdims=True)
    
    # 确定k范围
    max_k = min(max_clusters, max(2, n_samples // 20))  # 确保至少为2
    k_range = list(range(2, max_k + 1))
    
    # 检查k_range是否有效
    if len(k_range) == 0:
        return c, np.mean(data, axis=0, keepdims=True)
    
    # 处理并行计算
    if len(k_range) == 1:
        # 只有一个k值，直接计算
        k = k_range[0]
        _, centers, _ = compute_kmeans(data, k)
        return c, centers
    else:
        # 多个k值，使用并行计算
        results = Parallel(n_jobs=min(len(k_range), 4))(
            delayed(compute_kmeans)(data, k) for k in k_range
        )
        
        # 选择最佳结果
        best_score = -1
        best_centers = None
        for _, centers, score in results:
            if score > best_score:
                best_score = score
                best_centers = centers
        
        return c, best_centers if best_centers is not None else np.mean(data, axis=0, keepdims=True)

def compute_kmeans(data, k):
    """计算单个k值的KMeans"""
    if len(data) < k:
        return k, None, -1
    
    # 使用MiniBatchKMeans加速
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=min(1024, len(data)),  # 自适应批大小
        n_init=3,
        compute_labels=False,
        random_state=42
    )
    kmeans.fit(data)
    centers = kmeans.cluster_centers_
    
    # 计算轮廓系数（使用子采样）
    if k > 1:
        sample_size = min(1000, len(data))
        sample_indices = np.random.choice(len(data), sample_size, replace=False)
        sample_data = data[sample_indices]
        sample_labels = kmeans.predict(sample_data)
        
        try:
            score = silhouette_score(sample_data, sample_labels)
        except:
            score = -1
    else:
        score = 0
        
    return k, centers, score
def evaluate_clustering_quality(sample_class_mean):
    """评估聚类质量并生成报告"""
    quality_report = {}
    
    for layer_idx, class_dict in sample_class_mean.items():
        layer_report = {}
        for class_id, centers in class_dict.items():
            n_clusters = centers.shape[0]
            layer_report[class_id] = {
                'n_clusters': n_clusters,
                'cluster_sizes': [],  # 后续添加
                'quality': 'N/A'  # 后续添加
            }
        
        quality_report[layer_idx] = {
            'class_report': layer_report,
            'avg_clusters': np.mean([c['n_clusters'] for c in layer_report.values()])
        }
    
    return quality_report

def print_clustering_report(report):
    """打印聚类质量报告"""
    for layer_idx, layer_data in report.items():
        print(f"\n=== Layer {layer_idx} ===")
        print(f"平均聚类数: {layer_data['avg_clusters']:.2f}")
        
        # 统计聚类数分布
        cluster_counts = {}
        for class_data in layer_data['class_report'].values():
            n = class_data['n_clusters']
            cluster_counts[n] = cluster_counts.get(n, 0) + 1
        
        print("聚类数分布:")
        for n, count in sorted(cluster_counts.items()):
            print(f"  {n}个聚类: {count}个类别")