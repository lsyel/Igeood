import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, average_precision_score, 
    roc_curve, auc
)

# =========================================================
# 假设你的项目结构中有这些模块，请保持引用
# 如果你的文件路径不同，请根据实际情况修改这里
# =========================================================
import sys
import os
# sys.path.append("/root/wzhdesign/Igeood") # 如果需要，取消注释并修改路径

try:
    from src.mahalanobis_plus import (
        hidden_feature_estimator, 
        hidden_feature_estimator_ood, 
        get_enhanced_mahalanobis_score
    )
    import utils.data_and_nn_loader as dl
    # import src.ensemble_method as em # 如果需要用到 ensemble_method
except ImportError as e:
    print(f"警告: 无法导入项目模块 ({e})。请确保在正确的项目目录下运行，或将项目根目录添加到 sys.path。")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. 辅助函数：计算详细传统OOD指标 (新增)
# ==========================================
def compute_traditional_ood_metrics(in_scores, out_scores):
    """
    计算传统的OOD检测指标: FPR@TPR95, AUROC, AUPR, Detection Error
    in_scores: 分布内数据的分数 (投票比例，理论上应该较小)
    out_scores: 分布外数据的分数 (投票比例，理论上应该较大)
    """
    # 1. 准备标签和分数
    scores = np.concatenate([in_scores, out_scores])
    # 标签: 0 为分布内 (ID), 1 为分布外 (OOD)
    labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
    
    # 2. AUROC
    # 既然是OOD检测，分数越高越代表是异常(Positive Class)
    auroc = roc_auc_score(labels, scores) * 100
    
    # 3. AUPR (Out) - 正类为OOD
    aupr_out = average_precision_score(labels, scores) * 100
    
    # 4. AUPR (In) - 正类为ID (需要反转分数)
    # ID数据的分数应该更低，所以取负号
    aupr_in = average_precision_score(1 - labels, -scores) * 100
    
    # 获取ROC曲线数据
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # 5. FPR at TPR 95% (Out)
    # 含义：当我们要检测出95%的OOD样本时，有多少ID样本被误报为OOD
    target_tpr = 0.95
    # roc_curve 返回的 tpr 是递增的
    idx = np.searchsorted(tpr, target_tpr)
    if idx >= len(fpr): idx = len(fpr) - 1
    fpr_at_tpr95_out = fpr[idx] * 100
    
    # 6. FPR at TPR 95% (In)
    # 含义：当我们要保留95%的ID样本时，有多少OOD样本被漏判为ID
    # 这里我们翻转问题：将ID视为正类，分数取反
    fpr_in, tpr_in, _ = roc_curve(1-labels, -scores)
    idx_in = np.searchsorted(tpr_in, 0.95)
    if idx_in >= len(fpr_in): idx_in = len(fpr_in) - 1
    fpr_at_tpr95_in = fpr_in[idx_in] * 100

    # 7. Detection Error
    # PE = 0.5 * (1 - TPR) + 0.5 * FPR
    # 假设 ID 和 OOD 先验概率相等 (0.5)
    det_err = np.min(0.5 * (1 - tpr) + 0.5 * fpr) * 100

    print("\n" + "="*40)
    print("🚀 详细性能指标 (Traditional Metrics)")
    print("="*40)
    print(f"FPR at TPR 95% (In):         {fpr_at_tpr95_in:.2f}%")
    print(f"FPR at TPR 95% (Out):        {fpr_at_tpr95_out:.2f}%")
    print(f"Detection error:             {det_err:.2f}%")
    print(f"AUROC:                       {auroc:.2f}%")
    print(f"AUPR (In):                   {aupr_in:.2f}%")
    print(f"AUPR (Out):                  {aupr_out:.2f}%")
    print("="*40 + "\n")
    
    return {
        'fpr_at_tpr95_in': fpr_at_tpr95_in,
        'fpr_at_tpr95_out': fpr_at_tpr95_out,
        'detection_error': det_err,
        'auroc': auroc,
        'aupr_in': aupr_in,
        'aupr_out': aupr_out
    }

def detect_ood_multi_layer_voting(
    nn_name,
    in_dataset_name,
    out_dataset_name,
    eps=0.0,
    batch_size=64,
    gpu=None,
    use_multi_centroid=False,
    threshold_percentile=95, 
    voting_threshold=0.5, 
    num_layers=5,
    name='',
    verbose=True
):
    """
    改进版：基于标准化分数的软投票OOD检测
    """
    if verbose:
        print(f"开始改进版软投票OOD检测: {name}")
    
    # 1. 加载模型和计算中心
    model = dl.load_pre_trained_nn(nn_name, gpu)
    model.eval()
    
    num_classes = dl.get_num_classes(in_dataset_name)
    single_means, inverse, _, multi_means = hidden_feature_estimator(
        nn_name, in_dataset_name, batch_size, gpu, True, max_clusters=5
    )
    ood_sample_mean, ood_inverse, _ = hidden_feature_estimator_ood(
        nn_name, out_dataset_name, batch_size=10, gpu=gpu
    )
    
    sample_mean = multi_means if use_multi_centroid else single_means

    # 2. 计算训练集（分布内）统计量用于标准化
    if verbose:
        print("计算训练集统计量 (用于Z-Score标准化)...")
    in_dataloader = dl.train_dataloader(in_dataset_name, in_dataset_name, batch_size=batch_size)
    
    # [N_train, num_layers]
    train_scores = get_enhanced_mahalanobis_score(
        model, in_dataloader, sample_mean, inverse, ood_sample_mean, ood_inverse,
        num_classes, nn_name, num_layers, eps, gpu, False, use_multi_centroid
    )
    train_scores_abs = np.abs(train_scores)
    
    # === 关键改进 1: 计算每层的均值和标准差 ===
    layer_means = np.mean(train_scores_abs, axis=0)
    layer_stds = np.std(train_scores_abs, axis=0) + 1e-10 # 防止除零
    
    if verbose:
        print(f"各层均值: {layer_means}")
        print(f"各层标准差: {layer_stds}")

    # 3. 计算测试集分数
    if verbose:
        print("计算测试集马氏距离...")
    
    # ID 测试集
    in_test_dataloader = dl.test_dataloader(in_dataset_name, in_dataset_name, batch_size=batch_size)
    in_test_scores = get_enhanced_mahalanobis_score(
        model, in_test_dataloader, sample_mean, inverse, ood_sample_mean, ood_inverse,
        num_classes, nn_name, num_layers, eps, gpu, False, use_multi_centroid
    )
    in_test_scores_abs = np.abs(in_test_scores)
    
    # OOD 测试集
    out_dataloader = dl.test_dataloader(out_dataset_name, out_dataset_name, batch_size=batch_size)
    out_scores = get_enhanced_mahalanobis_score(
        model, out_dataloader, sample_mean, inverse, ood_sample_mean, ood_inverse,
        num_classes, nn_name, num_layers, eps, gpu, False, use_multi_centroid
    )
    out_scores_abs = np.abs(out_scores)
    
    # === 关键改进 2: 软投票计算逻辑 ===
    def compute_soft_scores(raw_scores, means, stds):
        """
        计算标准化后的综合分数
        """
        # Z-Score 标准化: (x - mean) / std
        # 这样每一层的分数都在同一个量级上，表示"偏离均值多少个标准差"
        normalized_scores = (raw_scores - means) / stds
        
        # 综合分数：取各层的平均值 (或者求和)
        # 这就是"软投票"，保留了偏离的程度信息
        soft_scores = np.mean(normalized_scores, axis=1)
        
        return soft_scores, normalized_scores

    # 计算软分数 (连续值，不再是0/1)
    in_soft_scores, _ = compute_soft_scores(in_test_scores_abs, layer_means, layer_stds)
    out_soft_scores, _ = compute_soft_scores(out_scores_abs, layer_means, layer_stds)
    
    # 4. 生成预测标签 (仅用于计算Accuracy/F1等硬指标)
    # 为了兼容之前的代码结构，我们需要选一个阈值将软分数转为硬标签
    # 我们可以使用 TPR95 的阈值，或者简单地使用训练集分数的95分位点作为基准
    
    # 在标准化空间计算阈值 (训练集分数的95%分位点)
    train_soft_scores, _ = compute_soft_scores(train_scores_abs, layer_means, layer_stds)
    unified_threshold = np.percentile(train_soft_scores, threshold_percentile)
    
    if verbose:
        print(f"综合软分数阈值 (基于训练集{threshold_percentile}%): {unified_threshold:.4f}")

    in_predictions = (in_soft_scores > unified_threshold).astype(int)
    out_predictions = (out_soft_scores > unified_threshold).astype(int)
    
    # 5. 准备评估数据
    y_true = np.concatenate([np.zeros(len(in_predictions)), np.ones(len(out_predictions))])
    y_pred = np.concatenate([in_predictions, out_predictions])
    
    # 6. 计算详细指标 (使用连续的软分数)
    # 这里的 in_soft_scores 和 out_soft_scores 是连续的，ROC曲线会非常平滑
    trad_metrics = compute_traditional_ood_metrics(in_soft_scores, out_soft_scores)
    
    # 7. 计算基础指标
    metrics = calculate_ood_metrics(y_true, y_pred, None, verbose)
    
    # 绘制混淆矩阵
    if name:
        plot_confusion_matrix_custom(confusion_matrix(y_true, y_pred), 
                                   figsize=(10, 8), save_path=f'{name}_soft_confusion.png')
    
    # 8. 整合结果
    metrics.update({
        'in_predictions': in_predictions,
        'out_predictions': out_predictions,
        # 这里用软分数替换原来的 vote_ratios
        'in_vote_ratios': in_soft_scores, 
        'out_vote_ratios': out_soft_scores,
        'y_true': y_true,
        'y_pred': y_pred,
        **trad_metrics
    })
    
    return metrics
# ==========================================
# 2. 基础指标计算与绘图函数
# ==========================================
def calculate_ood_metrics(y_true, y_pred, y_scores=None, verbose=True, plot_confusion_matrix=True, figsize=(10, 8)):
    """计算OOD检测的基础分类指标 (Accuracy, F1等)"""
    metrics = {}
    
    # 基础分类指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # 计算特异性（真负率）
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = metrics['recall']
    
    # 混淆矩阵元素
    metrics['confusion_matrix'] = {
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'true_positive': tp
    }
    
    # 计算检测率
    metrics['detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    if verbose:
        print("\n" + "="*50)
        print("OOD检测系统评估结果 (基于固定投票阈值)")
        print("="*50)
        print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"精确率 (Precision): {metrics['precision']:.4f}")
        print(f"召回率 (Recall): {metrics['recall']:.4f}")
        print(f"特异性 (Specificity): {metrics['specificity']:.4f}")
        print(f"F1分数: {metrics['f1_score']:.4f}")
        print(f"误报率 (FAR): {metrics['false_alarm_rate']:.4f}")
        print("="*50)
    return metrics

def plot_confusion_matrix_custom(cm, figsize=(10, 8), save_path=None):
    """绘制混淆矩阵"""
    plt.figure(figsize=figsize)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': '样本数量'})
    ax.set_xlabel('预测标签', fontsize=12, fontweight='bold')
    ax.set_ylabel('真实标签', fontsize=12, fontweight='bold')
    class_names = ['分布内 (ID)', '分布外 (OOD)']
    ax.set_xticklabels(class_names, rotation=0, fontsize=11)
    ax.set_yticklabels(class_names, rotation=0, fontsize=11)
    ax.set_title('混淆矩阵', fontsize=14, fontweight='bold', pad=20)
    
    # 添加百分比
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})"
            ax.text(j + 0.5, i + 0.6, text, ha='center', va='center', 
                    fontsize=10, color='black' if cm_normalized[i, j] < 0.7 else 'white')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {save_path}")
    # plt.show() # 如果在服务器运行，可以注释掉这行

# ==========================================
# 3. 核心检测函数
# ==========================================
def detect_ood_multi_layer_voting_hard(
    nn_name,
    in_dataset_name,
    out_dataset_name,
    eps=0.0,
    batch_size=64,
    gpu=None,
    use_multi_centroid=False,
    threshold_percentile=95, 
    voting_threshold=0.5, 
    num_layers=5,
    name='',
    verbose=True
):
    """
    多层投票OOD检测：每层独立判断，多数投票决定最终结果
    """
    if verbose:
        print(f"开始多层投票OOD检测: {name}")
    
    # 1. 加载模型和计算统计量
    model = dl.load_pre_trained_nn(nn_name, gpu)
    model.eval()
    
    num_classes = dl.get_num_classes(in_dataset_name)
    single_means, inverse, _, multi_means = hidden_feature_estimator(
        nn_name, in_dataset_name, batch_size, gpu, True, max_clusters=5
    )
    ood_sample_mean, ood_inverse, _ = hidden_feature_estimator_ood(
        nn_name, out_dataset_name, batch_size=10, gpu=gpu
    )
    
    sample_mean = multi_means if use_multi_centroid else single_means

    # 2. 计算分布内数据用于确定阈值
    if verbose:
        print("计算分布内数据以确定各层阈值...")
    in_dataloader = dl.train_dataloader(in_dataset_name, in_dataset_name, batch_size=batch_size)
    
    train_scores = get_enhanced_mahalanobis_score(
        model, in_dataloader, sample_mean, inverse, ood_sample_mean, ood_inverse,
        num_classes, nn_name, num_layers, eps, gpu, False, use_multi_centroid
    )
    train_scores_abs = np.abs(train_scores)
    
    # 3. 计算阈值
    layer_thresholds = []
    for layer_idx in range(num_layers):
        layer_scores = train_scores_abs[:, layer_idx]
        threshold = np.percentile(layer_scores, threshold_percentile)
        layer_thresholds.append(threshold)
    
    if verbose:
        print(f"各层{threshold_percentile}%分位数阈值: {layer_thresholds}")

    # 4. 计算测试分数
    if verbose:
        print("计算测试数据的马氏距离...")
    
    # ID 测试集
    in_test_dataloader = dl.test_dataloader(in_dataset_name, in_dataset_name, batch_size=batch_size)
    in_scores = get_enhanced_mahalanobis_score(
        model, in_test_dataloader, sample_mean, inverse, ood_sample_mean, ood_inverse,
        num_classes, nn_name, num_layers, eps, gpu, False, use_multi_centroid
    )
    in_scores_abs = np.abs(in_scores)
    
    # OOD 测试集
    out_dataloader = dl.test_dataloader(out_dataset_name, out_dataset_name, batch_size=batch_size)
    out_scores = get_enhanced_mahalanobis_score(
        model, out_dataloader, sample_mean, inverse, ood_sample_mean, ood_inverse,
        num_classes, nn_name, num_layers, eps, gpu, False, use_multi_centroid
    )
    out_scores_abs = np.abs(out_scores)
    
    # 5. 投票预测逻辑
    def voting_predict(scores_abs, layer_thresholds, voting_threshold):
        n_samples = len(scores_abs)
        num_layers = len(layer_thresholds)
        
        # 每层的判断结果
        layer_decisions = np.zeros((n_samples, num_layers), dtype=bool)
        for layer_idx in range(num_layers):
            layer_decisions[:, layer_idx] = scores_abs[:, layer_idx] > layer_thresholds[layer_idx]
        
        # 投票结果：OOD层数比例 (0.0 ~ 1.0)
        # 这个值将作为后续计算 AUROC 的 Score
        vote_ratios = np.mean(layer_decisions, axis=1)
        
        # 最终预测：硬分类
        final_predictions = (vote_ratios >= voting_threshold).astype(int)
        
        return final_predictions, vote_ratios, layer_decisions
    
    in_predictions, in_vote_ratios, in_layer_decisions = voting_predict(
        in_scores_abs, layer_thresholds, voting_threshold
    )
    out_predictions, out_vote_ratios, out_layer_decisions = voting_predict(
        out_scores_abs, layer_thresholds, voting_threshold
    )
    
    # 6. 准备评估数据
    y_true = np.concatenate([np.zeros(len(in_predictions)), np.ones(len(out_predictions))])
    y_pred = np.concatenate([in_predictions, out_predictions])
    
    # 7. 计算基础指标
    metrics = calculate_ood_metrics(y_true, y_pred, None, verbose)
    cm = confusion_matrix(y_true, y_pred)
    
    # 绘制混淆矩阵（如果name不为空）
    if name:
        plot_confusion_matrix_custom(cm, figsize=(10, 8), save_path=f'{name}_confusion_matrix.png')
    
    # ==========================================
    # 8. [新增] 计算 FPR95, AUROC, AUPR 等指标
    # 使用 投票比例 (vote_ratios) 作为软分数
    # ==========================================
    trad_metrics = compute_traditional_ood_metrics(in_vote_ratios, out_vote_ratios)
    
    # 9. 整合所有结果
    metrics.update({
        'layer_thresholds': layer_thresholds,
        'in_predictions': in_predictions,
        'out_predictions': out_predictions,
        'in_vote_ratios': in_vote_ratios,
        'out_vote_ratios': out_vote_ratios,
        'y_true': y_true,
        'y_pred': y_pred,
        # 加入新指标
        **trad_metrics
    })
    
    return metrics

# ==========================================
# 4. 配置与主执行函数
# ==========================================

# 测试配置
test_configurations = [
    {
        'name': '模型0 混合',
        'nn_name': 'icarl_0',
        'in_dataset_name': 'ustc_task_0_in',
        'out_dataset_name': 'ustc_task_0_out',
        'voting_threshold': 0.6,
        'threshold_percentile': 95,
        
    },
    {
        'name': '模型1 混合',
        'nn_name': 'icarl_1', # 注意：这里你原本写的是 icarl_2, 是否应为 icarl_1? 请确认
        'in_dataset_name': 'ustc_task_1_in', # 同上
        'out_dataset_name': 'ustc_task_1_out',
        'voting_threshold': 0.6,
        'threshold_percentile': 95,
        
    },
    {
        'name': '模型2 混合',
        'nn_name': 'icarl_2',
        'in_dataset_name': 'ustc_task_2_in',
        'out_dataset_name': 'ustc_task_2_out',
        'voting_threshold': 0.6,
        'threshold_percentile': 95,
        
    },
]

# 固定参数
common_params = {
    'batch_size': 64,
    'gpu': "cuda:0",
    'use_multi_centroid': True,
    'num_layers': 5,
    
}

def main_multi_layer_voting():
    """
    主函数：执行多层投票OOD检测
    """
    print("开始多层投票OOD检测实验...")
    print("=" * 60)
    
    all_results = []
    
    for config in test_configurations:
        print(f"\n>>>> 测试: {config['name']}")
        print("-" * 40)
        
        # 合并参数
        params = {**common_params, **config}
        
        # 执行检测
        try:
            result = detect_ood_multi_layer_voting(**params)
            
            # 这里可以根据需要保存结果
            all_results.append({
                'config': config,
                'metrics': result
            })
            
        except Exception as e:
            print(f"❌ 错误: {config['name']} 执行失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n所有实验结束。")

if __name__ == "__main__":
    main_multi_layer_voting()