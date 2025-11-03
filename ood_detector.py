import numpy as np
import logging
from src.ensemble_method import WeightRegression

from sklearn.ensemble import IsolationForest
from src.mahalanobis_plus import *

# 配置日志
logger = logging.getLogger(__name__)
    # 测试配置
test_configurations = [
        {
            'name': '模型0 内分布',
            'nn_name': 'icarl_0',
            'in_dataset_name': 'ustc_task_0_in',
            'test_dataset_name': 'ustc_task_0_in',
            'voting_threshold': 0.2  # 60%的层认为是OOD则判定为OOD
            
        },
        {
            'name': '模型0 外分布',
            'nn_name': 'icarl_0',
            'in_dataset_name': 'ustc_task_0_in',
            'test_dataset_name': 'ustc_task_0_out',
            'voting_threshold': 0.2  # 60%的层认为是OOD则判定为OOD
            
        },
        {
            'name': '模型0 混合',
            'nn_name': 'icarl_0',
            'in_dataset_name': 'ustc_task_0_in',
            'test_dataset_name': 'ustc_task_1_in',
            'voting_threshold': 0.2  # 60%的层认为是OOD则判定为OOD
            
        },
        {
            'name':'模型1 内分布',   
            'nn_name': 'icarl_1',
            'in_dataset_name': 'ustc_task_1_in',
            'test_dataset_name': 'ustc_task_1_in',
            'voting_threshold': 0.2  # 60%的层认为是OOD则判定为OOD
            
        },
        {
            'name':'模型1 外分布',   
            'nn_name': 'icarl_1',
            'in_dataset_name': 'ustc_task_1_in',
            'test_dataset_name': 'ustc_task_1_out',
            'voting_threshold': 0.2  # 60%的层认为是OOD则判定为OOD
            
        },
        {
            'name':'模型1 混合',   
            'nn_name': 'icarl_1',
            'in_dataset_name': 'ustc_task_1_in',
            'test_dataset_name': 'ustc_task_2_in',
            'voting_threshold': 0.2  # 60%的层认为是OOD则判定为OOD
            
        },
        {
            'name':'模型2 内分布',   
            'nn_name': 'icarl_2',
            'in_dataset_name': 'ustc_task_2_in',
            'test_dataset_name': 'ustc_task_2_in',
            'voting_threshold': 0.4  # 60%的层认为是OOD则判定为OOD
            
        },
        {
            'name':'模型2 外分布',   
            'nn_name': 'icarl_2',
            'in_dataset_name': 'ustc_task_2_in',
            'test_dataset_name': 'ustc_task_2_out',
            'voting_threshold': 0.4  # 60%的层认为是OOD则判定为OOD
            
        },
        {
            'name':'模型2 混合',   
            'nn_name': 'icarl_2',
            'in_dataset_name': 'ustc_task_2_in',
            'test_dataset_name': 'ustc_task_3_in',
            'voting_threshold': 0.4  # 60%的层认为是OOD则判定为OOD
            
        },
    ]
    
    # 固定参数
common_params = {
        'batch_size': 64,
        'gpu': 0,
        'use_multi_centroid': True,
        'num_layers': 5,
    }

def detect_ood_multi_layer_voting(
    nn_name,
    in_dataset_name,
    test_dataset_name,
    eps=0.0,
    batch_size=64,
    gpu=None,
    use_multi_centroid=False,
    threshold_percentile=95,  # 每层的阈值百分位数
    voting_threshold=0.5,     # 投票阈值：多少比例的层认为是OOD才判定为OOD
    num_layers=5,
    name='',
):
    """
    多层投票OOD检测：每层独立判断，多数投票决定最终结果
    """
    print("开始多层投票OOD检测...")
    
    # 1. 加载模型和计算统计量
    model = dl.load_pre_trained_nn(nn_name, gpu)
    model.eval()
    
    num_classes = dl.get_num_classes(in_dataset_name)
    single_means, inverse, _, multi_means = hidden_feature_estimator(
        nn_name, in_dataset_name, batch_size, gpu, True, max_clusters=5
    )
    ood_sample_mean, ood_inverse, _ = hidden_feature_estimator_ood(
        nn_name, test_dataset_name, batch_size=10, gpu=gpu
    )
    
    sample_mean = multi_means if use_multi_centroid else single_means

    # 2. 计算分布内数据的马氏距离（用于确定各层阈值）
    print("计算分布内数据以确定各层阈值...")
    in_dataloader = dl.train_dataloader(in_dataset_name, in_dataset_name, batch_size=batch_size)
    
    in_scores = get_enhanced_mahalanobis_score(
        model, in_dataloader, sample_mean, inverse, ood_sample_mean, ood_inverse,
        num_classes, nn_name, num_layers, eps, gpu, False, use_multi_centroid
    )
    
    # 取绝对值
    in_scores_abs = np.abs(in_scores)
    # 3. 为每层计算阈值（95%分位数）
    layer_thresholds = []
    for layer_idx in range(num_layers):
        layer_scores = in_scores_abs[:, layer_idx]
        threshold = np.percentile(layer_scores, threshold_percentile)
        layer_thresholds.append(threshold)
        
        # 打印每层的阈值信息
        print(f"层{layer_idx}: 阈值={threshold:.4f} "
                   f"(范围: {np.min(layer_scores):.2f}-{np.max(layer_scores):.2f})")
    
    print(f"各层{threshold_percentile}%分位数阈值: {layer_thresholds}")

    # 4. 计算测试数据的马氏距离
    print("计算测试数据的马氏距离...")
    test_dataloader = dl.test_dataloader(test_dataset_name, in_dataset_name, batch_size=batch_size)
    
    test_scores = get_enhanced_mahalanobis_score(
        model, test_dataloader, sample_mean, inverse, ood_sample_mean, ood_inverse,
        num_classes, nn_name, num_layers, eps, gpu, False, use_multi_centroid
    )
    
    test_scores_abs = np.abs(test_scores)
    
    # 5. 进行多层投票检测
    n_test_samples = len(test_scores_abs)
    
    # 每层的判断结果
    layer_decisions = np.zeros((n_test_samples, num_layers), dtype=bool)
    
    for layer_idx in range(num_layers):
        threshold = layer_thresholds[layer_idx]
        layer_scores = test_scores_abs[:, layer_idx]
        layer_decisions[:, layer_idx] = layer_scores > threshold
    
    # 计算每样本的OOD票数
    ood_votes_per_sample = np.sum(layer_decisions, axis=1)
    
    # 多数投票：超过 voting_threshold 比例的层认为是OOD则判定为OOD
    vote_threshold_count = int(voting_threshold * num_layers)
    is_ood = ood_votes_per_sample >= vote_threshold_count
    
    ood_count = np.sum(is_ood)
    total_samples = n_test_samples
    ood_ratio = ood_count / total_samples
    
    # 6. 打印关键结果
    print(f"多层投票检测结果:")
    print(f"  总样本数: {total_samples}")
    print(f"  OOD样本数: {ood_count} ({ood_ratio*100:.2f}%)")
    print(f"  投票阈值: {vote_threshold_count}/{num_layers}层")
    print(f"  平均OOD票数: {np.mean(ood_votes_per_sample):.2f}层")
    
    # 7. 返回简化结果
    return {
        'name': name,
        'ood_count': ood_count,
        'total_samples': total_samples,
        'ood_ratio': ood_ratio,
        'layer_thresholds': layer_thresholds,
        'avg_ood_votes': np.mean(ood_votes_per_sample),
        'vote_distribution': np.bincount(ood_votes_per_sample, minlength=num_layers+1)
    }

def analyze_voting_performance(results, num_layers):
    """
    分析投票性能
    """
    ood_count = results['ood_count']
    total_samples = results['total_samples']
    ood_ratio = results['ood_ratio']
    avg_ood_votes = results['avg_ood_votes']
    vote_distribution = results['vote_distribution']
    name = results['name']
    print("\n" + "="*60)
    print(f"多层投票OOD检测结果 - {name}")
    print("="*60)
    
    print(f"📊 总体结果:")
    print(f"   总样本数: {total_samples}")
    print(f"   OOD检测数: {ood_count} ({ood_ratio*100:.2f}%)")
    print(f"   平均OOD票数: {avg_ood_votes:.2f}/{num_layers}层")
    
    print(f"\n🎯 各层阈值:")
    for i, threshold in enumerate(results['layer_thresholds']):
        print(f"   层{i}: {threshold:.4f}")
    
    print(f"\n📈 投票分布:")
    for votes in range(num_layers + 1):
        count = vote_distribution[votes]
        percentage = count / total_samples * 100
        is_ood = votes >= (num_layers // 2 + 1)  # 多数判定
        status = "🚩 OOD" if is_ood else "✅ 分布内"
        print(f"   {votes}层投票OOD: {count:4d}样本 ({percentage:5.1f}%) {status}")
    
    # 计算一致性
    unanimous_ood = vote_distribution[num_layers]  # 所有层都认为是OOD
    unanimous_in = vote_distribution[0]  # 所有层都认为是分布内
    unanimous_ratio = (unanimous_ood + unanimous_in) / total_samples * 100
    
    print(f"\n🔍 一致性分析:")
    print(f"   完全一致OOD: {unanimous_ood}样本 ({unanimous_ood/total_samples*100:.1f}%)")
    print(f"   完全一致分布内: {unanimous_in}样本 ({unanimous_in/total_samples*100:.1f}%)")
    print(f"   完全一致比例: {unanimous_ratio:.1f}%")
    
    # 层间一致性分析
    if unanimous_ratio > 70:
        print("   ✅ 层间判断高度一致")
    elif unanimous_ratio < 30:
        print("   ⚠️  层间判断差异较大")
    else:
        print("   📊 层间判断一致性一般")
    
    print("="*60)
    
    return {
        'unanimous_ood': unanimous_ood,
        'unanimous_in': unanimous_in,
        'unanimous_ratio': unanimous_ratio
    }
def detect_ood_regression(
    nn_name,
    in_dataset_name,
    test_dataset_name,
    eps=0.0,
    batch_size=64,
    gpu=None,
    use_multi_centroid=False,
    threshold_percentile=95,  # 每层的阈值百分位数
    voting_threshold=0.5,     # 投票阈值：多少比例的层认为是OOD才判定为OOD
    num_layers=5,
    name='',
):
    """
    多层投票OOD检测：每层独立判断，多数投票决定最终结果
    """
    print("开始多层投票OOD检测...")
    
    # 1. 加载模型和计算统计量
    model = dl.load_pre_trained_nn(nn_name, gpu)
    model.eval()
    
    num_classes = dl.get_num_classes(in_dataset_name)
    single_means, inverse, _, multi_means = hidden_feature_estimator(
        nn_name, in_dataset_name, batch_size, gpu, True, max_clusters=5
    )
    ood_sample_mean, ood_inverse, _ = hidden_feature_estimator_ood(
        nn_name, test_dataset_name, batch_size=10, gpu=gpu
    )
    
    sample_mean = multi_means if use_multi_centroid else single_means

    # 2. 计算分布内数据的马氏距离（用于确定各层阈值）
    print("计算分布内数据以确定各层阈值...")
    in_dataloader = dl.train_dataloader(in_dataset_name, in_dataset_name, batch_size=batch_size)
    
    in_scores = get_enhanced_mahalanobis_score(
        model, in_dataloader, sample_mean, inverse, ood_sample_mean, ood_inverse,
        num_classes, nn_name, num_layers, eps, gpu, False, use_multi_centroid
    )
    
    
    # 4. 计算测试数据的马氏距离
    print("计算测试数据的马氏距离...")
    test_dataloader = dl.test_dataloader(test_dataset_name, in_dataset_name, batch_size=batch_size)
    
    test_scores = get_enhanced_mahalanobis_score(
        model, test_dataloader, sample_mean, inverse, ood_sample_mean, ood_inverse,
        num_classes, nn_name, num_layers, eps, gpu, False, use_multi_centroid
    )
    
    classifier = WeightRegression()
    s_in,s_test = classifier(in_scores, test_scores)
    (
        fpr_at_tpr_in,
        fpr_at_tpr_out,
        detection,
        auroc,
        aupr_in,
        aupr_out,
    ) = em.print_metrics_and_info(
        s_in,
        s_test,
        nn_name,
        nn_name + '_' + in_dataset_name+'_'+test_dataset_name,
        test_dataset_name,
        'detect_ood',
        True,
        False,
        True,
    )

def main_multi_layer_voting():
    """
    主函数：执行多层投票OOD检测
    """
    print("开始多层投票OOD检测实验...")
    print("=" * 60)
    
    all_results = []
    
    for config in test_configurations:
        print(f"\n测试: {config['name']}")
        print("-" * 40)
        
        # 合并参数
        params = {**common_params, **config}
        
        # 执行检测
        result = detect_ood_multi_layer_voting(**params)
        
        # 分析性能
        analysis = analyze_voting_performance(result, common_params['num_layers'])
        
        # 存储结果
        all_results.append({
            'config': config,
            'result': result,
            'analysis': analysis
        })
        
        print(f"✅ 完成: OOD检测率 = {result['ood_ratio']*100:.2f}%")
            
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("多层投票OOD检测汇总结果")
    print("=" * 60)
    
    successful_results = [r for r in all_results if 'result' in r]
    if successful_results:
        ood_ratios = [r['result']['ood_ratio'] for r in successful_results]
        avg_ratio = np.mean(ood_ratios) * 100
        min_ratio = np.min(ood_ratios) * 100
        max_ratio = np.max(ood_ratios) * 100
        
        print(f"📊 总体统计:")
        print(f"   平均OOD检测率: {avg_ratio:.2f}%")
        print(f"   范围: {min_ratio:.2f}% - {max_ratio:.2f}%")
        print(f"   测试数量: {len(successful_results)}")
        
        print(f"\n📋 详细结果:")
        for result in successful_results:
            config = result['config']
            ood_ratio = result['result']['ood_ratio'] * 100
            avg_votes = result['result']['avg_ood_votes']
            unanimous = result['analysis']['unanimous_ratio']
            
            print(f"   {config['name']}: {ood_ratio:5.1f}% OOD "
                  f"(平均{avg_votes:.1f}票, 一致性{unanimous:.1f}%)")
    
    print("=" * 60)
    
    return all_results

def main_regression():
    """
    主函数：执行回归OOD检测
    """
    print("开始回归OOD检测实验...")
    print("=" * 60)
    for config in test_configurations[1::3]:
        print(f"\n测试: {config['name']}")
        print("-" * 40)
        
        # 合并参数
        params = {**common_params, **config}
        
        # 执行检测
        detect_ood_regression(**params)
        
            
    

if __name__ == "__main__":
    # results = main_multi_layer_voting()
    main_regression()
    