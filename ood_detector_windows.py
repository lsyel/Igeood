import numpy as np
import logging
from src.ensemble_method import WeightRegression

from sklearn.ensemble import IsolationForest
from src.mahalanobis_plus import *

import numpy as np
import logging
import collections
import time
from sklearn.utils import shuffle

# 配置日志格式，模拟真实系统日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("OOD_System")

import numpy as np
import logging
import collections
import time
from sklearn.utils import shuffle
import matplotlib.pyplot as plt  # [新增] 导入绘图库

# ... (保留原有的 import 和 logging 配置) ...
# 配置日志格式，模拟真实系统日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("OOD_System")

# 
# 这里的标签是为了触发生成的图表概念，实际运行代码会生成具体的图

import numpy as np
import logging
import collections
import time
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("OOD_System")

def simulate_realtime_ood_detection(
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
    window_size=1000,       
    alarm_threshold=0.3,
    ood_burst_scale=1.0,      # OOD 突发流量时长缩放因子
    id_background_ratio=5.0   # ID 背景流量相对于 OOD 总量的倍数
):
    print(f"\n🚀 开始模拟实时OOD检测流程...")
    print(f"⚙️ 配置: 窗口={window_size}, 告警阈值={alarm_threshold:.0%}")
    
    # ---------------------------------------------------------
    # 1. 准备阶段 (保持逻辑不变)
    # ---------------------------------------------------------
    print("⏳ [系统初始化] 加载模型与计算统计量...")
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

    print("🔧 [系统校准] 计算各层判决阈值...")
    in_dataloader = dl.train_dataloader(in_dataset_name, in_dataset_name, batch_size=batch_size)
    train_scores = get_enhanced_mahalanobis_score(
        model, in_dataloader, sample_mean, inverse, ood_sample_mean, ood_inverse,
        num_classes, nn_name, num_layers, eps, gpu, False, use_multi_centroid
    )
    train_scores_abs = np.abs(train_scores)
    
    layer_thresholds = []
    for layer_idx in range(num_layers):
        threshold = np.percentile(train_scores_abs[:, layer_idx], threshold_percentile)
        layer_thresholds.append(threshold)
    
    # ---------------------------------------------------------
    # 2. 构建混合测试数据流 (自动补全与增强)
    # ---------------------------------------------------------
    print("\n🌊 [数据流构建] 正在生成增强型混合流量...")

    # === 获取原始测试数据 ===
    in_test_dataloader = dl.test_dataloader(in_dataset_name, in_dataset_name, batch_size=batch_size)
    in_scores = get_enhanced_mahalanobis_score(
        model, in_test_dataloader, sample_mean, inverse, ood_sample_mean, ood_inverse,
        num_classes, nn_name, num_layers, eps, gpu, False, use_multi_centroid
    )
    in_data_pool = np.abs(in_scores)
    in_labels_pool = np.zeros(len(in_data_pool))

    out_dataloader = dl.test_dataloader(out_dataset_name, out_dataset_name, batch_size=batch_size)
    out_scores = get_enhanced_mahalanobis_score(
        model, out_dataloader, sample_mean, inverse, ood_sample_mean, ood_inverse,
        num_classes, nn_name, num_layers, eps, gpu, False, use_multi_centroid
    )
    out_data_pool = np.abs(out_scores)
    out_labels_pool = np.ones(len(out_data_pool))

    # === 配置突发流量 (Burst) ===
    burst_1_len = int(window_size * 0.2 * ood_burst_scale)
    burst_2_len = int(window_size * 0.8 * ood_burst_scale)
    total_ood_needed = burst_1_len + burst_2_len

    # 自动填充 OOD
    if len(out_data_pool) < total_ood_needed:
        repeat_needed = int(np.ceil(total_ood_needed / len(out_data_pool)))
        out_data_pool = np.concatenate([out_data_pool] * repeat_needed, axis=0)
        out_labels_pool = np.concatenate([out_labels_pool] * repeat_needed, axis=0)
    
    out_data_pool, out_labels_pool = shuffle(out_data_pool, out_labels_pool, random_state=42)
    ood_burst_1_data = out_data_pool[:burst_1_len]
    ood_burst_1_labels = out_labels_pool[:burst_1_len]
    ood_burst_2_data = out_data_pool[burst_1_len : burst_1_len + burst_2_len]
    ood_burst_2_labels = out_labels_pool[burst_1_len : burst_1_len + burst_2_len]

    # === 配置背景流量 (ID Data) ===
    target_id_len = int(total_ood_needed * id_background_ratio)
    
    # 自动填充 ID
    if len(in_data_pool) < target_id_len:
        repeat_needed = int(np.ceil(target_id_len / len(in_data_pool)))
        in_data_pool = np.concatenate([in_data_pool] * repeat_needed, axis=0)
        in_labels_pool = np.concatenate([in_labels_pool] * repeat_needed, axis=0)
    
    in_data_pool, in_labels_pool = shuffle(in_data_pool, in_labels_pool, random_state=42)
    in_data_final = in_data_pool[:target_id_len]
    in_labels_final = in_labels_pool[:target_id_len]

    # === 拼接混合流 ===
    insert_idx_1 = int(len(in_data_final) * 0.2)
    insert_idx_2 = int(len(in_data_final) * 0.7) 
    
    mixed_scores = np.concatenate([
        in_data_final[:insert_idx_1],             
        ood_burst_1_data,                         
        in_data_final[insert_idx_1:insert_idx_2], 
        ood_burst_2_data,                         
        in_data_final[insert_idx_2:]              
    ], axis=0)
    
    mixed_labels = np.concatenate([
        in_labels_final[:insert_idx_1],
        ood_burst_1_labels,
        in_labels_final[insert_idx_1:insert_idx_2],
        ood_burst_2_labels,
        in_labels_final[insert_idx_2:]
    ], axis=0)
    
    # ---------------------------------------------------------
    # 3. 实时检测循环模拟
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("🔴 开始实时检测")
    
    window_history = collections.deque([0] * window_size, maxlen=window_size)   
    detected_ood_count = 0
    alarm_triggered_count = 0
    total_steps = len(mixed_scores)
    record_interval = max(1, int(window_size * 0.1))
    
    plot_x = []
    plot_y = [] 
    
    for i in range(total_steps):
        sample_score = mixed_scores[i]      
        
        # 检测逻辑
        layer_decisions = sample_score[:num_layers] > layer_thresholds
        vote_ratio = np.mean(layer_decisions)
        is_ood_pred = vote_ratio >= voting_threshold
        
        # 状态更新
        pred_val = 1 if is_ood_pred else 0
        window_history.append(pred_val)
        current_ood_ratio = sum(window_history) / len(window_history) if len(window_history) > 0 else 0
        
        # 记录
        if i % record_interval == 0:
            plot_x.append(i)
            plot_y.append(current_ood_ratio)
        
        if is_ood_pred:
            detected_ood_count += 1
        
        if current_ood_ratio > alarm_threshold:
            alarm_triggered_count += 1

        if i % 2000 == 0:
             logger.info(f"Step {i:06d}/{total_steps} | Ratio: {current_ood_ratio:.2%}")

    # ---------------------------------------------------------
    # 4. 结果可视化绘图 (中文版 + 无 Ground Truth)
    # ---------------------------------------------------------
    print("\n🎨 正在生成中文分析图...")
    
    # === [关键配置] 设置中文字体 ===
    # 优先尝试常见中文字体，防止乱码
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12, 5)) 
    
    # 1. 绘制曲线
    plt.plot(plot_x, plot_y, color='#1f77b4', linewidth=1.5, label='实时 OOD 比例')
    
    # 2. 绘制阈值线
    plt.axhline(y=alarm_threshold, color='r', linestyle='--', linewidth=2, label=f'告警阈值 ({alarm_threshold})')
    
    # 3. [已移除] 不再绘制 Ground Truth 散点图
    
    # 4. 中文标签与标题
    plt.xlabel('时间步 (采样点)', fontsize=12)
    plt.ylabel('滑动窗口中OOD 比例', fontsize=12)
    
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.15) # 顶部留白给手动标注
    
    save_path_png = 'ood_sim_cn.png'
    save_path_svg = 'ood_sim_cn.svg'
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_svg, format='svg', bbox_inches='tight')
    
    print(f"✅ 中文图表已保存:\n  - PNG: {save_path_png}\n  - SVG: {save_path_svg}")

    # ---------------------------------------------------------
    # 5. 总结
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("🏁 模拟结束")
    print(f"总样本数: {total_steps}")
    print(f"检出 OOD: {detected_ood_count}")
    print(f"告警时长: {alarm_triggered_count} 步")
    print("="*80)
def main_simulation():
    # 示例配置，取你在 test_configurations 中的混合配置
    config = {
        'name': '模型0 混合模拟',
        'nn_name': 'icarl_0',
        'in_dataset_name': 'ustc_task_0_in',
        'out_dataset_name': 'ustc_task_0_out',
        'voting_threshold': 0.4
    }
    common_params = {
        'batch_size': 64,
        'gpu': 0,
        'use_multi_centroid': True,
        'num_layers': 5,
    }
    # 合并参数
    params = {**common_params, **config}
    
    # 增加模拟特有的参数
    params['window_size'] = 200    # 窗口大小可以设小一点方便观察波动
    params['alarm_threshold'] = 0.2 # 当窗口内超过40%是OOD时报警
    
    # 去掉不被函数接受的参数（如果原来的params里有name等不必要的）
    # 这里直接传递，需要在函数定义里接收 **kwargs 或者手动清洗
    # 为简单起见，我们直接调用
    simulate_realtime_ood_detection(
        nn_name=params['nn_name'],
        in_dataset_name=params['in_dataset_name'],
        out_dataset_name=params['out_dataset_name'],
        eps=0.0,
        batch_size=params['batch_size'],
        gpu=params['gpu'],
        use_multi_centroid=params['use_multi_centroid'],
        voting_threshold=params['voting_threshold'],
        num_layers=params['num_layers'],
        window_size=params['window_size'],
        alarm_threshold=params['alarm_threshold']
    )

if __name__ == "__main__":
    # 原有的测试
    # results = main_multi_layer_voting()
    
    # 新的模拟运行
    main_simulation()