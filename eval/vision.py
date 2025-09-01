import matplotlib.pyplot as plt
import numpy as np
import os

# ===================== 数据准备 =====================
methods = ['Fisher-Rao-1', 'Fisher-Rao-3', 'Fisher-Rao-5', 'Fisher-Rao-8', 'Mahalanobis']
data = {
    'AUROC': {'values': [98.82, 99.63, 99.52, 99.20, 99.61], 
              'best_index': 1, 'best_value': 99.63},
    'FPR@95%TPR': {'values': [3.73, 2.44, 1.87, 2.30, 2.43], 
                   'best_index': 2, 'best_value': 1.87},
    'Detection Error': {'values': [3.15, 2.31, 2.23, 2.27, 2.76], 
                       'best_index': 2, 'best_value': 2.23}
}

# 图表配置
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
title_map = {
    'AUROC': 'AUROC Performance (Higher is Better)',
    'FPR@95%TPR': 'FPR@95%TPR Performance (Lower is Better)',
    'Detection Error': 'Detection Error Performance (Lower is Better)'
}
y_label_map = {
    'AUROC': 'AUROC Score (%)',
    'FPR@95%TPR': 'False Positive Rate (%)',
    'Detection Error': 'Error Rate (%)'
}
y_lim_map = {
    'AUROC': (97, 100.5),
    'FPR@95%TPR': (0, 4.5),
    'Detection Error': (0, 3.5)
}
footer_map = {
    'AUROC': "Figure Note: AUROC (Area Under the Receiver Operating Characteristic Curve) measures model's ability to distinguish ID from OOD samples",
    'FPR@95%TPR': "Figure Note: FPR@95%TPR measures the false positive rate when true positive rate is 95% - lower values indicate better OOD detection",
    'Detection Error': "Figure Note: Detection Error combines false positive and false negative rates - lower values indicate better overall OOD detection performance"
}
legend_labels = [
    'Fisher-Rao: 1 centroid', 
    'Fisher-Rao: 3 centroids', 
    'Fisher-Rao: 5 centroids', 
    'Fisher-Rao: 8 centroids', 
    'Mahalanobis'
]

# ===================== 图表生成函数 =====================
def generate_ood_plot(metric_name, output_dir="ood_results"):
    """生成单张OOD检测性能图表并保存为PNG文件"""
    # 准备数据
    values = data[metric_name]['values']
    best_index = data[metric_name]['best_index']
    best_value = data[metric_name]['best_value']
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(title_map[metric_name], fontsize=16, fontweight='bold')
    
    # 绘制柱状图
    bars = ax.bar(methods, values, color=colors)
    
    # 智能放置数据标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        # 根据指标类型和数值位置优化标签放置
        if metric_name == 'AUROC' and value > 99.5:
            va = 'top'
            color = 'white'
            position = height - 0.1
        elif metric_name != 'AUROC' and value < 2.5:
            va = 'top'
            color = 'white'
            position = height - 0.1
        else:
            va = 'bottom'
            color = 'black'
            position = height + 0.15
        ax.text(bar.get_x() + bar.get_width()/2, position,
                f'{value}%', ha='center', va=va,
                fontsize=10, fontweight='bold', color=color)
    
    # 设置坐标轴
    ax.set_ylabel(y_label_map[metric_name], fontsize=12)
    ax.set_ylim(y_lim_map[metric_name])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 标注最佳值
    ax.annotate('Best', 
                xy=(best_index, best_value), 
                xytext=(best_index, best_value + (1 if metric_name=='AUROC' else -1)),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
                fontsize=12, ha='center', color='red', weight='bold')
    
    # 添加图例
    ax.legend(bars, legend_labels, 
              loc='upper center', 
              bbox_to_anchor=(0.5, -0.15),
              ncol=2,
              fontsize=10,
              frameon=True)
    
    # 添加脚注
    plt.figtext(0.5, 0.01, footer_map[metric_name],
                ha='center', fontsize=10, style='italic')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/ood_{metric_name.replace('@', '').replace('%', '').replace(' ', '_')}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"生成图表: {filename}")
    
    return filename

# ===================== 主执行函数 =====================
def generate_all_ood_plots():
    """生成所有OOD检测性能图表"""
    print("开始生成OOD检测性能图表...")
    metrics = list(data.keys())
    generated_files = []
    
    for metric in metrics:
        generated_files.append(generate_ood_plot(metric))
    
    print("\n所有图表生成完成:")
    for file in generated_files:
        print(f"- {file}")
    
    return generated_files

# ===================== 执行入口 =====================
if __name__ == "__main__":
    generate_all_ood_plots()