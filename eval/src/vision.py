import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置图形清晰度
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'  # 使用通用字体确保兼容性

# 创建数据
data = {
    'Task': ['Task 0', 'Task 0', 'Task 1', 'Task 1', 'Task 2', 'Task 2'],
    'Method': ['Mahalanobis', 'Mahalanobis+', 'Mahalanobis', 'Mahalanobis+', 'Mahalanobis', 'Mahalanobis+'],
    'FPR@95%(In)': [11.40, 2.60, 25.50, 1.20, 15.89, 0.29],
    'FPR@95%(Out)': [14.40, 0.90, 41.47, 0.27, 25.46, 0.10],
    'Detection Error(%)': [7.55, 2.55, 14.80, 1.40, 9.26, 0.81],
    'AUROC(%)': [96.65, 99.20, 92.94, 99.51, 95.54, 99.88],
    'AUPR(In)(%)': [96.46, 98.60, 97.17, 99.76, 91.91, 99.78],
    'AUPR(Out)(%)': [95.56, 99.42, 86.95, 98.84, 96.94, 99.93]
}

df = pd.DataFrame(data)

# 创建改进百分比数据
improvements = []
for i in range(0, len(df), 2):
    original = df.iloc[i]
    improved = df.iloc[i+1]
    
    fpr_in_improve = (original['FPR@95%(In)'] - improved['FPR@95%(In)']) / original['FPR@95%(In)'] * 100
    fpr_out_improve = (original['FPR@95%(Out)'] - improved['FPR@95%(Out)']) / original['FPR@95%(Out)'] * 100
    det_err_improve = (original['Detection Error(%)'] - improved['Detection Error(%)']) / original['Detection Error(%)'] * 100
    auroc_improve = (improved['AUROC(%)'] - original['AUROC(%)']) / original['AUROC(%)'] * 100
    aupr_in_improve = (improved['AUPR(In)(%)'] - original['AUPR(In)(%)']) / original['AUPR(In)(%)'] * 100
    aupr_out_improve = (improved['AUPR(Out)(%)'] - original['AUPR(Out)(%)']) / original['AUPR(Out)(%)'] * 100
    
    improvements.append({
        'Task': original['Task'],
        'FPR@95%(In)': f"{fpr_in_improve:.1f}%",
        'FPR@95%(Out)': f"{fpr_out_improve:.1f}%",
        'Detection Error': f"{det_err_improve:.1f}%",
        'AUROC': f"{auroc_improve:.1f}%",
        'AUPR(In)': f"{aupr_in_improve:.1f}%",
        'AUPR(Out)': f"{aupr_out_improve:.1f}%"
    })

improvement_df = pd.DataFrame(improvements)

# 创建性能对比表格
fig, ax = plt.subplots(figsize=(16, 8))
ax.axis('off')

# 创建表格标题
title = ax.set_title('OOD Detection Performance: Mahalanobis vs Mahalanobis+', 
                    fontsize=18, pad=20, fontweight='bold')

# 计算表格行数和列数
nrows = len(df) + 1  # 数据行数 + 标题行
ncols = len(df.columns)

# 创建单元格颜色数组
cell_colors = []
# 标题行颜色
cell_colors.append(['#2E7D32'] * ncols)  # 深绿色

# 数据行颜色
for i in range(len(df)):
    if i % 2 == 0:  # Mahalanobis行
        row_colors = ['#e8f5e9'] * ncols
    else:  # Mahalanobis+行
        row_colors = ['#c8e6c9'] * ncols
    cell_colors.append(row_colors)

# 创建表格
table = ax.table(
    cellText=[df.columns.values.tolist()] + df.values.tolist(),
    cellLoc='center',
    loc='center',
    cellColours=cell_colors
)

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8)

# 设置标题行样式
for i in range(ncols):
    table[(0, i)].set_text_props(weight='bold', color='white')

# 设置Mahalanobis+行文字加粗
for row in [2, 4, 6]:  # 第2、4、6行是Mahalanobis+
    for col in range(ncols):
        table[(row, col)].set_text_props(weight='bold')

# 添加网格线
for i in range(nrows):
    for j in range(ncols):
        table[(i, j)].set_edgecolor('#bdbdbd')

# 添加性能改进标注
for i, task in enumerate(['Task 0', 'Task 1', 'Task 2']):
    # Mahalanobis行索引: 1, 3, 5
    # Mahalanobis+行索引: 2, 4, 6
    mah_row = i * 2 + 1
    plus_row = i * 2 + 2
    
    for col in range(2, ncols):  # 从第3列开始（跳过任务和方法列）
        original_val = df.iloc[i*2, col]
        improved_val = df.iloc[i*2+1, col]
        
        # 计算改进值
        if col in [2, 3, 4]:  # 误差类指标
            improvement = original_val - improved_val
            symbol = "↓"
            color = "#d32f2f"  # 红色
        else:  # 准确率类指标
            improvement = improved_val - original_val
            symbol = "↑"
            color = "#1976d2"  # 蓝色
            
        # 添加标注
        ax.text(0.5 + col*0.14, 0.68 - i*0.28, 
                f"{symbol}{improvement:.2f}", 
                transform=ax.transAxes,
                fontsize=10, weight='bold', color=color,
                ha='center', va='center')

plt.tight_layout()
plt.savefig('performance_comparison_table.png', bbox_inches='tight')
print("Performance comparison table saved as 'performance_comparison_table.png'")

# 创建改进百分比表格
fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.axis('off')

# 创建表格标题
title2 = ax2.set_title('Performance Improvement of Mahalanobis+ over Mahalanobis', 
                      fontsize=16, pad=20, fontweight='bold')

# 计算表格行数和列数
nrows_imp = len(improvement_df) + 1  # 数据行数 + 标题行
ncols_imp = len(improvement_df.columns)

# 创建单元格颜色数组
cell_colors_imp = []
# 标题行颜色
cell_colors_imp.append(['#1565C0'] * ncols_imp)  # 深蓝色

# 数据行颜色
for i in range(len(improvement_df)):
    cell_colors_imp.append(['#bbdefb'] * ncols_imp)  # 浅蓝色

# 创建表格
table2 = ax2.table(
    cellText=[improvement_df.columns.values.tolist()] + improvement_df.values.tolist(),
    cellLoc='center',
    loc='center',
    cellColours=cell_colors_imp
)

# 设置表格样式
table2.auto_set_font_size(False)
table2.set_fontsize(12)
table2.scale(1.2, 1.8)

# 设置标题行样式
for i in range(ncols_imp):
    table2[(0, i)].set_text_props(weight='bold', color='white')

# 设置数据行文字加粗
for row in range(1, nrows_imp):
    for col in range(ncols_imp):
        table2[(row, col)].set_text_props(weight='bold')

# 添加网格线
for i in range(nrows_imp):
    for j in range(ncols_imp):
        table2[(i, j)].set_edgecolor('#bdbdbd')

plt.tight_layout()
plt.savefig('improvement_percentage_table.png', bbox_inches='tight')
print("Improvement percentage table saved as 'improvement_percentage_table.png'")