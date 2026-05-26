import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 实验数据
methods = ['Baseline\n(原始AnomalyCLIP)', 'Fusion\n(从零训练)', 'Fusion\n(baseline微调)']
pixel_auroc = [95.5, 95.3, 95.4]
pixel_aupro = [37.0, 37.1, 37.2]
image_auroc = [91.4, 91.7, 91.8]
image_ap = [95.9, 95.7, 96.2]

colors = ['#4A90D9', '#F5A623', '#7B61FF']

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('BTAD 数据集实验结果对比', fontsize=16, fontweight='bold', y=0.98)

metrics = [
    ('Pixel AUROC (%)', pixel_auroc, 94.5, 96.5),
    ('Pixel AUPRO (%)', pixel_aupro, 35.5, 38.5),
    ('Image AUROC (%)', image_auroc, 90.5, 92.5),
    ('Image AP (%)', image_ap, 95.0, 97.0),
]

for ax, (title, values, y_min, y_max) in zip(axes.flat, metrics):
    bars = ax.bar(methods, values, color=colors, width=0.55, edgecolor='white', linewidth=1.5)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (y_max - y_min) * 0.02,
                f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 标注最佳
    best_idx = np.argmax(values)
    bars[best_idx].set_edgecolor('#E74C3C')
    bars[best_idx].set_linewidth(2.5)
    ax.text(bars[best_idx].get_x() + bars[best_idx].get_width() / 2,
            y_min + (y_max - y_min) * 0.05,
            'BEST', ha='center', va='bottom', fontsize=9, color='#E74C3C', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('./results/btad_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
print('图表已保存: ./results/btad_comparison.png')

# 额外：各指标提升幅度图
fig2, ax2 = plt.subplots(figsize=(8, 5))
metric_names = ['Pixel AUROC', 'Pixel AUPRO', 'Image AUROC', 'Image AP']
baseline_vals = [95.5, 37.0, 91.4, 95.9]
ft_vals = [95.4, 37.2, 91.8, 96.2]
diffs = [f - b for f, b in zip(ft_vals, baseline_vals)]

bar_colors = ['#E74C3C' if d < 0 else '#27AE60' for d in diffs]
bars = ax2.barh(metric_names, diffs, color=bar_colors, height=0.5, edgecolor='white', linewidth=1.5)
ax2.axvline(x=0, color='black', linewidth=0.8)
ax2.set_xlabel('相对 Baseline 的变化 (%)', fontsize=12)
ax2.set_title('Fusion (baseline微调) vs Baseline 提升幅度', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for bar, d in zip(bars, diffs):
    offset = 0.02 if d >= 0 else -0.02
    ax2.text(d + offset, bar.get_y() + bar.get_height() / 2,
             f'{d:+.1f}', ha='left' if d >= 0 else 'right',
             va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('./results/btad_improvement.png', dpi=200, bbox_inches='tight', facecolor='white')
print('图表已保存: ./results/btad_improvement.png')
