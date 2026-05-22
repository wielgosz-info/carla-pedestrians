#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional Figure 4 Generation for Top-tier Journal
3D Grid Cell Network with FCC Lattice Structure
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
import os

# 设置输出目录（跨平台相对路径）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'kbs', 'kbs_1', 'fig')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置全局字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'stix'

print("开始生成Figure 4的三个子图...")

# ============================================================================
# 子图4: 3D Activity Packet (改进版 - 更专业)
# ============================================================================
print("\n生成子图4: 3D Activity Packet...")

fig = plt.figure(figsize=(8, 7), facecolor='white')
ax = fig.add_subplot(111, projection='3d')

# 创建3D高斯分布
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
z = np.linspace(-3, 3, 50)
X, Y, Z = np.meshgrid(x, y, z)

# 高斯活动包
sigma = 0.8
G = np.exp(-((X**2 + Y**2 + Z**2) / (2 * sigma**2)))

# 绘制多层等值面（使用简单的scatter3D方法）
colors = ['#4ECDC4', '#44A5C2', '#3D7EA6', '#355C8A']
alphas = [0.2, 0.3, 0.4, 0.5]
levels = [0.3, 0.5, 0.7, 0.85]

# 采样点
for level, color, alpha in zip(levels, colors, alphas):
    mask = (G > level) & (G < level + 0.05)
    xs, ys, zs = X[mask], Y[mask], Z[mask]
    ax.scatter(xs, ys, zs, c=color, s=20, alpha=alpha, edgecolors='none')

# 绘制中心标记
ax.scatter([0], [0], [0], c='red', s=100, marker='*', 
           edgecolors='darkred', linewidths=2, label='Peak activity')

# 坐标轴设置
ax.set_xlabel('X (cells)', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel('Y (cells)', fontsize=14, fontweight='bold', labelpad=10)
ax.set_zlabel('Z (cells)', fontsize=14, fontweight='bold', labelpad=10)

# 标题
ax.set_title('(a) 3D Activity Packet\nGaussian Population Code', 
             fontsize=18, fontweight='bold', pad=20)

# 添加文字说明
ax.text2D(0.5, 0.02, r'$\sigma = 6$ cells | Distributed encoding of position $(x,y,z)$',
          transform=ax.transAxes, ha='center', fontsize=12,
          bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))

ax.view_init(elev=20, azim=45)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4.pdf'), dpi=300, bbox_inches='tight')
print(f"✓ 子图4保存完成: {os.path.join(OUTPUT_DIR, '4.pdf')}")
plt.close()

# ============================================================================
# 子图5: FCC Lattice Structure (专业重新设计)
# ============================================================================
print("\n生成子图5: FCC Lattice Structure...")

fig = plt.figure(figsize=(9, 10), facecolor='white')

# 主图区域（上方65%）
ax_main = fig.add_axes([0.1, 0.38, 0.85, 0.57], projection='3d')

# FCC晶格参数
a = 1.0

# 第一层 (z=0) - 深蓝色
layer1_x = np.array([0, a, 2*a, 0, a, 2*a, 0, a, 2*a])
layer1_y = np.array([0, 0, 0, a, a, a, 2*a, 2*a, 2*a])
layer1_z = np.zeros(9)

# 第二层 (z=a/2) - 橙色
layer2_x = np.array([a/2, 3*a/2, a/2, 3*a/2])
layer2_y = np.array([a/2, a/2, 3*a/2, 3*a/2])
layer2_z = np.ones(4) * a/2

# 第三层 (z=a) - 绿色
layer3_x = layer1_x.copy()
layer3_y = layer1_y.copy()
layer3_z = np.ones(9) * a

# 绘制连接线（灰色，细线）
for i in range(len(layer1_x)):
    for j in range(i+1, len(layer1_x)):
        dist = np.sqrt((layer1_x[i]-layer1_x[j])**2 + (layer1_y[i]-layer1_y[j])**2)
        if abs(dist - a) < 0.1:
            ax_main.plot([layer1_x[i], layer1_x[j]], 
                        [layer1_y[i], layer1_y[j]], 
                        [layer1_z[i], layer1_z[j]], 
                        'gray', linewidth=1.5, alpha=0.6)

# 绘制12邻居连接（红色虚线，突出显示）
center_idx = 4  # 中心点
for j in range(len(layer2_x)):
    dist = np.sqrt((layer1_x[center_idx]-layer2_x[j])**2 + 
                   (layer1_y[center_idx]-layer2_y[j])**2 + 
                   (layer1_z[center_idx]-layer2_z[j])**2)
    if dist < a*0.9:
        ax_main.plot([layer1_x[center_idx], layer2_x[j]], 
                    [layer1_y[center_idx], layer2_y[j]], 
                    [layer1_z[center_idx], layer2_z[j]], 
                    'r--', linewidth=2.5, alpha=0.8)

# 绘制原子（大球体）
ax_main.scatter(layer1_x, layer1_y, layer1_z, s=300, c='#2E5C8A', 
               edgecolors='black', linewidths=2, alpha=0.9, label='Layer 1 (z=0)')
ax_main.scatter(layer2_x, layer2_y, layer2_z, s=300, c='#E89C31', 
               edgecolors='black', linewidths=2, alpha=0.9, label='Layer 2 (z=a/2)')
ax_main.scatter(layer3_x, layer3_y, layer3_z, s=300, c='#4CAF50', 
               edgecolors='black', linewidths=2, alpha=0.9, label='Layer 3 (z=a)')

# 标注中心原子
ax_main.scatter([layer1_x[center_idx]], [layer1_y[center_idx]], [layer1_z[center_idx]], 
               s=150, c='red', marker='o', edgecolors='darkred', linewidths=3, zorder=10)

# 坐标轴
ax_main.set_xlabel('X', fontsize=14, fontweight='bold', labelpad=8)
ax_main.set_ylabel('Y', fontsize=14, fontweight='bold', labelpad=8)
ax_main.set_zlabel('Z', fontsize=14, fontweight='bold', labelpad=8)
ax_main.set_title('(b) FCC Lattice Structure\nOptimal 3D Spatial Encoding', 
                 fontsize=18, fontweight='bold', pad=15)

ax_main.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax_main.view_init(elev=20, azim=45)
ax_main.grid(True, alpha=0.3)
ax_main.set_xlim([-0.5, 2.5])
ax_main.set_ylim([-0.5, 2.5])
ax_main.set_zlim([-0.3, 1.3])

# 文字说明区域（下方33%）
ax_text = fig.add_axes([0.05, 0.02, 0.9, 0.33])
ax_text.axis('off')

# 添加说明文字（清晰分点）
text_content = [
    r'$\bf{Key\ Features:}$',
    r'$\bullet$ Hexagonal pattern in 2D projection mimics biological grid cells',
    r'$\bullet$ Three layers offset by $a/2$ for optimal 3D space coverage',
    r'$\bullet$ Each cell connects to 12 nearest neighbors (red dashed lines)',
    r'$\bullet$ Packing efficiency: 74% vs. 52% (simple cubic lattice)',
    r'$\bullet$ Enables dense, uniform spatial representation for SLAM'
]

y_pos = 0.95
for i, text in enumerate(text_content):
    fontsize = 14 if i == 0 else 12
    fontweight = 'bold' if i == 0 else 'normal'
    ax_text.text(0.02, y_pos, text, fontsize=fontsize, fontweight=fontweight,
                verticalalignment='top', transform=ax_text.transAxes)
    y_pos -= 0.18

plt.savefig(os.path.join(OUTPUT_DIR, '5.pdf'), dpi=300, bbox_inches='tight')
print(f"✓ 子图5保存完成: {os.path.join(OUTPUT_DIR, '5.pdf')}")
plt.close()

# ============================================================================
# 子图6: 4-DoF Encoding Scheme (学术流程图)
# ============================================================================
print("\n生成子图6: 4-DoF Encoding Scheme...")

fig = plt.figure(figsize=(12, 5), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis('off')

# 标题
ax.text(6, 4.7, '(c) 4-DoF Encoding Scheme', fontsize=20, fontweight='bold',
        ha='center', va='top')

# 定义框的样式
def draw_box(ax, x, y, w, h, text, color, textsize=12):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, fontsize=textsize, fontweight='bold',
            ha='center', va='center', multialignment='center')

# 定义箭头
def draw_arrow(ax, x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    if label:
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x, mid_y+0.15, label, fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Position输入
draw_box(ax, 0.5, 2.8, 1.2, 0.8, 'Position\n(x, y, z)', '#E8E8E8', 13)

# 3D Grid Cell Network
draw_box(ax, 2.2, 2.5, 1.6, 1.4, '3D Grid Cell\nNetwork\n(FCC)', '#B3D9FF', 12)

# g_xyz
draw_box(ax, 4.3, 2.9, 0.9, 0.6, r'$\mathbf{g}_{xyz}$', '#E8E8E8', 14)

# Heading输入
draw_box(ax, 0.5, 1.2, 1.2, 0.8, 'Heading\nψ', '#E8E8E8', 13)

# Head Direction Cell
draw_box(ax, 2.2, 0.9, 1.6, 1.4, 'Head Direction\nCell Network', '#FFE4B3', 12)

# h_psi
draw_box(ax, 4.3, 1.3, 0.9, 0.6, r'$\mathbf{h}_{\psi}$', '#E8E8E8', 14)

# Concatenation
draw_box(ax, 6.8, 1.8, 1.2, 1.0, 'Concatenate\n[g; h]', '#C8F7C5', 12)

# 输出
draw_box(ax, 8.8, 2.0, 0.9, 0.6, r'$\mathbf{s}_{4D}$', '#A3D5FF', 15)

# 绘制箭头
draw_arrow(ax, 1.7, 3.2, 2.2, 3.2)
draw_arrow(ax, 3.8, 3.2, 4.3, 3.2)
draw_arrow(ax, 1.7, 1.6, 2.2, 1.6)
draw_arrow(ax, 3.8, 1.6, 4.3, 1.6)

# 合并箭头
ax.plot([5.2, 6.5, 6.8], [3.2, 2.6, 2.3], 'k-', lw=2)
ax.plot([6.5, 6.8], [2.6, 2.3], 'k-', lw=2, marker='>', markersize=8)
ax.plot([5.2, 6.5, 6.8], [1.6, 2.2, 2.3], 'k-', lw=2)
ax.plot([6.5, 6.8], [2.2, 2.3], 'k-', lw=2, marker='>', markersize=8)

draw_arrow(ax, 8.0, 2.3, 8.8, 2.3)

# 添加维度标注
ax.text(5.2, 3.5, 'N×1', fontsize=10, ha='center', style='italic', color='blue')
ax.text(5.2, 1.0, 'M×1', fontsize=10, ha='center', style='italic', color='blue')
ax.text(9.3, 2.7, '(N+M)×1', fontsize=10, ha='center', style='italic', color='blue')

# 添加说明
ax.text(6, 0.3, 'Unified 4-DoF spatial representation for place recognition and loop closure',
        fontsize=11, ha='center', style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))

plt.savefig(os.path.join(OUTPUT_DIR, '6.pdf'), dpi=300, bbox_inches='tight')
print(f"✓ 子图6保存完成: {os.path.join(OUTPUT_DIR, '6.pdf')}")
plt.close()

print("\n" + "="*60)
print("✓ 所有子图生成完成！")
print(f"输出目录: {OUTPUT_DIR}")
print("文件列表:")
print("  - 4.pdf (3D Activity Packet - 改进版)")
print("  - 5.pdf (FCC Lattice Structure - 专业重新设计)")
print("  - 6.pdf (4-DoF Encoding Scheme - 学术流程图)")
print("="*60)
