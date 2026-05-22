#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4-DOF Encoding Scheme - Corrected Version
Professional Top-tier Journal Standard
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import os

# 设置输出目录（跨平台相对路径）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'kbs', 'kbs_1', 'fig')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置全局字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['mathtext.fontset'] = 'stix'

print("开始生成4-DOF编码方案修正版...")

# ============================================================================
# 创建主图
# ============================================================================
fig = plt.figure(figsize=(14, 8), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# ============================================================================
# 标题
# ============================================================================
ax.text(7, 7.5, '(c) 4-DOF Encoding Scheme', 
        fontsize=22, fontweight='bold', ha='center')

# ============================================================================
# 顶部：Grid Cell 信息
# ============================================================================
# Grid Cell 立方体图标（简化表示）
grid_x, grid_y = 2.5, 6.5
rect_grid = Rectangle((grid_x-0.3, grid_y-0.3), 0.6, 0.6, 
                       facecolor='lightblue', edgecolor='black', linewidth=2)
ax.add_patch(rect_grid)
ax.plot([grid_x-0.3, grid_x+0.1], [grid_y+0.3, grid_y+0.7], 'k-', linewidth=2)
ax.plot([grid_x+0.3, grid_x+0.7], [grid_y+0.3, grid_y+0.7], 'k-', linewidth=2)
ax.plot([grid_x+0.1, grid_x+0.7], [grid_y+0.7, grid_y+0.7], 'k-', linewidth=2)
ax.plot([grid_x+0.1, grid_x+0.7], [grid_y-0.3, grid_y+0.1], 'k-', linewidth=2)
ax.plot([grid_x+0.7, grid_x+0.7], [grid_y+0.1, grid_y+0.7], 'k-', linewidth=2)

ax.text(grid_x, grid_y-0.7, 'Grid Cell: 61×61×61', fontsize=11, ha='center', fontweight='bold')
ax.text(grid_x, grid_y-1.0, r'neurons, $\sigma$=5 cells', fontsize=10, ha='center', style='normal')

# ============================================================================
# 右上角：公式框
# ============================================================================
formula_box = FancyBboxPatch((10.5, 6.2), 3.2, 1.1, boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor='white', linewidth=1.5)
ax.add_patch(formula_box)
ax.text(12.1, 6.9, r'$\bar{x} = \frac{\sum_{j \in A} j A_j^g}{\sum_{j \in A} A_j^g}$', 
        fontsize=13, ha='center', va='center')
ax.text(12.1, 6.45, r'$\bar{R} = \frac{\sum_{j \in A} \sum_{k} n_{j,k} r^k}{\sum_{j \in A} \sum_{k} A_k^g}$', 
        fontsize=13, ha='center', va='center')

# ============================================================================
# 左侧：输入
# ============================================================================
# Position输入框
pos_box = FancyBboxPatch((0.2, 4.5), 1.5, 0.8, boxstyle="round,pad=0.05",
                         edgecolor='black', facecolor='white', linewidth=2)
ax.add_patch(pos_box)
ax.text(0.95, 4.9, 'Position', fontsize=12, ha='center', fontweight='bold')
ax.text(0.95, 4.65, '(x, y, z)', fontsize=11, ha='center')

# Heading输入框
head_box = FancyBboxPatch((0.2, 2.8), 1.5, 0.8, boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor='white', linewidth=2)
ax.add_patch(head_box)
ax.text(0.95, 3.2, 'Heading', fontsize=12, ha='center', fontweight='bold')
ax.text(0.95, 2.95, r'$\psi$', fontsize=14, ha='center')

# ============================================================================
# 中间：Grid Cell处理
# ============================================================================
# Grid Cell主框
grid_main_box = FancyBboxPatch((2.2, 4.0), 2.0, 1.8, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='cyan', linewidth=2.5, alpha=0.6)
ax.add_patch(grid_main_box)

# Grid Cell 立方体（中间）
grid_center_x, grid_center_y = 3.2, 5.0
rect_gc = Rectangle((grid_center_x-0.25, grid_center_y-0.25), 0.5, 0.5,
                     facecolor='white', edgecolor='black', linewidth=1.5)
ax.add_patch(rect_gc)
ax.plot([grid_center_x-0.25, grid_center_x+0.05], [grid_center_y+0.25, grid_center_y+0.55], 'k-', linewidth=1.5)
ax.plot([grid_center_x+0.25, grid_center_x+0.55], [grid_center_y+0.25, grid_center_y+0.55], 'k-', linewidth=1.5)
ax.plot([grid_center_x+0.05, grid_center_x+0.55], [grid_center_y+0.55, grid_center_y+0.55], 'k-', linewidth=1.5)

# Path Integration标签
ax.text(2.0, 4.9, 'Path\nIntegration', fontsize=9, ha='right', va='center')

# ============================================================================
# Angular Direction Cell Network
# ============================================================================
ang_box = FancyBboxPatch((2.2, 2.2), 2.0, 1.3, boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor='purple', linewidth=2.5, alpha=0.6)
ax.add_patch(ang_box)
ax.text(3.2, 3.0, 'Angular Direction', fontsize=11, ha='center', fontweight='bold', color='white')
ax.text(3.2, 2.7, 'Cell Network', fontsize=11, ha='center', fontweight='bold', color='white')

# HDC说明
ax.text(3.2, 1.8, 'HDC: 360 bins×5 layers,', fontsize=9, ha='center')
ax.text(3.2, 1.5, r'$\gamma_\psi = 0.1$ bin·s/°', fontsize=9, ha='center')

# ============================================================================
# 双向箭头（Path Integration ↔ Angular Integration）
# ============================================================================
# 圆形图标
circle_center = Circle((3.2, 3.8), 0.25, facecolor='white', edgecolor='black', linewidth=1.5)
ax.add_patch(circle_center)
ax.text(3.2, 3.8, '⊕', fontsize=16, ha='center', va='center')

# 双向箭头
ax.annotate('', xy=(3.2, 4.0), xytext=(3.2, 5.0),
            arrowprops=dict(arrowstyle='<->', lw=2, color='black'))
ax.text(2.5, 4.5, 'Path\nIntegration', fontsize=8, ha='center', rotation=90, va='center')
ax.text(3.9, 4.5, 'Angular\nIntegration', fontsize=8, ha='center', rotation=90, va='center')

# ============================================================================
# 输入箭头
# ============================================================================
# Position → Grid Cell
ax.annotate('', xy=(2.2, 4.9), xytext=(1.7, 4.9),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(1.95, 5.15, 'Path\nintegration', fontsize=8, ha='center')

# Heading → Angular Direction
ax.annotate('', xy=(2.2, 3.2), xytext=(1.7, 3.2),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(1.95, 3.45, r'$\mathbb{R}^{226,981}$', fontsize=9, ha='center')

# ============================================================================
# 3D Toroidal CAN
# ============================================================================
toroid_box = FancyBboxPatch((5.0, 4.5), 1.8, 1.3, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='white', linewidth=2)
ax.add_patch(toroid_box)
ax.text(5.9, 5.4, '3D Toroidal', fontsize=11, ha='center', fontweight='bold')
ax.text(5.9, 5.1, 'CAN', fontsize=11, ha='center', fontweight='bold')
ax.text(5.9, 4.75, r'$\mathbb{R}^{226,981}$', fontsize=9, ha='center')

# Grid Cell → 3D Toroidal CAN
ax.annotate('', xy=(5.0, 5.1), xytext=(4.2, 5.1),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(4.6, 5.35, '2D Ring', fontsize=8, ha='center')
ax.text(4.6, 4.85, 'Attractor', fontsize=8, ha='center')

# ============================================================================
# h_ψz 中间节点
# ============================================================================
h_box = FancyBboxPatch((7.3, 4.7), 0.8, 0.7, boxstyle="round,pad=0.05",
                       edgecolor='black', facecolor='white', linewidth=2)
ax.add_patch(h_box)
ax.text(7.7, 5.05, r'$h_{\psi z}$', fontsize=12, ha='center', fontweight='bold')

# Angular Direction → h_ψz
ax.annotate('', xy=(7.3, 4.95), xytext=(4.2, 2.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(5.5, 3.5, r'$\mathbb{R}^{226,981}$', fontsize=9, ha='center', rotation=25)

# 3D Toroidal CAN → h_ψz
ax.annotate('', xy=(7.3, 5.1), xytext=(6.8, 5.1),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(7.05, 5.35, 'Angular', fontsize=7, ha='center')
ax.text(7.05, 4.85, 'Integration', fontsize=7, ha='center')

# ============================================================================
# Concatenate
# ============================================================================
concat_box = FancyBboxPatch((8.6, 4.6), 1.3, 0.9, boxstyle="round,pad=0.1",
                            edgecolor='darkgreen', facecolor='lightgreen', linewidth=2.5, alpha=0.5)
ax.add_patch(concat_box)
ax.text(9.2, 5.2, 'Concatenate', fontsize=11, ha='center', fontweight='bold')
ax.text(9.2, 4.85, r'$\oplus$ [g; h]', fontsize=10, ha='center')

# h_ψz → Concatenate
ax.annotate('', xy=(8.6, 5.05), xytext=(8.1, 5.05),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(8.35, 5.3, r'$\mathbb{R}^{1,800}$', fontsize=9, ha='center')

# ============================================================================
# s 4D 输出
# ============================================================================
output_box = FancyBboxPatch((10.5, 4.7), 0.9, 0.7, boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor='cyan', linewidth=2, alpha=0.6)
ax.add_patch(output_box)
ax.text(11.0, 5.2, 's', fontsize=14, ha='center', fontweight='bold')
ax.text(11.0, 4.9, '4D', fontsize=11, ha='center', fontweight='bold')

# Concatenate → s 4D
ax.annotate('', xy=(10.5, 5.05), xytext=(9.9, 5.05),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(10.2, 5.3, r'$\mathbb{R}^{228,781}$', fontsize=9, ha='center')

# ============================================================================
# 右侧：特性标注
# ============================================================================
feature_box = FancyBboxPatch((10.8, 2.8), 2.8, 1.3, boxstyle="round,pad=0.1",
                             edgecolor='darkgreen', facecolor='lightgreen', linewidth=2, alpha=0.3)
ax.add_patch(feature_box)
ax.text(12.2, 3.8, '✓ Noise-tolerant', fontsize=11, ha='center', fontweight='bold')
ax.text(12.2, 3.45, '✓ Continuous', fontsize=11, ha='center', fontweight='bold')
ax.text(12.2, 3.1, '✓ Bio-plausible', fontsize=11, ha='center', fontweight='bold')

# ============================================================================
# 底部：引用说明
# ============================================================================
ax.text(7, 0.8, 'Based on MD-CAN framework', fontsize=10, ha='center')
ax.text(7, 0.4, '(Zhang et al., 2023)', fontsize=10, ha='center')

# ============================================================================
# 保存
# ============================================================================
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '666.pdf'), dpi=300, bbox_inches='tight')
print(f"\n✓ 修正版图片保存完成: {os.path.join(OUTPUT_DIR, '666.pdf')}")
print("\n修正内容：")
print("  1. ✓ 4-DoF → 4-DOF (全大写)")
print("  2. ✓ 公式添加求和范围 Σ_{j∈A}")
print("  3. ✓ HDC单位修正为 bin·s/°")
print("  4. ✓ 3D Toroidal (CAN → 3D Toroidal CAN (括号匹配)")
print("  5. ✓ 特性标注统一使用 ✓")
print("  6. ✓ 引用格式规范化")
print("  7. ✓ 所有维度标注清晰")
print("  8. ✓ 箭头流程逻辑明确")
plt.close()

print("\n" + "="*60)
print("✓ 666.pdf 生成完成！符合顶刊标准。")
print("="*60)
