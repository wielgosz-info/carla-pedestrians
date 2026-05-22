#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
紧凑版生物启发导航系统架构图
- 长宽比4:3
- 字体更大
- 框更紧凑
- 配色简洁清爽
- 右侧融合清晰可见
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Ellipse, Wedge
import numpy as np
from PIL import Image
import os

# 设置样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 12  # 增大基础字体
plt.rcParams['axes.unicode_minus'] = False

# 创建画布 - 4:3比例
fig = plt.figure(figsize=(16, 12), facecolor='white')
ax = fig.add_subplot(111)
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# ==================== 简洁配色方案 ====================
COLOR_PERCEPTION = '#FFF8E7'      # 极淡米色
COLOR_DECISION = '#FFF4E6'        # 极淡橙色  
COLOR_BORDER = '#8B7355'          # 棕色边框
COLOR_ARROW = '#6B5742'           # 深棕色箭头
COLOR_TEXT = '#2C2416'            # 深棕色文字
COLOR_VESTIBULAR = '#FFA726'      # 亮橙色（简洁）
COLOR_VISUAL = '#66BB6A'          # 亮绿色（简洁）
COLOR_GRID = '#42A5F5'            # 亮蓝色
COLOR_ACTION = '#AB47BC'          # 亮紫色

# ==================== 左侧：Environment Input ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(SCRIPT_DIR, '..', '..', 'data', '01_NeuroSLAM_Datasets', 'Town01Data_IMU_Fusion')

# Town01
try:
    img1 = Image.open(os.path.join(data_path, '0500.png'))
    img1_small = img1.resize((150, 100))
    ax.imshow(img1_small, extent=[0.3, 2.3, 7.5, 9.5], aspect='auto', zorder=10)
    
    rect1 = Rectangle((0.3, 7.5), 2.0, 2.0, fill=False, edgecolor=COLOR_BORDER, 
                     linewidth=2.5, zorder=11)
    ax.add_patch(rect1)
    
    # 标签
    ax.add_patch(FancyBboxPatch((0.4, 9.6), 1.8, 0.4, boxstyle="round,pad=0.05", 
                               facecolor='#4CAF50', edgecolor='white', linewidth=2))
    ax.text(1.3, 9.8, 'Town01', ha='center', va='center', 
           fontsize=13, weight='bold', color='white')
    
except Exception as e:
    print(f"照片加载失败: {e}")

# Town10
try:
    img2 = Image.open(os.path.join(data_path, '2500.png'))
    img2_small = img2.resize((150, 100))
    ax.imshow(img2_small, extent=[0.3, 2.3, 4.5, 6.5], aspect='auto', zorder=10)
    
    rect2 = Rectangle((0.3, 4.5), 2.0, 2.0, fill=False, edgecolor=COLOR_BORDER, 
                     linewidth=2.5, zorder=11)
    ax.add_patch(rect2)
    
    ax.add_patch(FancyBboxPatch((0.4, 4.0), 1.8, 0.4, boxstyle="round,pad=0.05", 
                               facecolor='#2196F3', edgecolor='white', linewidth=2))
    ax.text(1.3, 4.2, 'Town10', ha='center', va='center', 
           fontsize=13, weight='bold', color='white')
    
except Exception as e:
    print(f"照片加载失败: {e}")

# 标题
ax.text(1.3, 10.5, 'Environment', ha='center', fontsize=15, weight='bold', color=COLOR_TEXT)

# ==================== 中间：Perception Network ====================
# 外框 - 更紧凑
perception_box = FancyBboxPatch((2.8, 3.5), 5.5, 7.5, 
                                boxstyle="round,pad=0.15", 
                                facecolor=COLOR_PERCEPTION, 
                                edgecolor=COLOR_BORDER, 
                                linewidth=2.5,
                                alpha=0.9,
                                zorder=1)
ax.add_patch(perception_box)

ax.text(5.55, 11.3, 'Perception Network', ha='center', fontsize=16, 
       weight='bold', color=COLOR_TEXT)

# 大脑轮廓（更淡）
brain = Ellipse((5.55, 7.3), 4.5, 5.5, facecolor='#FFE0B2', 
                edgecolor=COLOR_BORDER, linewidth=1.5, alpha=0.12, zorder=2)
ax.add_patch(brain)

# ===== Vestibular System（更紧凑）=====
vest_bg = FancyBboxPatch((3.2, 9.0), 4.7, 1.6, boxstyle="round,pad=0.08", 
                         facecolor=COLOR_VESTIBULAR, edgecolor='white', 
                         linewidth=2, alpha=0.35, zorder=3)
ax.add_patch(vest_bg)

# 半规管
for i in range(3):
    arc = Wedge((3.8 + i*0.4, 9.8), 0.25, i*60, i*60+180, 
               width=0.07, facecolor=COLOR_VESTIBULAR, 
               edgecolor='white', linewidth=1.5, alpha=0.8, zorder=4)
    ax.add_patch(arc)

ax.text(5.55, 10.4, 'Vestibular (IMU)', ha='center', 
       fontsize=13, weight='bold', color=COLOR_TEXT)

# IMU输入
for label, x in [('Accel.', 6.4), ('Ang.Vel.', 7.2)]:
    ax.add_patch(FancyBboxPatch((x, 9.5), 0.65, 0.5, boxstyle="round,pad=0.03", 
                               facecolor='white', edgecolor=COLOR_BORDER, linewidth=1.5))
    ax.text(x + 0.325, 9.75, label, ha='center', va='center', fontsize=10, weight='bold')

# ===== Visual Cortex =====
visual_bg = FancyBboxPatch((3.2, 4.0), 4.7, 4.5, boxstyle="round,pad=0.08", 
                           facecolor=COLOR_VISUAL, edgecolor='white', 
                           linewidth=2, alpha=0.25, zorder=3)
ax.add_patch(visual_bg)

ax.text(5.55, 8.7, 'Visual Cortex (RGB)', ha='center', 
       fontsize=13, weight='bold', color=COLOR_TEXT)

# Dorsal通路（更简洁）
ax.text(5.55, 7.5, 'Dorsal (Motion)', ha='center', 
       fontsize=11, weight='bold', color='#5E35B1')

dorsal_nodes = [
    ('V1', 3.8, 7.0),
    ('MT', 4.7, 7.0),
    ('Grid\nCells', 6.4, 7.0)
]

for i, (label, x, y) in enumerate(dorsal_nodes):
    circle = Circle((x, y), 0.35, facecolor='#7E57C2', 
                   edgecolor='white', linewidth=2, zorder=5, alpha=0.85)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', 
           fontsize=10, weight='bold', color='white')
    
    if i < len(dorsal_nodes) - 1:
        arrow = FancyArrowPatch((x + 0.4, y), (dorsal_nodes[i+1][1] - 0.4, y),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=2.5, color='#7E57C2', zorder=4)
        ax.add_patch(arrow)

# Ventral通路（更简洁）
ax.text(5.55, 5.8, 'Ventral (Recognition)', ha='center', 
       fontsize=11, weight='bold', color='#E65100')

ventral_nodes = [
    ('V1', 3.6, 5.2),
    ('V2', 4.4, 5.2),
    ('V4', 5.2, 5.2),
    ('IT', 6.0, 5.2),
    ('VT', 6.8, 5.2)
]

for i, (label, x, y) in enumerate(ventral_nodes):
    circle = Circle((x, y), 0.3, facecolor='#FF6F00', 
                   edgecolor='white', linewidth=2, zorder=5, alpha=0.85)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', 
           fontsize=9, weight='bold', color='white')
    
    if i < len(ventral_nodes) - 1:
        arrow = FancyArrowPatch((x + 0.35, y), (ventral_nodes[i+1][1] - 0.35, y),
                               arrowstyle='->', mutation_scale=12, 
                               linewidth=2, color='#FF6F00', zorder=4)
        ax.add_patch(arrow)

# 算法标注（更大字体）
ax.text(5.55, 4.5, 'CLAHE → Gaussian → Norm', ha='center', 
       fontsize=10, style='italic', color=COLOR_TEXT, alpha=0.7)

# ==================== 右侧：Decision Network（重新设计）====================
decision_box = FancyBboxPatch((8.6, 3.5), 4.8, 7.5, 
                              boxstyle="round,pad=0.15", 
                              facecolor=COLOR_DECISION, 
                              edgecolor=COLOR_BORDER, 
                              linewidth=2.5,
                              alpha=0.9,
                              zorder=1)
ax.add_patch(decision_box)

ax.text(11.0, 11.3, 'Decision Network', ha='center', fontsize=16, 
       weight='bold', color=COLOR_TEXT)

# === 姿态表征（并排）===
# 3D Grid Cells
grid_box = FancyBboxPatch((8.95, 9.2), 1.8, 1.5, 
                          boxstyle="round,pad=0.08", 
                          facecolor=COLOR_GRID, 
                          edgecolor='white', 
                          linewidth=2, zorder=4, alpha=0.85)
ax.add_patch(grid_box)
ax.text(9.85, 10.3, '3D Grid', ha='center', fontsize=12, weight='bold', color='white')
ax.text(9.85, 10.0, 'Cells', ha='center', fontsize=12, weight='bold', color='white')
ax.text(9.85, 9.6, '(x,y,z)', ha='center', fontsize=10, color='white')

# Head Direction
hdc_box = FancyBboxPatch((11.05, 9.2), 1.8, 1.5, 
                         boxstyle="round,pad=0.08", 
                         facecolor=COLOR_VESTIBULAR, 
                         edgecolor='white', 
                         linewidth=2, zorder=4, alpha=0.85)
ax.add_patch(hdc_box)
ax.text(11.95, 10.3, 'Head Dir.', ha='center', fontsize=12, weight='bold', color='white')
ax.text(11.95, 10.0, 'Cells', ha='center', fontsize=12, weight='bold', color='white')
ax.text(11.95, 9.6, '(yaw)', ha='center', fontsize=10, color='white')

# === 融合模块（更清晰）===
# 融合框
fusion_box = FancyBboxPatch((9.2, 7.0), 3.6, 1.7, 
                            boxstyle="round,pad=0.1", 
                            facecolor='#E1BEE7', 
                            edgecolor='white', 
                            linewidth=2, zorder=4, alpha=0.9)
ax.add_patch(fusion_box)
ax.text(11.0, 8.4, 'Sensory Fusion', ha='center', 
       fontsize=13, weight='bold', color=COLOR_TEXT)

# 融合节点（大字体）
fusion_nodes = [
    ('Spatial\nAttn', 9.7, 7.5),
    ('LSTM', 11.0, 7.5),
    ('MLP', 12.3, 7.5)
]

for label, x, y in fusion_nodes:
    circle = Circle((x, y), 0.4, facecolor='#9C27B0', 
                   edgecolor='white', linewidth=2, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', 
           fontsize=10, weight='bold', color='white')

# 连接
for i in range(len(fusion_nodes) - 1):
    x1, y1 = fusion_nodes[i][1], fusion_nodes[i][2]
    x2, y2 = fusion_nodes[i+1][1], fusion_nodes[i+1][2]
    arrow = FancyArrowPatch((x1 + 0.45, y1), (x2 - 0.45, y2),
                           arrowstyle='->', mutation_scale=18, 
                           linewidth=3, color='white', zorder=5)
    ax.add_patch(arrow)

# === Experience Map（简化）===
exp_box = FancyBboxPatch((9.2, 4.7), 3.6, 1.8, 
                         boxstyle="round,pad=0.1", 
                         facecolor='#81C784', 
                         edgecolor='white', 
                         linewidth=2, zorder=4, alpha=0.85)
ax.add_patch(exp_box)
ax.text(11.0, 6.2, 'Experience Map', ha='center', 
       fontsize=13, weight='bold', color='white')
ax.text(11.0, 5.7, 'Loop Closure', ha='center', 
       fontsize=10, color='white')
ax.text(11.0, 5.3, 'Map Relaxation', ha='center', 
       fontsize=10, color='white')

# === 输出（更清晰）===
output_box = FancyBboxPatch((9.4, 3.8), 3.2, 0.6, 
                            boxstyle="round,pad=0.05", 
                            facecolor='#FF7043', 
                            edgecolor='white', 
                            linewidth=2, zorder=4, alpha=0.9)
ax.add_patch(output_box)
ax.text(11.0, 4.1, 'Control: Steering + Acceleration', ha='center', 
       fontsize=11, weight='bold', color='white')

# ==================== 箭头连接（简化）====================
# Environment → Perception
arrow1 = FancyArrowPatch((2.4, 8.5), (2.75, 9.5),
                        arrowstyle='->', mutation_scale=25, 
                        linewidth=3, color=COLOR_ARROW)
arrow2 = FancyArrowPatch((2.4, 5.5), (2.75, 5.8),
                        arrowstyle='->', mutation_scale=25, 
                        linewidth=3, color=COLOR_ARROW)
ax.add_patch(arrow1)
ax.add_patch(arrow2)

# Perception → Decision（主箭头）
arrow3 = FancyArrowPatch((8.4, 9.8), (8.9, 9.8),
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=4, color=COLOR_ARROW)
arrow4 = FancyArrowPatch((8.4, 7.0), (9.15, 7.5),
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=4, color=COLOR_ARROW,
                        connectionstyle="arc3,rad=0.2")
ax.add_patch(arrow3)
ax.add_patch(arrow4)

# Decision内部连接
arrow5 = FancyArrowPatch((10.8, 9.15), (10.8, 8.8),
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color='white')
arrow6 = FancyArrowPatch((11.0, 6.95), (11.0, 6.55),
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color='white')
arrow7 = FancyArrowPatch((11.0, 4.65), (11.0, 4.45),
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color='white')
ax.add_patch(arrow5)
ax.add_patch(arrow6)
ax.add_patch(arrow7)

# ==================== 最右侧：输出图标 ====================
# 方向盘
steering = Circle((14.3, 8.0), 0.6, facecolor='#757575', 
                 edgecolor='#212121', linewidth=2.5, zorder=5)
ax.add_patch(steering)
ax.add_patch(Circle((14.3, 8.0), 0.2, facecolor='white', 
                   edgecolor='#212121', linewidth=2, zorder=6))
ax.plot([13.9, 14.7], [8.0, 8.0], '-', color='#212121', linewidth=3, zorder=6)
ax.text(14.3, 7.0, 'Steering', ha='center', fontsize=11, weight='bold')

# 油门
accel = Rectangle((13.9, 5.3), 0.8, 0.6, facecolor='#66BB6A', 
                 edgecolor='#2E7D32', linewidth=2.5, zorder=5)
ax.add_patch(accel)
ax.text(14.3, 5.6, 'Accel', ha='center', va='center',
       fontsize=10, weight='bold', color='white')
ax.text(14.3, 4.8, 'Acceleration', ha='center', fontsize=10, weight='bold')

# Decision → Output
arrow8 = FancyArrowPatch((12.7, 7.8), (13.65, 8.0),
                         arrowstyle='->', mutation_scale=25, 
                         linewidth=3.5, color=COLOR_ARROW)
arrow9 = FancyArrowPatch((12.7, 5.3), (13.85, 5.6),
                         arrowstyle='->', mutation_scale=25, 
                         linewidth=3.5, color=COLOR_ARROW)
ax.add_patch(arrow8)
ax.add_patch(arrow9)

plt.tight_layout()

# 保存
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'fig')
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUTPUT_DIR, 'bio_nav_architecture.pdf'),
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(os.path.join(OUTPUT_DIR, 'bio_nav_architecture.png'),
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

print("✅ 紧凑版架构图已生成！")
print("\n📊 优化内容：")
print("   1. ✅ 长宽比调整为4:3（16×12）")
print("   2. ✅ 字体增大（基础12pt，标题16pt）")
print("   3. ✅ 框尺寸缩小，布局更紧凑")
print("   4. ✅ 右侧融合模块清晰可见（Spatial Attn → LSTM → MLP）")
print("   5. ✅ 配色简化（清爽不花哨）")
print("   6. ✅ 箭头加粗，连接更清晰")

plt.close()
