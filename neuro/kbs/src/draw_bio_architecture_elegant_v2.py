#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版生物启发导航系统架构图
改进点：
1. 更现代的配色方案（蓝绿色调）
2. 渐变背景和阴影效果
3. 更清晰的层次结构
4. 更专业的图标设计
5. 更好的间距和对齐
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Ellipse, Wedge, PathPatch
from matplotlib.path import Path
import numpy as np
from PIL import Image
import os

# 设置样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 路径配置（跨平台）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 创建画布
fig = plt.figure(figsize=(24, 8), facecolor='#F8F9FA')
ax = fig.add_subplot(111)
ax.set_xlim(0, 24)
ax.set_ylim(0, 8)
ax.axis('off')

# ==================== 优化的配色方案 ====================
# 主色调：现代蓝绿色系
COLOR_PERCEPTION = '#E3F2FD'      # 浅蓝色 - Perception
COLOR_DECISION = '#E0F2F1'        # 浅青色 - Decision  
COLOR_BORDER = '#455A64'          # 深蓝灰色边框
COLOR_ARROW = '#546E7A'           # 蓝灰色箭头
COLOR_TEXT = '#263238'            # 深蓝灰色文字
COLOR_VESTIBULAR = '#64B5F6'      # 前庭系统 - 蓝色
COLOR_VISUAL_DORSAL = '#9575CD'   # 背侧通路 - 紫色
COLOR_VISUAL_VENTRAL = '#FF9800'  # 腹侧通路 - 橙色
COLOR_BRAIN = '#B0BEC5'           # 大脑轮廓 - 灰蓝色
COLOR_GRID = '#66BB6A'            # Grid Cells - 绿色
COLOR_HDC = '#FFA726'             # HDC - 橙色
COLOR_EXP = '#42A5F5'             # Experience - 蓝色

# 添加整体背景渐变
background = Rectangle((0, 0), 24, 8, facecolor='#F8F9FA', 
                       edgecolor='none', zorder=0)
ax.add_patch(background)

# ==================== 左侧：Environment Input（CARLA场景）====================
# 场景框架 - 更大更突出
scene_bg = FancyBboxPatch((0.3, 1.5), 3.0, 5.0, 
                          boxstyle="round,pad=0.15", 
                          facecolor='white', 
                          edgecolor=COLOR_BORDER, 
                          linewidth=3,
                          alpha=0.95,
                          zorder=8)
ax.add_patch(scene_bg)

# 标题
ax.text(1.8, 6.8, 'Environment', ha='center', fontsize=15, 
       weight='bold', color=COLOR_TEXT)
ax.text(1.8, 6.45, '(CARLA Simulator)', ha='center', fontsize=10, 
       style='italic', color=COLOR_TEXT, alpha=0.7)

data_path = os.path.join(SCRIPT_DIR, '..', '..', 'data', '01_NeuroSLAM_Datasets', 'Town01Data_IMU_Fusion')

# Town01 场景
try:
    img_town01 = Image.open(os.path.join(data_path, '0500.png'))
    img_town01_small = img_town01.resize((220, 150))
    ax.imshow(img_town01_small, extent=[0.5, 3.1, 4.3, 6.3], aspect='auto', zorder=10)
    
    # 场景标签
    scene_label = FancyBboxPatch((2.3, 5.95), 0.7, 0.3, 
                                boxstyle="round,pad=0.05", 
                                facecolor=COLOR_VISUAL_VENTRAL, 
                                edgecolor='white', 
                                linewidth=2, zorder=12)
    ax.add_patch(scene_label)
    ax.text(2.65, 6.1, 'Town01', ha='center', va='center', 
           fontsize=9, weight='bold', color='white')
except:
    pass

# 传感器数据标签
sensor_labels = [
    ('📷 RGB Camera', 4.5, '#42A5F5'),
    ('⚙️ IMU Sensor', 5.5, '#FFA726')
]
for label, y, color in sensor_labels:
    badge = FancyBboxPatch((0.5, y), 2.6, 0.5, 
                          boxstyle="round,pad=0.08", 
                          facecolor=color, 
                          edgecolor='white', 
                          linewidth=2.5, zorder=10, alpha=0.85)
    ax.add_patch(badge)
    ax.text(1.8, y + 0.25, label, ha='center', va='center', 
           fontsize=10, weight='bold', color='white')

# 车辆图标（更精致）
car_body = FancyBboxPatch((0.7, 2.2), 1.0, 0.5, 
                         boxstyle="round,pad=0.05", 
                         facecolor='#455A64', 
                         edgecolor='white', 
                         linewidth=2, zorder=12)
ax.add_patch(car_body)
# 车窗
window = Rectangle((1.0, 2.4), 0.4, 0.25, facecolor='#90CAF9', 
                  edgecolor='white', linewidth=1.5, zorder=13)
ax.add_patch(window)
ax.text(1.8, 2.0, '🚗 Ego Vehicle', ha='center', fontsize=9, 
       weight='bold', color=COLOR_TEXT)

# ==================== 中间：Perception Network（带大脑图标）====================
# 外框 - 加阴影效果
shadow = FancyBboxPatch((3.95, 0.55), 9.6, 6.5, 
                       boxstyle="round,pad=0.2", 
                       facecolor='#CFD8DC', 
                       edgecolor='none', 
                       linewidth=0,
                       alpha=0.3,
                       zorder=1)
ax.add_patch(shadow)

perception_box = FancyBboxPatch((3.8, 0.6), 9.6, 6.5, 
                                boxstyle="round,pad=0.2", 
                                facecolor=COLOR_PERCEPTION, 
                                edgecolor=COLOR_BORDER, 
                                linewidth=3.5,
                                alpha=0.9,
                                zorder=2)
ax.add_patch(perception_box)

# 标题
ax.text(8.6, 7.45, '🧠 Perception Network', ha='center', fontsize=16, 
       weight='bold', color=COLOR_TEXT)
ax.text(8.6, 7.05, '(Brain-Inspired Visual Processing)', ha='center', fontsize=11, 
       style='italic', color=COLOR_TEXT, alpha=0.7)

# ===== 大脑轮廓图标（更精致）=====
brain_outline = Ellipse((8.6, 4.0), 7.0, 4.5, 
                        facecolor=COLOR_BRAIN, 
                        edgecolor=COLOR_BORDER, 
                        linewidth=2.5, 
                        alpha=0.12,
                        zorder=3)
ax.add_patch(brain_outline)

# 大脑分区线
ax.plot([8.6, 8.6], [1.8, 6.2], '--', color=COLOR_BORDER, 
       linewidth=2, alpha=0.25, zorder=3)

# ===== 上半部分：Vestibular System（前庭系统）=====
vest_bg = FancyBboxPatch((4.2, 5.0), 8.8, 1.7, 
                         boxstyle="round,pad=0.12", 
                         facecolor=COLOR_VESTIBULAR, 
                         edgecolor='white', 
                         linewidth=3,
                         alpha=0.3,
                         zorder=4)
ax.add_patch(vest_bg)

# 标题
ax.text(8.6, 6.5, '⚙️ Vestibular System', ha='center', 
       fontsize=13, weight='bold', color=COLOR_TEXT)
ax.text(8.6, 6.15, '(IMU Processing & Motion Integration)', ha='center', 
       fontsize=9, style='italic', color=COLOR_TEXT, alpha=0.8)

# 半规管图标（更精致）
for i, (angle, alpha_val) in enumerate([(0, 0.9), (60, 0.75), (120, 0.6)]):
    arc = Wedge((4.9 + i*0.6, 5.7), 0.35, angle, angle+180, 
               width=0.1, facecolor=COLOR_VESTIBULAR, 
               edgecolor='white', linewidth=2.5, alpha=alpha_val, zorder=6)
    ax.add_patch(arc)

ax.text(6.0, 5.7, 'Semicircular\nCanals', ha='center', va='center',
       fontsize=8, weight='bold', color=COLOR_TEXT)

# IMU处理节点
imu_nodes = [
    ('Acceleration\n(a_x, a_y, a_z)', 9.5),
    ('Angular Velocity\n(ω_x, ω_y, ω_z)', 11.5)
]
for label, x in imu_nodes:
    card = FancyBboxPatch((x, 5.4), 1.4, 0.9, 
                          boxstyle="round,pad=0.08", 
                          facecolor='white', 
                          edgecolor=COLOR_VESTIBULAR, 
                          linewidth=2.5, zorder=6, alpha=0.95)
    ax.add_patch(card)
    ax.text(x + 0.7, 5.85, label, ha='center', va='center', 
           fontsize=8, weight='bold', color=COLOR_TEXT)

# ===== 下半部分：Visual Cortex（视觉皮层）=====
visual_bg = FancyBboxPatch((4.2, 1.0), 8.8, 3.6, 
                           boxstyle="round,pad=0.12", 
                           facecolor='#E8F5E9', 
                           edgecolor='white', 
                           linewidth=3,
                           alpha=0.35,
                           zorder=4)
ax.add_patch(visual_bg)

ax.text(8.6, 4.45, '👁️ Visual Cortex', ha='center', 
       fontsize=13, weight='bold', color=COLOR_TEXT)
ax.text(8.6, 4.1, '(Hierarchical RGB Processing)', ha='center', 
       fontsize=9, style='italic', color=COLOR_TEXT, alpha=0.8)

# === Dorsal Pathway（背侧通路 - 运动和空间）===
ax.text(8.6, 3.5, '━━━ Dorsal Pathway (Motion & Spatial) ━━━', ha='center', 
       fontsize=10, weight='bold', color=COLOR_VISUAL_DORSAL)

# 节点定义
dorsal_nodes = [
    ('LGN', 4.8, 2.9, 'CLAHE\n(0.02)'),
    ('V1', 6.0, 2.9, 'Gaussian\n(σ=0.5)'),
    ('MT/MST', 7.4, 2.9, 'Sobel\nEdges'),
    ('Path\nInt.', 9.2, 2.9, 'Motion\nVector'),
    ('3D GC', 11.0, 2.9, 'Grid\nCells')
]

for i, (label, x, y, desc) in enumerate(dorsal_nodes):
    # 节点圆圈（添加渐变效果）
    circle = Circle((x, y), 0.38, facecolor=COLOR_VISUAL_DORSAL, 
                   edgecolor='white', linewidth=3, zorder=7, alpha=0.85)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', 
           fontsize=8, weight='bold', color='white')
    
    # 描述文字
    desc_box = FancyBboxPatch((x - 0.35, y - 0.75), 0.7, 0.35, 
                             boxstyle="round,pad=0.05", 
                             facecolor='white', 
                             edgecolor=COLOR_VISUAL_DORSAL, 
                             linewidth=1.5, zorder=6, alpha=0.9)
    ax.add_patch(desc_box)
    ax.text(x, y - 0.575, desc, ha='center', va='center',
           fontsize=6.5, color=COLOR_TEXT, weight='bold')
    
    # 连接箭头（更粗更明显）
    if i < len(dorsal_nodes) - 1:
        arrow = FancyArrowPatch((x + 0.42, y), (dorsal_nodes[i+1][1] - 0.42, y),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=2.5, color=COLOR_VISUAL_DORSAL, zorder=6)
        ax.add_patch(arrow)

# === Ventral Pathway（腹侧通路 - 识别）===
ax.text(8.6, 2.0, '━━━ Ventral Pathway (Recognition) ━━━', ha='center', 
       fontsize=10, weight='bold', color=COLOR_VISUAL_VENTRAL)

# 节点定义
ventral_nodes = [
    ('LGN', 4.8, 1.4, 'CLAHE\n(0.02)'),
    ('V1', 5.9, 1.4, 'Gaussian\n(σ=0.5)'),
    ('V2/V4', 7.1, 1.4, 'Sobel\nFusion'),
    ('IT', 8.5, 1.4, 'Min-Max\nNorm'),
    ('VT', 10.1, 1.4, 'Template\n(299)'),
    ('Match', 11.7, 1.4, 'Cosine\n(0.09)')
]

for i, (label, x, y, desc) in enumerate(ventral_nodes):
    # 节点圆圈
    circle = Circle((x, y), 0.38, facecolor=COLOR_VISUAL_VENTRAL, 
                   edgecolor='white', linewidth=3, zorder=7, alpha=0.85)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', 
           fontsize=8, weight='bold', color='white')
    
    # 描述文字
    desc_box = FancyBboxPatch((x - 0.35, y + 0.4), 0.7, 0.35, 
                             boxstyle="round,pad=0.05", 
                             facecolor='white', 
                             edgecolor=COLOR_VISUAL_VENTRAL, 
                             linewidth=1.5, zorder=6, alpha=0.9)
    ax.add_patch(desc_box)
    ax.text(x, y + 0.575, desc, ha='center', va='center',
           fontsize=6.5, color=COLOR_TEXT, weight='bold')
    
    # 连接箭头
    if i < len(ventral_nodes) - 1:
        arrow = FancyArrowPatch((x + 0.42, y), (ventral_nodes[i+1][1] - 0.42, y),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=2.5, color=COLOR_VISUAL_VENTRAL, zorder=6)
        ax.add_patch(arrow)

# ==================== 环境到感知的连接 ====================
# Environment → Vestibular
arrow1 = FancyArrowPatch((3.3, 5.5), (4.15, 5.7),
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=4, color=COLOR_ARROW, zorder=12,
                        connectionstyle="arc3,rad=.2")
ax.add_patch(arrow1)
label1 = FancyBboxPatch((3.5, 5.2), 0.6, 0.35, 
                       boxstyle="round,pad=0.08", 
                       facecolor=COLOR_VESTIBULAR, 
                       edgecolor='white', linewidth=2, zorder=13)
ax.add_patch(label1)
ax.text(3.8, 5.375, 'IMU', ha='center', va='center',
       fontsize=9, weight='bold', color='white')

# Environment → Visual
arrow2 = FancyArrowPatch((3.3, 3.5), (4.15, 2.9),
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=4, color=COLOR_ARROW, zorder=12,
                        connectionstyle="arc3,rad=-.2")
ax.add_patch(arrow2)
label2 = FancyBboxPatch((3.5, 3.2), 0.6, 0.35, 
                       boxstyle="round,pad=0.08", 
                       facecolor=COLOR_VISUAL_VENTRAL, 
                       edgecolor='white', linewidth=2, zorder=13)
ax.add_patch(label2)
ax.text(3.8, 3.375, 'RGB', ha='center', va='center',
       fontsize=9, weight='bold', color='white')

# ==================== 右侧：Decision Network（建图功能）====================
# 外框阴影
shadow2 = FancyBboxPatch((13.95, 0.55), 7.5, 6.5, 
                        boxstyle="round,pad=0.2", 
                        facecolor='#CFD8DC', 
                        edgecolor='none', 
                        linewidth=0,
                        alpha=0.3,
                        zorder=1)
ax.add_patch(shadow2)

decision_box = FancyBboxPatch((13.8, 0.6), 7.5, 6.5, 
                              boxstyle="round,pad=0.2", 
                              facecolor=COLOR_DECISION, 
                              edgecolor=COLOR_BORDER, 
                              linewidth=3.5,
                              alpha=0.9,
                              zorder=2)
ax.add_patch(decision_box)

# 标题
ax.text(17.55, 7.45, '🗺️ Decision Network', ha='center', fontsize=16, 
       weight='bold', color=COLOR_TEXT)
ax.text(17.55, 7.05, '(Spatial Mapping & Loop Closure)', ha='center', fontsize=11, 
       style='italic', color=COLOR_TEXT, alpha=0.7)

# === Pose Representation（姿态表征）===
# 3D Grid Cells
grid_box = FancyBboxPatch((14.3, 5.2), 2.6, 1.5, 
                          boxstyle="round,pad=0.12", 
                          facecolor=COLOR_GRID, 
                          edgecolor='white', 
                          linewidth=3, zorder=5, alpha=0.85)
ax.add_patch(grid_box)
ax.text(15.6, 6.45, '3D Grid Cells', ha='center', va='top',
       fontsize=11, weight='bold', color='white')
ax.text(15.6, 6.1, 'FCC Lattice (36³)', ha='center', va='top',
       fontsize=8, color='white')
ax.text(15.6, 5.75, 'Position:', ha='center', fontsize=8, color='white', weight='bold')
ax.text(15.6, 5.45, '(x, y, z)', ha='center', fontsize=8, 
       color='white', style='italic')

# HDC
hdc_box = FancyBboxPatch((17.4, 5.2), 2.6, 1.5, 
                         boxstyle="round,pad=0.12", 
                         facecolor=COLOR_HDC, 
                         edgecolor='white', 
                         linewidth=3, zorder=5, alpha=0.85)
ax.add_patch(hdc_box)
ax.text(18.7, 6.45, 'Head Direction', ha='center', va='top',
       fontsize=11, weight='bold', color='white')
ax.text(18.7, 6.1, 'Ring Attractor (36²)', ha='center', va='top',
       fontsize=8, color='white')
ax.text(18.7, 5.75, 'Orientation:', ha='center', fontsize=8, color='white', weight='bold')
ax.text(18.7, 5.45, '(yaw, pitch)', ha='center', fontsize=8, 
       color='white', style='italic')

# Experience Map
exp_box = FancyBboxPatch((14.3, 3.0), 5.7, 1.8, 
                         boxstyle="round,pad=0.12", 
                         facecolor=COLOR_EXP, 
                         edgecolor='white', 
                         linewidth=3, zorder=5, alpha=0.85)
ax.add_patch(exp_box)
ax.text(17.15, 4.6, '🗺️ Experience Map', ha='center', va='top',
       fontsize=12, weight='bold', color='white')
ax.text(17.15, 4.25, '• Topological Mapping', ha='center', 
       fontsize=8, color='white')
ax.text(17.15, 3.95, '• Loop Closure Detection', ha='center', 
       fontsize=8, color='white')
ax.text(17.15, 3.65, '• Graph Relaxation', ha='center', 
       fontsize=8, color='white')
ax.text(17.15, 3.35, 'Nodes: ~426  |  RMSE: 126m', ha='center', 
       fontsize=7, color='white', style='italic')

# Action Selection
action_box1 = FancyBboxPatch((14.3, 1.2), 2.5, 1.4, 
                            boxstyle="round,pad=0.1", 
                            facecolor='#81C784', 
                            edgecolor='white', 
                            linewidth=2.5, zorder=5, alpha=0.85)
ax.add_patch(action_box1)
ax.text(15.55, 2.4, 'Steering', ha='center', va='top',
       fontsize=10, weight='bold', color='white')
ax.text(15.55, 2.1, 'Angle', ha='center', 
       fontsize=8, color='white')
ax.text(15.55, 1.75, '[-1, 1]', ha='center', 
       fontsize=8, color='white', style='italic')

action_box2 = FancyBboxPatch((17.5, 1.2), 2.5, 1.4, 
                            boxstyle="round,pad=0.1", 
                            facecolor='#4FC3F7', 
                            edgecolor='white', 
                            linewidth=2.5, zorder=5, alpha=0.85)
ax.add_patch(action_box2)
ax.text(18.75, 2.4, 'Throttle', ha='center', va='top',
       fontsize=10, weight='bold', color='white')
ax.text(18.75, 2.1, 'Control', ha='center', 
       fontsize=8, color='white')
ax.text(18.75, 1.75, '[0, 1]', ha='center', 
       fontsize=8, color='white', style='italic')

# Perception → Decision 连接
arrow_p2d1 = FancyArrowPatch((12.0, 2.9), (14.25, 5.5),
                            arrowstyle='->', mutation_scale=25, 
                            linewidth=3.5, color=COLOR_ARROW, zorder=12,
                            connectionstyle="arc3,rad=.3")
ax.add_patch(arrow_p2d1)

arrow_p2d2 = FancyArrowPatch((11.8, 1.4), (14.25, 3.8),
                            arrowstyle='->', mutation_scale=25, 
                            linewidth=3.5, color=COLOR_ARROW, zorder=12,
                            connectionstyle="arc3,rad=.2")
ax.add_patch(arrow_p2d2)

# 内部连接
arrow_gc2exp = FancyArrowPatch((15.6, 5.15), (15.6, 4.85),
                              arrowstyle='->', mutation_scale=18, 
                              linewidth=2.5, color='white', zorder=6)
arrow_hdc2exp = FancyArrowPatch((18.7, 5.15), (18.7, 4.85),
                               arrowstyle='->', mutation_scale=18, 
                               linewidth=2.5, color='white', zorder=6)
ax.add_patch(arrow_gc2exp)
ax.add_patch(arrow_hdc2exp)

arrow_exp2act1 = FancyArrowPatch((16.5, 2.95), (15.8, 2.65),
                                arrowstyle='->', mutation_scale=18, 
                                linewidth=2.5, color='white', zorder=6)
arrow_exp2act2 = FancyArrowPatch((17.8, 2.95), (18.5, 2.65),
                                arrowstyle='->', mutation_scale=18, 
                                linewidth=2.5, color='white', zorder=6)
ax.add_patch(arrow_exp2act1)
ax.add_patch(arrow_exp2act2)

# ==================== 最右侧：Output ====================
output_box = FancyBboxPatch((21.5, 2.0), 2.2, 4.0, 
                            boxstyle="round,pad=0.15", 
                            facecolor='#ECEFF1', 
                            edgecolor=COLOR_BORDER, 
                            linewidth=3.5,
                            alpha=0.95, zorder=5)
ax.add_patch(output_box)
ax.text(22.6, 6.15, '🎮 Control', ha='center', fontsize=13, 
       weight='bold', color=COLOR_TEXT)
ax.text(22.6, 5.8, 'Output', ha='center', fontsize=11, 
       weight='bold', color=COLOR_TEXT)

# 方向盘图标
steering = Circle((22.6, 4.7), 0.6, facecolor='#616161', 
                 edgecolor='white', linewidth=3, zorder=6)
ax.add_patch(steering)
ax.add_patch(Circle((22.6, 4.7), 0.2, facecolor='white', 
                   edgecolor='#424242', linewidth=2, zorder=7))
ax.plot([22.1, 23.1], [4.7, 4.7], '-', color='white', linewidth=3.5, zorder=7)
ax.text(22.6, 3.95, 'Steering', ha='center', fontsize=10, weight='bold')

# 油门图标
accel_icon = Rectangle((22.15, 2.8), 0.9, 0.6, facecolor='#66BB6A', 
                       edgecolor='white', linewidth=3, zorder=6)
ax.add_patch(accel_icon)
ax.text(22.6, 3.1, 'Throttle', ha='center', va='center',
       fontsize=9, weight='bold', color='white')
ax.text(22.6, 2.4, 'Acceleration', ha='center', fontsize=9, weight='bold')

# Decision → Output
arrow_d2o1 = FancyArrowPatch((16.2, 1.6), (21.45, 4.5),
                            arrowstyle='->', mutation_scale=30, 
                            linewidth=4, color=COLOR_ARROW, zorder=12,
                            connectionstyle="arc3,rad=-.2")
arrow_d2o2 = FancyArrowPatch((19.3, 1.6), (21.45, 3.2),
                            arrowstyle='->', mutation_scale=30, 
                            linewidth=4, color=COLOR_ARROW, zorder=12,
                            connectionstyle="arc3,rad=.2")
ax.add_patch(arrow_d2o1)
ax.add_patch(arrow_d2o2)

# ==================== 底部：Neural Anatomical Alignment ====================
alignment_box = FancyBboxPatch((7.5, 0.12), 9.0, 0.42, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#FFE0B2', 
                               edgecolor=COLOR_BORDER, 
                               linewidth=2.5,
                               alpha=0.75, zorder=3)
ax.add_patch(alignment_box)
ax.text(12.0, 0.33, '🧬 Neural Anatomical Alignment: Hippocampus → Grid Cells | Entorhinal Cortex → HDC', 
       ha='center', fontsize=10, weight='bold', color=COLOR_TEXT, style='italic')

# 连接虚线
ax.plot([8.6, 10.5], [0.6, 0.56], '--', color=COLOR_BORDER, 
       linewidth=1.5, alpha=0.35, zorder=3)
ax.plot([17.55, 13.5], [0.6, 0.56], '--', color=COLOR_BORDER, 
       linewidth=1.5, alpha=0.35, zorder=3)

plt.tight_layout()

# 保存
output_dir = os.path.join(SCRIPT_DIR, '..', 'fig')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f'{output_dir}/bio_nav_architecture.pdf', 
            dpi=300, bbox_inches='tight', facecolor='#F8F9FA', edgecolor='none')
plt.savefig(f'{output_dir}/bio_nav_architecture.png', 
            dpi=300, bbox_inches='tight', facecolor='#F8F9FA', edgecolor='none')

print("✅ 优化版生物启发导航架构图已生成！")
print("\n📊 优化改进：")
print("   1. ✨ 现代化配色方案（蓝绿色调）")
print("   2. 🎨 添加阴影和渐变效果")
print("   3. 📐 更清晰的层次结构")
print("   4. 🔗 更粗更明显的连接箭头")
print("   5. 📦 更精致的模块框架")
print("   6. 🏷️ 添加实际参数标注（VT=299, RMSE=126m等）")
print("   7. 💫 更大的画布和更好的间距")
print("\n输出文件：")
print(f"   • {output_dir}/bio_nav_architecture.pdf")
print(f"   • {output_dir}/bio_nav_architecture.png")

plt.close()
