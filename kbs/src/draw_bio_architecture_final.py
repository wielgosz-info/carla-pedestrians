#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终版生物启发导航系统架构图
参考用户提供的图片风格：
1. 左侧场景照片标注Town01和Town10，突出差异
2. 中间添加人脑图标
3. 标注实际的系统模型和功能
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Ellipse, Wedge
import numpy as np
from PIL import Image
import os

# 设置样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig = plt.figure(figsize=(22, 7), facecolor='white')
ax = fig.add_subplot(111)
ax.set_xlim(0, 22)
ax.set_ylim(0, 7)
ax.axis('off')

# ==================== 配色方案 ====================
COLOR_PERCEPTION = '#F8E8D8'      # 淡米色 - Perception
COLOR_DECISION = '#FFE8D6'        # 淡橙色 - Decision  
COLOR_BORDER = '#B8956A'          # 棕色边框
COLOR_ARROW = '#8B6F47'           # 深棕色箭头
COLOR_TEXT = '#2C2416'            # 深棕色文字
COLOR_VESTIBULAR = '#FFB88C'      # 前庭系统
COLOR_VISUAL_DORSAL = '#B39DDB'   # 背侧通路（紫色）
COLOR_VISUAL_VENTRAL = '#FFB74D'  # 腹侧通路（橙色）
COLOR_BRAIN = '#FFC1A1'           # 大脑轮廓

# ==================== 左侧：Environment Input（CARLA场景）====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(SCRIPT_DIR, '..', '..', 'data', '01_NeuroSLAM_Datasets', 'Town01Data_IMU_Fusion')

# Town01 场景
try:
    img_town01 = Image.open(os.path.join(data_path, '0500.png'))
    img_town01_small = img_town01.resize((180, 120))
    ax.imshow(img_town01_small, extent=[0.2, 2.2, 4.5, 6.2], aspect='auto', zorder=10)
    
    # 边框
    rect1 = Rectangle((0.2, 4.5), 2.0, 1.7, fill=False, edgecolor=COLOR_BORDER, 
                     linewidth=3, zorder=11)
    ax.add_patch(rect1)
    
    # 标签
    town01_label = FancyBboxPatch((0.3, 6.35), 1.8, 0.35, 
                                  boxstyle="round,pad=0.08", 
                                  facecolor='#4CAF50', 
                                  edgecolor='white', 
                                  linewidth=2)
    ax.add_patch(town01_label)
    ax.text(1.2, 6.525, 'CARLA Town01', ha='center', va='center', 
           fontsize=11, weight='bold', color='white')
    ax.text(1.2, 4.3, 'Urban Road', ha='center', fontsize=9, 
           style='italic', color=COLOR_TEXT)
    
except Exception as e:
    print(f"Town01照片加载失败: {e}")

# Town10 场景（选择不同的照片）
try:
    img_town10 = Image.open(os.path.join(data_path, '2500.png'))
    img_town10_small = img_town10.resize((180, 120))
    ax.imshow(img_town10_small, extent=[0.2, 2.2, 1.8, 3.5], aspect='auto', zorder=10)
    
    # 边框
    rect2 = Rectangle((0.2, 1.8), 2.0, 1.7, fill=False, edgecolor=COLOR_BORDER, 
                     linewidth=3, zorder=11)
    ax.add_patch(rect2)
    
    # 标签
    town10_label = FancyBboxPatch((0.3, 1.35), 1.8, 0.35, 
                                  boxstyle="round,pad=0.08", 
                                  facecolor='#2196F3', 
                                  edgecolor='white', 
                                  linewidth=2)
    ax.add_patch(town10_label)
    ax.text(1.2, 1.525, 'CARLA Town10', ha='center', va='center', 
           fontsize=11, weight='bold', color='white')
    ax.text(1.2, 1.6, 'Different Scene', ha='center', fontsize=9, 
           style='italic', color=COLOR_TEXT)
    
except Exception as e:
    print(f"Town10照片加载失败: {e}")

# 环境输入标题
ax.text(1.2, 6.85, 'Environment Input', ha='center', fontsize=13, 
       weight='bold', color=COLOR_TEXT)

# 车辆和行人图标
car_icon = Rectangle((0.5, 3.7), 0.3, 0.18, facecolor='#757575', 
                     edgecolor='black', linewidth=1.5, zorder=12)
ped_icon = Circle((1.0, 3.79), 0.1, facecolor='#BDBDBD', 
                 edgecolor='black', linewidth=1.5, zorder=12)
ax.add_patch(car_icon)
ax.add_patch(ped_icon)

# ==================== 中间：Perception Network（带大脑图标）====================
# 外框
perception_box = FancyBboxPatch((2.8, 0.6), 8.0, 5.8, 
                                boxstyle="round,pad=0.2", 
                                facecolor=COLOR_PERCEPTION, 
                                edgecolor=COLOR_BORDER, 
                                linewidth=3,
                                alpha=0.85,
                                zorder=1)
ax.add_patch(perception_box)

# 标题
ax.text(6.8, 6.75, 'Perception Network', ha='center', fontsize=14, 
       weight='bold', color=COLOR_TEXT)
ax.text(6.8, 6.4, '(Brain-Inspired Visual Processing)', ha='center', fontsize=9, 
       style='italic', color=COLOR_TEXT, alpha=0.7)

# ===== 大脑轮廓图标（中间背景）=====
# 简化的大脑轮廓（椭圆 + 波浪纹理）
brain_outline = Ellipse((6.8, 3.5), 5.5, 3.8, 
                        facecolor=COLOR_BRAIN, 
                        edgecolor=COLOR_BORDER, 
                        linewidth=2, 
                        alpha=0.15,
                        zorder=2)
ax.add_patch(brain_outline)

# 大脑分区线（参考图1）
ax.plot([6.8, 6.8], [1.7, 5.3], '--', color=COLOR_BORDER, 
       linewidth=1.5, alpha=0.3, zorder=2)

# ===== 上半部分：Vestibular System（前庭系统）=====
vest_bg = FancyBboxPatch((3.2, 4.5), 7.0, 1.5, 
                         boxstyle="round,pad=0.1", 
                         facecolor=COLOR_VESTIBULAR, 
                         edgecolor='white', 
                         linewidth=2,
                         alpha=0.45,
                         zorder=3)
ax.add_patch(vest_bg)

# 半规管图标（三个平面）
for i, (angle, color_shade) in enumerate([(0, 0.9), (60, 0.8), (120, 0.7)]):
    arc = Wedge((3.8 + i*0.5, 5.2), 0.3, angle, angle+180, 
               width=0.08, facecolor=COLOR_VESTIBULAR, 
               edgecolor='white', linewidth=2, alpha=color_shade, zorder=4)
    ax.add_patch(arc)

ax.text(6.8, 5.75, 'Vestibular System (IMU Processing)', ha='center', 
       fontsize=11, weight='bold', color=COLOR_TEXT)
ax.text(5.2, 5.2, 'Half-circular\nCanals', ha='center', va='center',
       fontsize=8, color=COLOR_TEXT, style='italic')

# IMU输入卡片
imu_inputs = [
    ('Acceleration', 8.0),
    ('Angular Vel.', 9.2)
]
for label, x in imu_inputs:
    card = FancyBboxPatch((x, 5.0), 1.05, 0.45, 
                          boxstyle="round,pad=0.05", 
                          facecolor='white', 
                          edgecolor=COLOR_BORDER, 
                          linewidth=1.8, zorder=4)
    ax.add_patch(card)
    ax.text(x + 0.525, 5.225, label, ha='center', va='center', 
           fontsize=8, weight='bold')

# ===== 下半部分：Visual Cortex（视觉皮层）=====
visual_bg = FancyBboxPatch((3.2, 1.0), 7.0, 3.0, 
                           boxstyle="round,pad=0.1", 
                           facecolor='#E8F5E9', 
                           edgecolor='white', 
                           linewidth=2,
                           alpha=0.4,
                           zorder=3)
ax.add_patch(visual_bg)

ax.text(6.8, 3.85, 'Visual Cortex (RGB Processing)', ha='center', 
       fontsize=11, weight='bold', color=COLOR_TEXT)

# === Dorsal Pathway（背侧通路 - 运动和空间）===
ax.text(6.8, 3.2, 'Dorsal Pathway (Motion & Spatial)', ha='center', 
       fontsize=9, weight='bold', color=COLOR_VISUAL_DORSAL)

# V1 → MT → MST节点
dorsal_nodes = [
    ('Retina/\nLGN', 3.7, 2.7, 'Grayscale\nCLAHE'),
    ('V1', 4.8, 2.7, 'Gaussian\nSmooth'),
    ('MT/MST', 6.0, 2.7, 'Motion\nFeatures'),
    ('Path\nIntegration', 7.8, 2.7, '3D Grid\nCells')
]

for i, (label, x, y, desc) in enumerate(dorsal_nodes):
    # 节点圆圈
    circle = Circle((x, y), 0.35, facecolor=COLOR_VISUAL_DORSAL, 
                   edgecolor='white', linewidth=2.5, zorder=5, alpha=0.8)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', 
           fontsize=7, weight='bold', color='white')
    
    # 描述文字
    ax.text(x, y - 0.55, desc, ha='center', va='top',
           fontsize=6, color=COLOR_TEXT, style='italic')
    
    # 连接箭头
    if i < len(dorsal_nodes) - 1:
        arrow = FancyArrowPatch((x + 0.4, y), (dorsal_nodes[i+1][1] - 0.4, y),
                               arrowstyle='->', mutation_scale=12, 
                               linewidth=2, color=COLOR_VISUAL_DORSAL, zorder=4)
        ax.add_patch(arrow)

# === Ventral Pathway（腹侧通路 - 识别）===
ax.text(6.8, 1.75, 'Ventral Pathway (Recognition)', ha='center', 
       fontsize=9, weight='bold', color=COLOR_VISUAL_VENTRAL)

# V1 → V2 → V4 → IT节点
ventral_nodes = [
    ('Retina/\nLGN', 3.7, 1.3, 'Grayscale\nCLAHE'),
    ('V1', 4.7, 1.3, 'Gaussian\nSmooth'),
    ('V2/V4', 5.9, 1.3, 'Downsample\n64×64'),
    ('IT', 7.1, 1.3, 'Row-mean\nNorm'),
    ('VT', 8.5, 1.3, 'Visual\nTemplate')
]

for i, (label, x, y, desc) in enumerate(ventral_nodes):
    # 节点圆圈
    circle = Circle((x, y), 0.35, facecolor=COLOR_VISUAL_VENTRAL, 
                   edgecolor='white', linewidth=2.5, zorder=5, alpha=0.8)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', 
           fontsize=7, weight='bold', color='white')
    
    # 描述文字（实际算法）
    ax.text(x, y + 0.55, desc, ha='center', va='bottom',
           fontsize=6, color=COLOR_TEXT, style='italic')
    
    # 连接箭头
    if i < len(ventral_nodes) - 1:
        arrow = FancyArrowPatch((x + 0.4, y), (ventral_nodes[i+1][1] - 0.4, y),
                               arrowstyle='->', mutation_scale=12, 
                               linewidth=2, color=COLOR_VISUAL_VENTRAL, zorder=4)
        ax.add_patch(arrow)

# ==================== 中间连接：环境到感知 ====================
# Environment → Vestibular
arrow1 = FancyArrowPatch((2.3, 5.5), (3.15, 5.2),
                        arrowstyle='->', mutation_scale=25, 
                        linewidth=3, color=COLOR_ARROW, zorder=12)
ax.add_patch(arrow1)
ax.text(2.7, 5.6, 'IMU', ha='center', fontsize=8, 
       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLOR_ARROW))

# Environment → Visual
arrow2 = FancyArrowPatch((2.3, 3.0), (3.15, 2.5),
                        arrowstyle='->', mutation_scale=25, 
                        linewidth=3, color=COLOR_ARROW, zorder=12)
ax.add_patch(arrow2)
ax.text(2.7, 2.6, 'RGB', ha='center', fontsize=8,
       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLOR_ARROW))

# ==================== 右侧：Decision Network（建图功能）====================
# 外框
decision_box = FancyBboxPatch((11.2, 0.6), 6.5, 5.8, 
                              boxstyle="round,pad=0.2", 
                              facecolor=COLOR_DECISION, 
                              edgecolor=COLOR_BORDER, 
                              linewidth=3,
                              alpha=0.85,
                              zorder=1)
ax.add_patch(decision_box)

# 标题
ax.text(14.45, 6.75, 'Decision Network', ha='center', fontsize=14, 
       weight='bold', color=COLOR_TEXT)
ax.text(14.45, 6.4, '(Spatial Mapping & Navigation)', ha='center', fontsize=9, 
       style='italic', color=COLOR_TEXT, alpha=0.7)

# === Pose Representation（姿态表征）===
# 3D Grid Cells
grid_box = FancyBboxPatch((11.7, 4.8), 2.2, 1.3, 
                          boxstyle="round,pad=0.1", 
                          facecolor='#81C784', 
                          edgecolor='white', 
                          linewidth=2.5, zorder=4)
ax.add_patch(grid_box)
ax.text(12.8, 5.85, '3D Grid Cells', ha='center', va='top',
       fontsize=10, weight='bold', color='white')
ax.text(12.8, 5.55, 'FCC Lattice', ha='center', va='top',
       fontsize=8, color='white', style='italic')
ax.text(12.8, 5.2, 'Position:', ha='center', fontsize=7, color='white')
ax.text(12.8, 4.95, '(x, y, z)', ha='center', fontsize=7, 
       weight='bold', color='white')

# Head Direction Cells
hdc_box = FancyBboxPatch((14.2, 4.8), 2.2, 1.3, 
                         boxstyle="round,pad=0.1", 
                         facecolor='#FFB74D', 
                         edgecolor='white', 
                         linewidth=2.5, zorder=4)
ax.add_patch(hdc_box)
ax.text(15.3, 5.85, 'Head Direction', ha='center', va='top',
       fontsize=10, weight='bold', color='white')
ax.text(15.3, 5.55, 'Multilayered', ha='center', va='top',
       fontsize=8, color='white', style='italic')
ax.text(15.3, 5.2, 'Orientation:', ha='center', fontsize=7, color='white')
ax.text(15.3, 4.95, 'yaw', ha='center', fontsize=7, 
       weight='bold', color='white')

# === Spatial Mapping（空间建图）===
# Experience Map
exp_map_box = FancyBboxPatch((11.7, 2.9), 4.7, 1.5, 
                             boxstyle="round,pad=0.12", 
                             facecolor='#64B5F6', 
                             edgecolor='white', 
                             linewidth=2.5, zorder=4)
ax.add_patch(exp_map_box)
ax.text(14.05, 4.2, 'Experience Map', ha='center', 
       fontsize=11, weight='bold', color='white')
ax.text(14.05, 3.9, '(Spatial Graph Construction)', ha='center', 
       fontsize=8, color='white', style='italic')

# 地图功能列表
map_features = [
    '• Loop Closure Detection',
    '• VT Association',
    '• Map Relaxation'
]
for i, feature in enumerate(map_features):
    ax.text(14.05, 3.55 - i*0.25, feature, ha='center', 
           fontsize=7, color='white')

# === Navigation Control（导航控制）===
# Action Selection
action_box = FancyBboxPatch((11.7, 1.1), 2.1, 1.4, 
                            boxstyle="round,pad=0.1", 
                            facecolor='#BA68C8', 
                            edgecolor='white', 
                            linewidth=2.5, zorder=4)
ax.add_patch(action_box)
ax.text(12.75, 2.3, 'Action', ha='center', 
       fontsize=9, weight='bold', color='white')
ax.text(12.75, 2.1, 'Selection', ha='center', 
       fontsize=9, weight='bold', color='white')
ax.text(12.75, 1.8, 'CPC', ha='center', 
       fontsize=8, color='white', style='italic')
ax.text(12.75, 1.5, 'Conjunctive', ha='center', 
       fontsize=7, color='white')
ax.text(12.75, 1.3, 'Pose Cells', ha='center', 
       fontsize=7, color='white')

# Trajectory Planning
traj_box = FancyBboxPatch((14.3, 1.1), 2.1, 1.4, 
                          boxstyle="round,pad=0.1", 
                          facecolor='#9575CD', 
                          edgecolor='white', 
                          linewidth=2.5, zorder=4)
ax.add_patch(traj_box)
ax.text(15.35, 2.3, 'Trajectory', ha='center', 
       fontsize=9, weight='bold', color='white')
ax.text(15.35, 2.1, 'Planning', ha='center', 
       fontsize=9, weight='bold', color='white')
ax.text(15.35, 1.8, 'MLP', ha='center', 
       fontsize=8, color='white', style='italic')
ax.text(15.35, 1.5, 'Multi-layer', ha='center', 
       fontsize=7, color='white')
ax.text(15.35, 1.3, 'Perceptron', ha='center', 
       fontsize=7, color='white')

# ==================== 内部连接箭头 ====================
# Perception → Decision（多条连接）
# Vestibular → Grid Cells
arrow3 = FancyArrowPatch((10.3, 5.2), (11.65, 5.45),
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color=COLOR_ARROW, zorder=12)
ax.add_patch(arrow3)

# Visual Dorsal → Grid Cells
arrow4 = FancyArrowPatch((8.3, 2.7), (11.65, 5.2),
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color=COLOR_ARROW, 
                        connectionstyle="arc3,rad=0.3", zorder=12)
ax.add_patch(arrow4)

# Visual Ventral → Experience Map
arrow5 = FancyArrowPatch((9.0, 1.3), (11.65, 3.65),
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color=COLOR_ARROW, 
                        connectionstyle="arc3,rad=0.2", zorder=12)
ax.add_patch(arrow5)

# Grid Cells + HDC → Experience Map
arrow6 = FancyArrowPatch((12.8, 4.75), (12.8, 4.45),
                        arrowstyle='->', mutation_scale=15, 
                        linewidth=2, color='white', zorder=5)
arrow7 = FancyArrowPatch((15.3, 4.75), (15.3, 4.45),
                        arrowstyle='->', mutation_scale=15, 
                        linewidth=2, color='white', zorder=5)
ax.add_patch(arrow6)
ax.add_patch(arrow7)

# Experience Map → Action
arrow8 = FancyArrowPatch((14.05, 2.85), (13.5, 2.55),
                        arrowstyle='->', mutation_scale=15, 
                        linewidth=2, color='white', zorder=5)
arrow9 = FancyArrowPatch((14.05, 2.85), (14.8, 2.55),
                        arrowstyle='->', mutation_scale=15, 
                        linewidth=2, color='white', zorder=5)
ax.add_patch(arrow8)
ax.add_patch(arrow9)

# ==================== 最右侧：Output ====================
output_box = FancyBboxPatch((18.0, 1.5), 2.5, 3.5, 
                            boxstyle="round,pad=0.15", 
                            facecolor='#E8EAF6', 
                            edgecolor=COLOR_BORDER, 
                            linewidth=3,
                            alpha=0.9, zorder=4)
ax.add_patch(output_box)
ax.text(19.25, 5.3, 'Control', ha='center', fontsize=12, 
       weight='bold', color=COLOR_TEXT)
ax.text(19.25, 5.05, 'Output', ha='center', fontsize=12, 
       weight='bold', color=COLOR_TEXT)

# 方向盘图标
steering = Circle((19.25, 4.0), 0.55, facecolor='#9E9E9E', 
                 edgecolor='#424242', linewidth=2.5, zorder=5)
ax.add_patch(steering)
ax.add_patch(Circle((19.25, 4.0), 0.18, facecolor='white', 
                   edgecolor='#424242', linewidth=2, zorder=6))
# 方向盘横杆
ax.plot([18.85, 19.65], [4.0, 4.0], '-', color='#424242', linewidth=3, zorder=6)
ax.text(19.25, 3.3, 'Steering', ha='center', fontsize=9, weight='bold')
ax.text(19.25, 3.05, 'Angle', ha='center', fontsize=8)

# 油门/刹车图标
accel_icon = Rectangle((18.85, 2.0), 0.8, 0.5, facecolor='#66BB6A', 
                       edgecolor='#2E7D32', linewidth=2.5, zorder=5)
ax.add_patch(accel_icon)
ax.text(19.25, 2.25, 'Accel.', ha='center', va='center',
       fontsize=8, weight='bold', color='white')
ax.text(19.25, 1.7, 'Acceleration', ha='center', fontsize=8)

# Decision → Output连接
arrow10 = FancyArrowPatch((13.9, 1.8), (17.95, 3.0),
                         arrowstyle='->', mutation_scale=25, 
                         linewidth=3, color=COLOR_ARROW, zorder=12)
arrow11 = FancyArrowPatch((15.5, 1.8), (17.95, 2.5),
                         arrowstyle='->', mutation_scale=25, 
                         linewidth=3, color=COLOR_ARROW, zorder=12)
ax.add_patch(arrow10)
ax.add_patch(arrow11)

# ==================== 底部：Neural Anatomical Alignment ====================
alignment_box = FancyBboxPatch((6.0, 0.08), 10.0, 0.38, 
                               boxstyle="round,pad=0.08", 
                               facecolor='#FFE4E1', 
                               edgecolor=COLOR_BORDER, 
                               linewidth=2,
                               alpha=0.7)
ax.add_patch(alignment_box)
ax.text(11.0, 0.27, 'Neural Anatomical Alignment', ha='center', 
       fontsize=10, weight='bold', color=COLOR_TEXT, style='italic')

# 连接到Alignment
ax.plot([6.8, 9.5], [0.6, 0.5], '--', color=COLOR_BORDER, 
       linewidth=1.2, alpha=0.4)
ax.plot([14.45, 12.5], [0.6, 0.5], '--', color=COLOR_BORDER, 
       linewidth=1.2, alpha=0.4)

plt.tight_layout()

# 保存
output_dir = os.path.join(SCRIPT_DIR, '..', 'fig')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'bio_nav_architecture.pdf'),
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(os.path.join(output_dir, 'bio_nav_architecture.png'),
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

print("✅ 最终版生物启发导航架构图已生成！")
print("\n📊 架构图特点：")
print("   1. ✅ 左侧：CARLA Town01和Town10场景照片，标注清晰")
print("   2. ✅ 中间：添加大脑轮廓图标，展示双通路处理")
print("   3. ✅ 腹侧通路：标注实际算法（CLAHE、Gaussian、Row-mean等）")
print("   4. ✅ 背侧通路：标注3D Grid Cells和运动处理")
print("   5. ✅ 右侧：标注实际建图功能（Experience Map、Loop Closure等）")
print("   6. ✅ 参考用户提供的图片风格：简约、专业、清晰")

plt.close()
