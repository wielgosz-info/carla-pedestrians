#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简约优雅的生物启发导航系统架构图
参考用户提供的风格：简约、干净、使用真实照片和图标
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Wedge
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import numpy as np
from PIL import Image
import os

# 设置样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# 创建画布（简约大气）
fig = plt.figure(figsize=(20, 6), facecolor='white')
ax = fig.add_subplot(111)
ax.set_xlim(0, 20)
ax.set_ylim(0, 6)
ax.axis('off')

# ==================== 配色方案（参考用户图片的淡雅风格）====================
COLOR_BG_PERCEPTION = '#F5E6D3'  # 淡米色 - Perception
COLOR_BG_DECISION = '#FFE4CC'    # 淡橙色 - Decision
COLOR_BORDER = '#D4A574'         # 棕褐色边框
COLOR_ARROW = '#8B7355'          # 深棕色箭头
COLOR_TEXT_MAIN = '#2C2416'      # 深棕色文字
COLOR_VESTIBULAR = '#FFB88C'     # 柔和橙色
COLOR_VISUAL = '#98D8C8'         # 柔和绿色

# ==================== 左侧：Environment Input（使用真实照片）====================
# 加载CARLA真实照片
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(SCRIPT_DIR, '..', '..', 'data', '01_NeuroSLAM_Datasets', 'Town01Data_IMU_Fusion')

try:
    # 选择几张代表性照片
    img1 = Image.open(os.path.join(data_path, '0100.png'))  # 道路场景
    img2 = Image.open(os.path.join(data_path, '1000.png'))  # 不同场景
    
    # 缩小照片
    img1_small = img1.resize((120, 80))
    img2_small = img2.resize((120, 80))
    
    # 显示照片
    ax.imshow(img1_small, extent=[0.3, 1.8, 3.8, 5.2], aspect='auto', zorder=10)
    ax.imshow(img2_small, extent=[0.3, 1.8, 1.3, 2.7], aspect='auto', zorder=10)
    
    # 照片边框
    for y in [3.8, 1.3]:
        rect = Rectangle((0.3, y), 1.5, 1.4, fill=False, edgecolor=COLOR_BORDER, 
                        linewidth=2, zorder=11)
        ax.add_patch(rect)
    
    # 标签
    ax.text(1.05, 5.4, 'Environment Input', ha='center', fontsize=12, 
           weight='bold', color=COLOR_TEXT_MAIN)
    ax.text(1.05, 3.6, 't-1', ha='center', fontsize=9, 
           style='italic', color=COLOR_TEXT_MAIN)
    ax.text(1.05, 1.1, 't', ha='center', fontsize=9, 
           style='italic', color=COLOR_TEXT_MAIN)
    
    # 图标：车辆和行人（简化）
    car_icon = Rectangle((0.55, 3.3), 0.2, 0.15, facecolor='gray', 
                         edgecolor='black', linewidth=1)
    ped_icon = Circle((0.85, 3.375), 0.08, facecolor='gray', 
                     edgecolor='black', linewidth=1)
    ax.add_patch(car_icon)
    ax.add_patch(ped_icon)
    
except Exception as e:
    print(f"照片加载失败: {e}")
    # 备用方案：使用简单矩形
    ax.add_patch(Rectangle((0.3, 3.8), 1.5, 1.4, facecolor='lightblue', 
                          edgecolor=COLOR_BORDER, linewidth=2))
    ax.add_patch(Rectangle((0.3, 1.3), 1.5, 1.4, facecolor='lightblue', 
                          edgecolor=COLOR_BORDER, linewidth=2))
    ax.text(1.05, 5.4, 'Environment\nInput', ha='center', fontsize=11, weight='bold')

# ==================== 中间：Perception Network ====================
# 外框（大圆角矩形，淡色背景）
perception_box = FancyBboxPatch((2.2, 0.8), 6.5, 4.4, 
                                boxstyle="round,pad=0.15", 
                                facecolor=COLOR_BG_PERCEPTION, 
                                edgecolor=COLOR_BORDER, 
                                linewidth=2.5,
                                alpha=0.8,
                                zorder=1)
ax.add_patch(perception_box)

# 标题
ax.text(5.45, 5.5, 'Perception Network', ha='center', fontsize=13, 
       weight='bold', color=COLOR_TEXT_MAIN)

# ===== Vestibular System（上半部分）=====
# 背景框
vest_box = FancyBboxPatch((2.6, 3.5), 5.7, 1.5, 
                          boxstyle="round,pad=0.08", 
                          facecolor=COLOR_VESTIBULAR, 
                          edgecolor='white', 
                          linewidth=2,
                          alpha=0.4,
                          zorder=2)
ax.add_patch(vest_box)

# 半规管图标（三个简化的圆弧）
for i, angle in enumerate([30, 90, 150]):
    arc = Wedge((3.2 + i*0.4, 4.2), 0.25, angle, angle+180, 
               width=0.06, facecolor=COLOR_VESTIBULAR, 
               edgecolor='white', linewidth=1.5, alpha=0.8)
    ax.add_patch(arc)

ax.text(5.0, 4.7, 'Vestibular System', ha='center', fontsize=11, 
       weight='bold', color=COLOR_TEXT_MAIN)
ax.text(5.0, 4.4, '(Half-circular canals)', ha='center', fontsize=8, 
       style='italic', color=COLOR_TEXT_MAIN, alpha=0.7)

# IMU数据输入（小卡片）
imu_card1 = FancyBboxPatch((6.0, 4.0), 1.0, 0.35, 
                           boxstyle="round,pad=0.05", 
                           facecolor='white', 
                           edgecolor=COLOR_BORDER, 
                           linewidth=1.5)
imu_card2 = FancyBboxPatch((7.2, 4.0), 1.0, 0.35, 
                           boxstyle="round,pad=0.05", 
                           facecolor='white', 
                           edgecolor=COLOR_BORDER, 
                           linewidth=1.5)
ax.add_patch(imu_card1)
ax.add_patch(imu_card2)
ax.text(6.5, 4.175, 'Accel.', ha='center', fontsize=9)
ax.text(7.7, 4.175, 'Ang. Vel.', ha='center', fontsize=9)
ax.text(8.35, 4.175, '(IMU)', ha='left', fontsize=7, 
       color='gray', style='italic')

# ===== Visual Cortex（下半部分）=====
# 背景框
visual_box = FancyBboxPatch((2.6, 1.2), 5.7, 2.0, 
                            boxstyle="round,pad=0.08", 
                            facecolor=COLOR_VISUAL, 
                            edgecolor='white', 
                            linewidth=2,
                            alpha=0.3,
                            zorder=2)
ax.add_patch(visual_box)

ax.text(5.0, 3.0, 'Visual Cortex', ha='center', fontsize=11, 
       weight='bold', color=COLOR_TEXT_MAIN)
ax.text(7.5, 3.0, '(RGB)', ha='left', fontsize=7, 
       color='gray', style='italic')

# Dorsal Pathway（背侧通路）
dorsal_path = [
    (2.9, 2.4), (3.5, 2.4), (4.1, 2.4), (4.7, 2.4), (5.3, 2.4)
]
ax.plot([p[0] for p in dorsal_path], [p[1] for p in dorsal_path], 
       'o-', color='#7B68A6', markersize=10, linewidth=2, 
       markerfacecolor='white', markeredgewidth=2, markeredgecolor='#7B68A6')
ax.text(3.5, 2.7, 'Dorsal Pathway', ha='center', fontsize=9, 
       weight='bold', color='#7B68A6')
ax.text(2.9, 2.1, 'V1', ha='center', fontsize=7)
ax.text(3.5, 2.1, 'MT', ha='center', fontsize=7)
ax.text(4.7, 2.1, 'MST', ha='center', fontsize=7)

# Ventral Pathway（腹侧通路）
ventral_path = [
    (2.9, 1.6), (3.5, 1.6), (4.1, 1.6), (4.7, 1.6), (5.3, 1.6), (5.9, 1.6)
]
ax.plot([p[0] for p in ventral_path], [p[1] for p in ventral_path], 
       'o-', color='#E67E22', markersize=10, linewidth=2, 
       markerfacecolor='white', markeredgewidth=2, markeredgecolor='#E67E22')
ax.text(4.0, 1.3, 'Ventral Pathway', ha='center', fontsize=9, 
       weight='bold', color='#E67E22')
ax.text(2.9, 1.85, 'V1', ha='center', fontsize=7)
ax.text(3.5, 1.85, 'V2', ha='center', fontsize=7)
ax.text(4.1, 1.85, 'V4', ha='center', fontsize=7)
ax.text(5.3, 1.85, 'IT', ha='center', fontsize=7)

# ==================== 中间连接：箭头 ====================
# Environment → Perception
arrow1 = FancyArrowPatch((1.9, 4.5), (2.15, 4.3),
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2, color=COLOR_ARROW, 
                        connectionstyle="arc3,rad=0")
arrow2 = FancyArrowPatch((1.9, 2.0), (2.15, 2.2),
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2, color=COLOR_ARROW,
                        connectionstyle="arc3,rad=0")
ax.add_patch(arrow1)
ax.add_patch(arrow2)

# ==================== 右侧：Decision Network ====================
# 外框
decision_box = FancyBboxPatch((9.0, 0.8), 5.5, 4.4, 
                              boxstyle="round,pad=0.15", 
                              facecolor=COLOR_BG_DECISION, 
                              edgecolor=COLOR_BORDER, 
                              linewidth=2.5,
                              alpha=0.8,
                              zorder=1)
ax.add_patch(decision_box)

ax.text(11.75, 5.5, 'Decision Network', ha='center', fontsize=13, 
       weight='bold', color=COLOR_TEXT_MAIN)

# === Embedding层 ===
# Speed Embedding
speed_emb = FancyBboxPatch((9.5, 4.0), 1.3, 0.5, 
                           boxstyle="round,pad=0.05", 
                           facecolor='#FFE5B4', 
                           edgecolor=COLOR_BORDER, 
                           linewidth=1.5)
ax.add_patch(speed_emb)
ax.text(10.15, 4.25, 'Speed\nEmbed.', ha='center', va='center', fontsize=8)

# Navigation Embedding
nav_emb = FancyBboxPatch((9.5, 2.8), 1.3, 0.5, 
                         boxstyle="round,pad=0.05", 
                         facecolor='#FFD1A4', 
                         edgecolor=COLOR_BORDER, 
                         linewidth=1.5)
ax.add_patch(nav_emb)
ax.text(10.15, 3.05, 'Nav.\nEmbed.', ha='center', va='center', fontsize=8)

# Positional Embedding
pos_emb = FancyBboxPatch((9.5, 1.6), 1.3, 0.5, 
                         boxstyle="round,pad=0.05", 
                         facecolor='#FFBD8A', 
                         edgecolor=COLOR_BORDER, 
                         linewidth=1.5)
ax.add_patch(pos_emb)
ax.text(10.15, 1.85, 'Pos.\nEmbed.', ha='center', va='center', fontsize=8)

# === 中间处理层 ===
# DPC
dpc_box = FancyBboxPatch((11.2, 3.8), 0.8, 0.7, 
                         boxstyle="round,pad=0.08", 
                         facecolor='#F4A460', 
                         edgecolor='white', 
                         linewidth=2,
                         zorder=3)
ax.add_patch(dpc_box)
ax.text(11.6, 4.15, 'DPC', ha='center', va='center', fontsize=9, weight='bold')

# MPC
mpc_box = FancyBboxPatch((11.2, 2.6), 0.8, 0.7, 
                         boxstyle="round,pad=0.08", 
                         facecolor='#F4A460', 
                         edgecolor='white', 
                         linewidth=2,
                         zorder=3)
ax.add_patch(mpc_box)
ax.text(11.6, 2.95, 'MPC', ha='center', va='center', fontsize=9, weight='bold')

# VPC
vpc_box = FancyBboxPatch((11.2, 1.4), 0.8, 0.7, 
                         boxstyle="round,pad=0.08", 
                         facecolor='#F4A460', 
                         edgecolor='white', 
                         linewidth=2,
                         zorder=3)
ax.add_patch(vpc_box)
ax.text(11.6, 1.75, 'VPC', ha='center', va='center', fontsize=9, weight='bold')

# === Action Selection ===
action_box = FancyBboxPatch((12.5, 2.0), 1.2, 2.2, 
                            boxstyle="round,pad=0.1", 
                            facecolor='#FFDAB9', 
                            edgecolor=COLOR_BORDER, 
                            linewidth=2)
ax.add_patch(action_box)
ax.text(13.1, 3.9, 'Action', ha='center', fontsize=9, weight='bold')
ax.text(13.1, 3.6, 'Selection', ha='center', fontsize=9, weight='bold')

# CPC
ax.add_patch(Circle((13.1, 3.1), 0.25, facecolor='white', 
                   edgecolor=COLOR_BORDER, linewidth=1.5))
ax.text(13.1, 3.1, 'CPC', ha='center', va='center', fontsize=7, weight='bold')

# ==================== 最右侧：Decision Output ====================
output_box = FancyBboxPatch((15.0, 1.0), 2.0, 4.0, 
                            boxstyle="round,pad=0.15", 
                            facecolor='#E8EAF6', 
                            edgecolor=COLOR_BORDER, 
                            linewidth=2.5,
                            alpha=0.7)
ax.add_patch(output_box)
ax.text(16.0, 5.3, 'Decision Output', ha='center', fontsize=12, 
       weight='bold', color=COLOR_TEXT_MAIN)

# 方向盘图标（简化）
steering_wheel = Circle((16.0, 2.2), 0.5, facecolor='#BDBDBD', 
                       edgecolor='#424242', linewidth=2)
ax.add_patch(steering_wheel)
ax.add_patch(Circle((16.0, 2.2), 0.15, facecolor='white', 
                   edgecolor='#424242', linewidth=1.5))
ax.text(16.0, 1.5, 'steering\nangle', ha='center', fontsize=8)

# 油门图标（简化）
accel_icon = Rectangle((15.6, 3.8), 0.8, 0.4, facecolor='#81C784', 
                       edgecolor='#2E7D32', linewidth=2)
ax.add_patch(accel_icon)
ax.text(16.0, 4.4, 'acceleration', ha='center', fontsize=8)

# ==================== 内部连接箭头 ====================
# Perception → Decision
arrow3 = FancyArrowPatch((8.75, 4.2), (9.45, 4.2),
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2, color=COLOR_ARROW)
arrow4 = FancyArrowPatch((8.75, 3.0), (9.45, 3.0),
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2, color=COLOR_ARROW)
arrow5 = FancyArrowPatch((8.75, 1.8), (9.45, 1.8),
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2, color=COLOR_ARROW)
ax.add_patch(arrow3)
ax.add_patch(arrow4)
ax.add_patch(arrow5)

# Embedding → Processing
for y1, y2 in [(4.25, 4.15), (3.05, 2.95), (1.85, 1.75)]:
    arrow = FancyArrowPatch((10.85, y1), (11.15, y2),
                           arrowstyle='->', mutation_scale=15, 
                           linewidth=1.5, color=COLOR_ARROW)
    ax.add_patch(arrow)

# Processing → Action
for y in [4.15, 2.95, 1.75]:
    arrow = FancyArrowPatch((12.05, y), (12.45, 3.1),
                           arrowstyle='->', mutation_scale=15, 
                           linewidth=1.5, color=COLOR_ARROW)
    ax.add_patch(arrow)

# Action → Output
arrow6 = FancyArrowPatch((13.75, 4.0), (14.95, 4.0),
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color=COLOR_ARROW)
arrow7 = FancyArrowPatch((13.75, 2.2), (14.95, 2.2),
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2.5, color=COLOR_ARROW)
ax.add_patch(arrow6)
ax.add_patch(arrow7)

# ==================== 下方的Alignment（参考用户图2）====================
# 大脑图标区域（简化）
brain_region = FancyBboxPatch((7.5, 0.15), 4.5, 0.5, 
                              boxstyle="round,pad=0.08", 
                              facecolor='#FFE4E1', 
                              edgecolor=COLOR_BORDER, 
                              linewidth=1.5,
                              alpha=0.6)
ax.add_patch(brain_region)
ax.text(9.75, 0.4, 'Alignment', ha='center', fontsize=10, 
       weight='bold', color=COLOR_TEXT_MAIN, style='italic')

# Perception和Decision的连接到Alignment
ax.plot([5.45, 9.0], [0.8, 0.55], '--', color=COLOR_BORDER, 
       linewidth=1, alpha=0.5)
ax.plot([11.75, 10.5], [0.8, 0.55], '--', color=COLOR_BORDER, 
       linewidth=1, alpha=0.5)

plt.tight_layout()
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'fig')
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUTPUT_DIR, 'bio_nav_architecture.pdf'),
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(os.path.join(OUTPUT_DIR, 'bio_nav_architecture.png'),
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✅ 简约优雅的生物启发导航架构图已生成！")
print("   - 使用了真实CARLA场景照片")
print("   - 参考了用户提供的简约风格")
print("   - 添加了图标和清晰的视觉层次")
plt.close()
