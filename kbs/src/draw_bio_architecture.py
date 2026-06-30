#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'fig')
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(20, 6))
ax.set_xlim(0, 20)
ax.set_ylim(0, 6)
ax.axis('off')

# 左侧：CARLA场景
ax.add_patch(Rectangle((0.3, 1), 1.8, 4, facecolor='lightblue', edgecolor='black', linewidth=2))
ax.text(1.2, 4.2, 'CARLA\nTown01', ha='center', va='center', fontsize=11, weight='bold')
ax.text(1.2, 2.8, 'CARLA\nTown10', ha='center', va='center', fontsize=11, weight='bold')

# 中间上：前庭系统（IMU）
ax.add_patch(FancyBboxPatch((2.5, 3.5), 6, 2, boxstyle="round,pad=0.1", 
                            facecolor='#FFE6CC', edgecolor='#FF6B35', linewidth=2.5))
ax.text(5.5, 5.2, 'Vestibular System', ha='center', fontsize=12, weight='bold', color='#CC0000')
ax.text(5.5, 4.7, '(Half-circular canals)', ha='center', fontsize=9, style='italic')
ax.text(3.2, 4.2, 'Acceleration', fontsize=9, bbox=dict(boxstyle='round', facecolor='white'))
ax.text(5.5, 4.2, 'Angular\nVelocity', fontsize=9, bbox=dict(boxstyle='round', facecolor='white'))
ax.text(7.5, 4.2, '(IMU)', fontsize=8, color='gray', style='italic')

# 中间下：视觉系统（RGB）
ax.add_patch(FancyBboxPatch((2.5, 0.5), 6, 2.5, boxstyle="round,pad=0.1",
                            facecolor='#E6F3FF', edgecolor='#0066CC', linewidth=2.5))
ax.text(5.5, 2.7, 'Visual Cortex', ha='center', fontsize=12, weight='bold', color='#0066CC')
ax.text(7.8, 2.7, '(RGB)', fontsize=8, color='gray', style='italic')

# 腹侧通路
ax.text(3.2, 2.2, 'Ventral Stream', fontsize=9, weight='bold', color='#006600')
ax.text(3.2, 1.9, 'V1→V2→V4→IT', fontsize=8, bbox=dict(boxstyle='round', facecolor='#CCFFCC'))

# 背侧通路  
ax.text(3.2, 1.3, 'Dorsal Stream', fontsize=9, weight='bold', color='#660066')
ax.text(3.2, 1.0, 'V1→MT/MST', fontsize=8, bbox=dict(boxstyle='round', facecolor='#FFCCFF'))

# 融合
ax.add_patch(FancyBboxPatch((9, 2), 2, 2, boxstyle="round,pad=0.1",
                            facecolor='#FFFFCC', edgecolor='black', linewidth=2))
ax.text(10, 3, 'Sensory\nFusion', ha='center', va='center', fontsize=10, weight='bold')

# 右侧：注意力和决策
ax.add_patch(Rectangle((11.5, 1.5), 2, 3, facecolor='#FFE6E6', edgecolor='black', linewidth=2))
ax.text(12.5, 4.2, 'Spatial\nAttention', ha='center', fontsize=10, weight='bold')
ax.text(12.5, 3, 'V1', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow'))

# Dorsal/Ventral streams
ax.text(12.5, 2.3, 'Dorsal\nStream', ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='#FFCCFF'))
ax.text(12.5, 1.8, 'Ventral\nStream', ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='#CCFFCC'))

# LSTM-OT-MLP
ax.add_patch(Rectangle((14, 2), 1.5, 2, facecolor='yellow', edgecolor='black', linewidth=2))
ax.text(14.75, 3.5, 'LSTM', ha='center', fontsize=10, weight='bold')

ax.add_patch(Ellipse((16, 3), 0.8, 1.2, facecolor='lightgreen', edgecolor='black', linewidth=2))
ax.text(16, 3, 'OT', ha='center', va='center', fontsize=10, weight='bold')

ax.add_patch(Rectangle((17, 2), 1.5, 2, facecolor='orange', edgecolor='black', linewidth=2))
ax.text(17.75, 3, 'MLP', ha='center', va='center', fontsize=10, weight='bold')

# 箭头
ax.arrow(2.2, 3, 0.2, 1.5, head_width=0.15, head_length=0.1, fc='black', ec='black')
ax.arrow(2.2, 3, 0.2, -1, head_width=0.15, head_length=0.1, fc='black', ec='black')
ax.arrow(8.5, 4.5, 0.4, -1, head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2)
ax.arrow(8.5, 1.5, 0.4, 1, head_width=0.15, head_length=0.1, fc='blue', ec='blue', linewidth=2)
ax.arrow(11, 3, 0.4, 0, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=2)
ax.arrow(13.5, 3, 0.4, 0, head_width=0.12, head_length=0.08, fc='black', ec='black')
ax.arrow(15.5, 3, 0.3, 0, head_width=0.12, head_length=0.08, fc='black', ec='black')
ax.arrow(16.8, 3, 0.1, 0, head_width=0.12, head_length=0.08, fc='black', ec='black')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'bio_nav_architecture.pdf'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'bio_nav_architecture.png'),
            dpi=300, bbox_inches='tight')
print("✅ Bio-inspired architecture saved")
plt.close()
