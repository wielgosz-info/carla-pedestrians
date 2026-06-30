## 📊 Paper Figures 2-7: Drawing Method & Reproduction Guide
This document details the drawing logic, corresponding scripts, data sources, and reproduction steps for all core figures in the paper, fully aligned with the repository structure.

---

### 📌 Figure 2: Complementary IMU-Visual Fusion Architecture
#### 1. Figure Core Content
- **Left Subplot (Frequency Response)**
  - Blue curve: Low-pass filter amplitude-frequency response of the visual system (low-frequency retention, high-frequency attenuation)
  - Orange curve: High-pass filter amplitude-frequency response of the IMU (vestibular system) (high-frequency retention, low-frequency attenuation)
  - Cutoff frequency $\omega_c=10\ \text{rad/s}$, amplitude $R=0.707$ (-3dB point) marked as the core parameter of the complementary filter
- **Middle Subplot (Time-Domain Fusion)**
  - Black dashed line: Ground Truth (GT) velocity
  - Blue curve: Raw visual velocity (stable at low frequencies, slow response)
  - Yellow curve: Raw IMU velocity (sensitive at high frequencies, with drift)
  - Green curve: Fused velocity (combines advantages of both, drift-free, fast response)
  - Error metrics before/after fusion ($X\ 3.52 \to Y\ 1.74493$) marked to verify fusion effect
- **Right Subplot (Biological Analogy)**
  - Visual system (eyes) corresponds to low-pass filtering, vestibular system (ears) corresponds to high-pass filtering
  - CNS (vestibular nuclei + cerebellum) completes fusion to output robust self-motion estimation
  - Fusion formula $v^f = H_{LP} \cdot v^{vis} + H_{HP} \cdot v^{imu}$ and core advantages (drift-free, fast response) marked

#### 2. Corresponding Repository Files
- Drawing Script: `neuro/kbs/fig/generate_imu_visual_fusion.py`
- Generated PDF: `neuro/kbs/fig/imu_visual_complementary_fusion.pdf`
- Data Source: Synchronized visual/IMU sensor data collected from CARLA simulation environment

#### 3. Reproduction Method
- Dependencies: `scipy.signal` (filter design), `matplotlib` + `numpy` (plotting), `matplotlib.patches` (block diagram drawing)
- Run Command:
  ```bash
  python generate_imu_visual_fusion.py
  Output: PDF file imu_visual_complementary_fusion.pdf
￼
📌 Figure 3: 3D Grid Cell Network & 4-DoF Encoding
1. Figure Core Content
• Left Subplot (3D Grid Cell Activity)
◦ 3D layered activity distribution of 
 grid cells, with color scales distinguishing activity intensity at different depths, corresponding to distributed encoding of spatial positions
◦ Grid cell scale (
) and head direction cell association marked
• Middle Subplot (Simple Cubic vs FCC Lattice)
◦ Comparison of simple cubic and FCC (face-centered cubic) lattice structures, with FCC as the optimal spatial topology for grid cells
◦ Lattice coordinate transformation relation 
 marked
• Right Subplot (4-DoF Pose Encoding)
◦ 3D grid cells encode 3D position, head direction cells (Hdc) encode 1D heading, concatenated as 
◦ Core decoding formula marked, final output of 4-DoF (3D position + 1D heading) pose representation
2. Corresponding Repository Files
• Drawing Script: neuro/kbs/fig/generate_fig_3d_grid_cell.py
• Generated PDF: neuro/kbs/fig/3d_grid_cell_fcc_lattice.pdf
• Data Source: Grid cell activity field matrix output by the model, FCC lattice coordinate data
3. Reproduction Method
• Dependencies: matplotlib.mplot3d (3D visualization), numpy (FCC lattice coordinate calculation), matplotlib.patches (block diagram drawing)
• Run Command:
bash
￼
￼
运行
￼
￼
￼
￼
python generate_fig_3d_grid_cell.py
￼
• Output: PDF file 3d_grid_cell_fcc_lattice.pdf
￼
📌 Figure 4: Town01/MH03 Representative Performance
1. Figure Core Content
• Town01 Performance Metrics (Top-Left)
◦ Bar chart showing core metrics: RMSE (
￼
), Drift% (
￼
), RPE (
￼
), VT (
￼
), Loops (
￼
)
• Town01 Error Evolution (Top-Right)
◦ Frame-by-frame ATE error curve, comparing NeuroLocMap, EKF Fusion, Visual Odometry, demonstrating NeuroLocMap's low-error advantage
• MH03 Performance Metrics (Bottom-Left)
◦ Bar chart showing core metrics: RMSE (
￼
), Drift% (
￼
), RPE (
￼
), VT (
￼
), Loops (
￼
)
• MH03 Error Evolution (Bottom-Right)
◦ Frame-by-frame ATE error curve, comparing three methods, verifying NeuroLocMap's robustness in small scenes
2. Corresponding Repository Files
• Drawing Script: neuro/kbs/fig/generate_fig4_professional.py
• Generated PDF: neuro/kbs/fig/representative_performance.pdf
• Data Source: SLAM positioning results and frame-by-frame error data from Town01 and MH03 datasets
3. Reproduction Method
• Dependencies: matplotlib (plotting), pandas (experimental metrics and error data processing)
• Run Command:
bash
￼
￼
运行
￼
￼
￼
￼
python generate_fig4_professional.py
￼
• Output: PDF file representative_performance.pdf
￼
📌 Figure 5: Ablation Study & Visual Template Growth
1. Figure Core Content
• Left Subplot (Ablation Study RMSE Comparison)
◦ Bar chart comparing 5 model configurations: Full, w/o IMU, w/o Exp Map, w/o Transformer, w/o Dual-stream
◦ RMSE values for each configuration marked (Full: 
￼
, w/o IMU: 
￼
), with error bars to reflect result stability
• Right Subplot (Visual Template Growth Curve)
◦ Visual template count growth curves for 6 datasets (Town01/Town02/Town10/MH01/MH03/KITTI07) with frame index
◦ Comparison with RatSLAM baseline (~5 templates), Town10's 195 templates marked to reflect model's template accumulation capability
2. Corresponding Repository Files
• Drawing Scripts: neuro/kbs/fig/generate_ablation_unified.py, neuro/kbs/fig/generate_vt_growth.py
• Generated PDFs: neuro/kbs/fig/ablation_unified.pdf, neuro/kbs/fig/vt_growth_all_datasets.pdf
• Data Source: Ablation experiment results, visual template count time-series data from each dataset
3. Reproduction Method
• Dependencies: matplotlib + seaborn (plotting), numpy (template count and ablation data processing)
• Run Commands:
bash
￼
￼
运行
￼
￼
￼
￼
python generate_ablation_unified.py
python generate_vt_growth.py
￼
• Output: Corresponding PDF files ablation_unified.pdf, vt_growth_all_datasets.pdf
￼
📌 Figure 6: 6-Dataset Performance Summary
1. Figure Core Content
• (a) RMSE Comparison Bar Chart
◦ RMSE comparison of NeuroLocMap/EKF/VO for 6 datasets (Town01/Town02/Town10/KITTI07/MH01/MH03)
• (b) Improvement Bubble Chart
◦ Improvement rate relative to EKF, bubble size proportional to trajectory length, improvement percentage for each dataset marked (e.g., Town10: 
)
• (c) Success Rate Pie Chart
◦ Localization success rate 
￼
 (5/6 datasets successful), 
￼
 failure (corresponding to MH01)
2. Corresponding Repository Files
• Drawing Script: neuro/kbs/fig/generate_performance_summary.py
• Generated PDF: neuro/kbs/fig/performance_summary.pdf
• Data Source: SLAM positioning results and RMSE statistics from 6 datasets
3. Reproduction Method
• Dependencies: matplotlib (plotting), pandas (improvement rate calculation and summary data processing)
• Run Command:
bash
￼
￼
运行
￼
￼
￼
￼
python generate_performance_summary.py
￼
• Output: PDF file performance_summary.pdf
￼
📌 Figure 7: KITTI-07 Input-Output Visualization
1. Figure Core Content
• Top-Left: RGB Images
◦ 4 key frames from KITTI-07 dataset (Frame 77/110/330/551), showing visual input
• Bottom-Left: IMU Sensor Data
◦ Time-series curves of IMU acceleration (Accel) and angular velocity (GyroZ), with key frame positions marked
• Top-Right: 3D Trajectory Comparison
◦ 3D comparison of Ground Truth and NeuroLocMap predicted trajectory, start (green) and end (red) marked
• Bottom-Right: Experience Map Topology
◦ Experience map topology: 51 nodes, 8 loop closures, blue solid lines for sequential links, green dashed lines for loop closure edges, start/end marked
2. Corresponding Repository Files
• Drawing Script: neuro/kbs/fig/KITTI_07_manual_save_picture.m
• Generated PDF: neuro/kbs/fig/KITTI_07_input_output_visualization.pdf
• Data Source: RGB images, IMU data, positioning trajectory, experience map topology data from KITTI-07 dataset
3. Reproduction Method
• Dependencies: cv2 (OpenCV, image reading), matplotlib.mplot3d (3D trajectory), networkx (topology drawing), matplotlib (IMU curve plotting)
• Run Command (MATLAB):
matlab
￼
￼
￼
￼
￼
run KITTI_07_manual_save_picture.
m
￼
• Output: PDF file KITTI_07_input_output_visualization.pdf
