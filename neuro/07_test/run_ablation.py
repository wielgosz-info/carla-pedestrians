"""消融实验：多方法多场景对比 (Pure IMU / Pure VO / EKF Fusion vs Ground Truth)"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '07_test', 'experiment_supplement'))
from experiment_templates import ExperimentRecorder

DATA_ROOT = os.path.join(SCRIPT_DIR, '..', 'data')


def load_csv(path):
    return pd.read_csv(path, encoding='utf-8')


def compute_ate(gt_xy, pred_xy):
    """Absolute Trajectory Error (RMSE) in meters"""
    n = min(len(gt_xy), len(pred_xy))
    diff = gt_xy[:n] - pred_xy[:n]
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def compute_rpe(gt_xy, pred_xy):
    """Relative Pose Error per step (mean) in meters/frame"""
    n = min(len(gt_xy), len(pred_xy))
    if n < 2:
        return 0.0
    gt_delta = np.diff(gt_xy[:n], axis=0)
    pred_delta = np.diff(pred_xy[:n], axis=0)
    return float(np.mean(np.linalg.norm(gt_delta - pred_delta, axis=1)))


def compute_drift_rate(gt_xy, pred_xy):
    """漂移率: ATE / GT总路程 (%)"""
    n = min(len(gt_xy), len(pred_xy))
    gt_dist = np.sum(np.linalg.norm(np.diff(gt_xy[:n], axis=0), axis=1))
    ate = compute_ate(gt_xy, pred_xy)
    return float(ate / gt_dist * 100) if gt_dist > 0 else float('inf')


def compute_windowed_ate(gt_xy, pred_xy, window_size=100):
    """每N帧窗口的ATE，返回 [(start_frame, ate), ...]"""
    n = min(len(gt_xy), len(pred_xy))
    windows = []
    for start in range(0, n - window_size + 1, window_size):
        end = start + window_size
        ate = compute_ate(gt_xy[start:end], pred_xy[start:end])
        windows.append((start, ate))
    return windows


def evaluate_dataset(data_dir):
    """评估单个数据集，返回 (dataset_name, list_of_results, trajectories_dict)"""
    name = os.path.basename(data_dir)
    gt_path = os.path.join(data_dir, 'ground_truth.txt')
    vo_path = os.path.join(data_dir, 'visual_odometry.txt')
    fusion_path = os.path.join(data_dir, 'fusion_pose.txt')

    for p, label in [(gt_path, 'GT'), (vo_path, 'VO'), (fusion_path, 'Fusion')]:
        if not os.path.exists(p):
            print(f"  [SKIP] {name}: missing {label} ({p})")
            return None

    gt = load_csv(gt_path)
    vo = load_csv(vo_path)
    fusion = load_csv(fusion_path)

    n_frames = min(len(gt), len(vo), len(fusion))
    dataset_name = f"{name} ({n_frames} frames)"

    # ---- 提取轨迹 ----
    gt_xy = gt[['pos_x', 'pos_y']].values[:n_frames]
    gt_xy = gt_xy - gt_xy[0]

    vo_xy = vo[['vo_x', 'vo_y']].values[:n_frames]
    vo_xy = vo_xy - vo_xy[0]

    fusion_xy = fusion[['pos_x', 'pos_y']].values[:n_frames]
    fusion_xy = fusion_xy - fusion_xy[0]

    # Pure IMU: 从 fusion_pose.txt 的 imu_pos_x, imu_pos_y 列提取
    imu_xy = fusion[['imu_pos_x', 'imu_pos_y']].values[:n_frames]
    imu_xy = imu_xy - imu_xy[0]

    # ---- 构建方法列表 ----
    methods = [
        ('Pure IMU', imu_xy, 'mo-', 'orange'),
        ('Pure VO', vo_xy, 'g--', 'green'),
        ('EKF Fusion', fusion_xy, 'r-', 'red'),
    ]

    results = []
    traj = {'gt': gt_xy, 'methods': []}

    print(f"\n{'─' * 65}")
    print(f"  {dataset_name}")
    print(f"{'─' * 65}")
    print(f"{'Method':<18} {'ATE(m)':<10} {'RPE(m/f)':<10} {'Drift%':<10} {'MaxWinATE':<12}")
    print("-" * 60)

    for method_name, pred_xy, style, color in methods:
        ate = compute_ate(gt_xy, pred_xy)
        rpe = compute_rpe(gt_xy, pred_xy)
        drift = compute_drift_rate(gt_xy, pred_xy)
        windows = compute_windowed_ate(gt_xy, pred_xy, window_size=100)
        max_win = max(w[1] for w in windows) if windows else 0

        print(f"{method_name:<18} {ate:<10.2f} {rpe:<10.4f} {drift:<10.2f} {max_win:<12.2f}")

        results.append({
            'dataset': dataset_name,
            'method': method_name,
            'ATE_m': round(ate, 2),
            'RPE_m': round(rpe, 4),
            'Drift_pct': round(drift, 2),
            'MaxWin100_ATE': round(max_win, 2),
        })
        traj['methods'].append((method_name, pred_xy, style, color))

    return dataset_name, results, traj


def make_plots(dataset_name, traj, data_dir):
    """生成对比图: 轨迹图 + 窗口化ATE"""
    gt_xy = traj['gt']

    # ---- Figure 1: 轨迹对比 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.plot(gt_xy[:, 0], gt_xy[:, 1], 'b-', lw=2, label='Ground Truth')
    for method_name, pred_xy, style, color in traj['methods']:
        ax1.plot(pred_xy[:, 0], pred_xy[:, 1], style, lw=1.2, label=method_name,
                 alpha=0.8 if method_name == 'Pure IMU' else 1.0)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'Trajectory — {dataset_name}')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # ---- Figure 2: 窗口化ATE ----
    for method_name, pred_xy, style, color in traj['methods']:
        windows = compute_windowed_ate(gt_xy, pred_xy, window_size=100)
        if windows:
            starts, ates = zip(*windows)
            linestyle = '--' if '--' in style else '-'
            mk = 's' if '--' in style else 'o'
            ax2.plot(starts, ates, linestyle=linestyle, marker=mk,
                     ms=3, lw=1, label=method_name, color=color, alpha=0.8)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('ATE (m) per 100-frame window')
    ax2.set_title('Error Accumulation Over Time')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(data_dir, 'ablation_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved: {path}")
    return path


def discover_datasets():
    """发现所有数据目录 (排除备份)"""
    datasets = []
    for entry in sorted(os.listdir(DATA_ROOT)):
        full = os.path.join(DATA_ROOT, entry)
        if os.path.isdir(full) and 'Town' in entry and 'backup' not in entry.lower():
            gt = os.path.join(full, 'ground_truth.txt')
            if os.path.exists(gt):
                datasets.append(full)
    if not datasets:
        # 回退到默认
        default = os.path.join(DATA_ROOT, 'Town01Data_IMU_Fusion')
        if os.path.exists(os.path.join(default, 'ground_truth.txt')):
            datasets.append(default)
    return datasets


def main():
    print("=" * 60)
    print("  Ablation Study: Multi-Method Comparison")
    print("  Pure IMU  |  Pure VO  |  EKF Fusion")
    print("=" * 60)

    datasets = discover_datasets()
    if not datasets:
        print("[ERROR] 没有找到数据目录。请先运行数据采集。")
        return

    print(f"\n发现 {len(datasets)} 个数据集:")
    for d in datasets:
        print(f"  - {os.path.basename(d)}")

    recorder = ExperimentRecorder()
    all_trajs = []

    for data_dir in datasets:
        result = evaluate_dataset(data_dir)
        if result is None:
            continue
        dataset_name, results, traj = result
        for r in results:
            recorder.add_result(r['dataset'], r['method'],
                                {k: v for k, v in r.items() if k not in ['dataset', 'method']})
        make_plots(dataset_name, traj, data_dir)
        all_trajs.append((dataset_name, traj, data_dir))

    # ---- 综合对比表 ----
    if len(datasets) > 1:
        print(f"\n{'=' * 60}")
        print("  Cross-Scenario Summary")
        print(f"{'=' * 60}")
        print(recorder.to_latex_table('Ablation Study: Multi-Scenario Comparison'))

    # ---- 保存结果 ----
    out_json = os.path.join(DATA_ROOT, 'ablation_summary.json')
    recorder.save_results(out_json)

    print("\n[OK] Ablation study complete.")


if __name__ == '__main__':
    main()
