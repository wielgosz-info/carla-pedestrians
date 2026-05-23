"""消融实验：对比 Pure VO / EKF Fusion 与 Ground Truth 的轨迹误差"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '07_test', 'experiment_supplement'))

from experiment_templates import ExperimentRecorder

DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'Town01Data_IMU_Fusion')


def load_csv(path):
    """加载CSV数据，去除表头注释行"""
    df = pd.read_csv(path, encoding='utf-8')
    # 去掉第一行如果是单位/注释
    return df


def compute_ate(gt_xy, pred_xy):
    """Absolute Trajectory Error (RMSE)"""
    n = min(len(gt_xy), len(pred_xy))
    diff = gt_xy[:n] - pred_xy[:n]
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


def compute_rpe(gt_xy, pred_xy):
    """Relative Pose Error per step (mean)"""
    n = min(len(gt_xy), len(pred_xy))
    if n < 2:
        return 0.0
    gt_delta = np.diff(gt_xy[:n], axis=0)
    pred_delta = np.diff(pred_xy[:n], axis=0)
    errors = np.linalg.norm(gt_delta - pred_delta, axis=1)
    return float(np.mean(errors))


def main():
    print("=" * 60)
    print("Ablation Study: VO vs EKF Fusion")
    print("=" * 60)

    gt = load_csv(os.path.join(DATA_DIR, 'ground_truth.txt'))
    vo = load_csv(os.path.join(DATA_DIR, 'visual_odometry.txt'))
    fusion = load_csv(os.path.join(DATA_DIR, 'fusion_pose.txt'))

    gt_xy = gt[['pos_x', 'pos_y']].values
    gt_xy -= gt_xy[0]  # 对齐到原点
    vo_xy = vo[['vo_x', 'vo_y']].values
    vo_xy -= vo_xy[0]
    fusion_xy = fusion[['pos_x', 'pos_y']].values
    fusion_xy -= fusion_xy[0]

    n_frames = len(gt)
    dataset_name = f'Town01 ({n_frames} frames)'

    # 计算指标
    vo_ate = compute_ate(gt_xy, vo_xy)
    fusion_ate = compute_ate(gt_xy, fusion_xy)
    vo_rpe = compute_rpe(gt_xy, vo_xy)
    fusion_rpe = compute_rpe(gt_xy, fusion_xy)

    print(f"\n{'Method':<20} {'ATE (m)':<12} {'RPE (m/frame)':<15}")
    print("-" * 47)
    print(f"{'Pure Visual (VO)':<20} {vo_ate:<12.4f} {vo_rpe:<15.4f}")
    print(f"{'EKF Fusion':<20} {fusion_ate:<12.4f} {fusion_rpe:<15.4f}")

    # 记录结果
    recorder = ExperimentRecorder()
    recorder.add_result(dataset_name, 'Pure Visual (VO)', {
        'ATE_m': round(vo_ate, 4),
        'RPE_m': round(vo_rpe, 4),
    })
    recorder.add_result(dataset_name, 'EKF Fusion (IMU+VO)', {
        'ATE_m': round(fusion_ate, 4),
        'RPE_m': round(fusion_rpe, 4),
    })

    print("\n" + recorder.to_latex_table('Ablation Study: Town01'))

    # 画图
    plt.figure(figsize=(10, 8))
    plt.plot(gt_xy[:, 0], gt_xy[:, 1], 'b-', lw=2, label='Ground Truth')
    plt.plot(vo_xy[:, 0], vo_xy[:, 1], 'g--', lw=1.5, label='Pure Visual (VO)')
    plt.plot(fusion_xy[:, 0], fusion_xy[:, 1], 'r--', lw=1.5, label='EKF Fusion')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory Comparison: Ablation Study')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    out_path = os.path.join(DATA_DIR, 'ablation_trajectory.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nTrajectory plot saved: {out_path}")

    result_path = os.path.join(DATA_DIR, 'ablation_results.json')
    recorder.save_results(result_path)

    print("\nDone. Next: increase MAX_SAVE_IMG to 5000 for longer trajectory.")


if __name__ == '__main__':
    main()
