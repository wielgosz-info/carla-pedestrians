#!/usr/bin/env python3
"""
NeuroSLAM 一键运行入口
串联 数据采集 -> 消融实验评估 全流程

用法:
    python main.py                    # 运行完整流程（需先启动CARLA）
    python main.py --collect-only     # 仅数据采集
    python main.py --ablate-only      # 仅消融评估（使用已有数据）
    python main.py --skip-collect     # 跳过采集，运行后续步骤
"""

import os
import sys
import time
import argparse
import subprocess
import socket

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECT_SCRIPT = os.path.join(ROOT_DIR, '00_collect_data', 'IMU_Vision_Fusion_EKF.py')
ABLATE_SCRIPT = os.path.join(ROOT_DIR, '07_test', 'run_ablation.py')
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'Town01Data_IMU_Fusion')
DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 2000


def check_carla_server(host=DEFAULT_HOST, port=DEFAULT_PORT, timeout=3.0):
    """检测CARLA服务器是否在运行"""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def find_carla_python():
    """
    找到能导入carla模块的Python解释器。
    CARLA 0.9.16的wheel是cp312的，需要Python 3.12。
    返回 (python_exe_path, version_string) 或 (None, error_msg)
    """
    try:
        result = subprocess.run(
            [sys.executable, '-c', 'import carla'],
            capture_output=True, timeout=10
        )
        if result.returncode == 0:
            return sys.executable, f"当前Python {sys.version_info.major}.{sys.version_info.minor}"
    except Exception:
        pass

    for launcher in ['py', 'python3']:
        for ver_suffix in ['-3.12', '3.12']:
            cmd = [launcher, ver_suffix] if ver_suffix.startswith('-') else [launcher + ver_suffix]
            try:
                result = subprocess.run(
                    [*cmd, '-c', 'import carla'],
                    capture_output=True, timeout=10
                )
                if result.returncode == 0:
                    py_str = f"{cmd[0]} {cmd[1]}" if len(cmd) > 1 else cmd[0]
                    return py_str, 'Python 3.12'
            except FileNotFoundError:
                continue

    localappdata_py = os.path.expandvars(
        r'%LOCALAPPDATA%\Programs\Python\Python312\python.exe'
    )
    if os.path.exists(localappdata_py):
        try:
            result = subprocess.run(
                [localappdata_py, '-c', 'import carla'],
                capture_output=True, timeout=10
            )
            if result.returncode == 0:
                return localappdata_py, 'Python 3.12'
        except Exception:
            pass

    return None, (
        "找不到可导入carla的Python解释器。\n"
        "CARLA 0.9.16需要Python 3.12，请安装carla wheel:\n"
        "  py -3.12 -m pip install <CARLA_DIR>\\PythonAPI\\carla\\dist\\carla-*.whl"
    )


def print_banner():
    print("=" * 60)
    print("   NeuroSLAM -- Bio-Inspired VIO Pipeline")
    print("=" * 60)


def run_script(script_path, desc, python_exe=None):
    """运行Python脚本，返回是否成功"""
    if python_exe is None:
        python_exe = sys.executable

    print(f"\n{'=' * 60}")
    print(f"  >>> {desc}")
    print(f"  >>> 脚本: {os.path.basename(script_path)}")
    print(f"{'=' * 60}\n")

    if isinstance(python_exe, str) and ' ' in python_exe:
        cmd = python_exe.split() + [script_path]
    else:
        cmd = [python_exe, script_path] if isinstance(python_exe, str) else python_exe + [script_path]

    print(f"[INFO] 使用解释器: {' '.join(cmd[:2])}")

    result = subprocess.run(cmd, cwd=os.path.dirname(script_path))
    if result.returncode != 0:
        print(f"\n[ERROR] {desc} 失败 (exit code {result.returncode})")
        return False
    print(f"\n[OK] {desc} 完成")
    return True


def step_collect(carla_host, carla_port):
    """Step 1: CARLA数据采集（需要Python 3.12 + carla）"""
    carla_py, info = find_carla_python()
    if carla_py is None:
        print(f"\n[ERROR] {info}")
        return False
    print(f"[INFO] CARLA数据采集使用: {info}")

    if not check_carla_server(carla_host, carla_port):
        print("\n" + "=" * 60)
        print("  [阻塞] 等待CARLA服务器...")
        print("  请手动启动CARLA (例如: CarlaUE4.exe -RenderOffScreen -quality-level=Low)")
        print("=" * 60)

        while not check_carla_server(carla_host, carla_port):
            time.sleep(3)
        print("[OK] CARLA服务器已连接\n")

    if os.path.exists(DATA_DIR):
        existing = [f for f in os.listdir(DATA_DIR) if f.endswith('.png')]
        if existing:
            print(f"[WARN] 数据目录已有 {len(existing)} 张图像")
            print(f"  目录: {DATA_DIR}")
            resp = input("  是否删除旧数据并重新采集? [y/N]: ").strip().lower()
            if resp == 'y':
                import shutil
                shutil.rmtree(DATA_DIR)
                print("  旧数据已删除")
            else:
                print("  跳过数据采集")
                return True

    return run_script(COLLECT_SCRIPT,
                      "Step 1/2: CARLA数据采集 (IMU+Vision EKF融合)",
                      python_exe=carla_py)


def step_ablate():
    """Step 2: 消融实验评估（当前Python即可，无需carla）"""
    gt_file = os.path.join(DATA_DIR, 'ground_truth.txt')
    vo_file = os.path.join(DATA_DIR, 'visual_odometry.txt')
    fusion_file = os.path.join(DATA_DIR, 'fusion_pose.txt')

    missing = []
    for f, name in [(gt_file, 'Ground Truth'), (vo_file, 'Visual Odometry'),
                     (fusion_file, 'Fusion Pose')]:
        if not os.path.exists(f):
            missing.append(f"  - {name}: {f}")

    if missing:
        print("\n[ERROR] 缺少数据文件，无法运行消融评估:")
        print("\n".join(missing))
        print("\n请先运行数据采集: python main.py --collect-only")
        return False

    return run_script(ABLATE_SCRIPT, "Step 2/2: 消融实验评估 (VO vs EKF)")


def print_results():
    """打印已有的评估结果"""
    result_file = os.path.join(DATA_DIR, 'ablation_results.json')
    if not os.path.exists(result_file):
        return

    import json
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    print(f"{'Method':<25} {'ATE(m)':<12} {'RPE(m/f)':<12} {'Drift%':<10}")
    print("-" * 60)
    for r in results:
        ate = r.get('ATE_m', 'N/A')
        rpe = r.get('RPE_m', 'N/A')
        drift = r.get('Drift_pct', 'N/A')
        print(f"{r['method']:<25} {str(ate):<12} {str(rpe):<12} {str(drift):<10}")

    plot_files = ['ablation_trajectory.png', 'ablation_windowed_ate.png']
    for pf in plot_files:
        path = os.path.join(DATA_DIR, pf)
        if os.path.exists(path):
            print(f"  图表: {path}")


def main():
    parser = argparse.ArgumentParser(
        description='NeuroSLAM -- 一键数据采集+消融评估',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                    一键运行完整流程
  python main.py --collect-only     仅采集CARLA数据
  python main.py --ablate-only      仅运行消融评估
  python main.py --skip-collect     跳过采集，只评估
"""
    )
    parser.add_argument('--collect-only', action='store_true',
                        help='仅运行数据采集')
    parser.add_argument('--ablate-only', action='store_true',
                        help='仅运行消融评估（需要已有数据）')
    parser.add_argument('--skip-collect', action='store_true',
                        help='跳过数据采集，直接评估')
    parser.add_argument('--host', default=DEFAULT_HOST,
                        help=f'CARLA服务器地址 (默认: {DEFAULT_HOST})')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help=f'CARLA服务器端口 (默认: {DEFAULT_PORT})')

    args = parser.parse_args()

    print_banner()

    if args.collect_only or (not args.ablate_only and not args.skip_collect):
        carla_py, info = find_carla_python()
        if carla_py is None:
            print(f"\n[FATAL] {info}")
            sys.exit(1)
        print(f"[INFO] CARLA Python: {info}")

    start_time = time.time()
    success = True

    if args.collect_only:
        success = step_collect(args.host, args.port)
    elif args.ablate_only:
        success = step_ablate()
    elif args.skip_collect:
        success = step_ablate()
    else:
        success = step_collect(args.host, args.port)
        if success:
            success = step_ablate()

    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n总耗时: {mins}分{secs}秒")

    if success:
        print_results()
        print("\n[SUCCESS] NeuroSLAM流程全部完成!")
    else:
        print("\n[FAILED] 流程中断，请检查上方错误信息")
        sys.exit(1)


if __name__ == '__main__':
    main()
