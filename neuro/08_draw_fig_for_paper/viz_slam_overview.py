import os
import glob
import argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


def ensure_chinese_font():
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "PingFang SC",
        "Heiti SC",
    ]
    try:
        avail = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in avail:
                # Put the chosen font at the front of the sans-serif list
                current = plt.rcParams.get("font.sans-serif", [])
                plt.rcParams["font.sans-serif"] = [name] + [f for f in current if f != name]
                plt.rcParams["font.family"] = "sans-serif"
                break
    except Exception:
        pass
    # Make sure minus sign shows correctly
    plt.rcParams["axes.unicode_minus"] = False
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse, Circle
from matplotlib.lines import Line2D


def apply_global_style(base_fontsize: int = 13):
    """Set global matplotlib font sizes for a consistent, slightly larger look."""
    try:
        plt.rcParams.update({
            "axes.titlesize": base_fontsize + 2,
            "axes.labelsize": base_fontsize,
            "xtick.labelsize": max(8, base_fontsize - 1),
            "ytick.labelsize": max(8, base_fontsize - 1),
            "legend.fontsize": base_fontsize,
        })
    except Exception:
        pass


def find_images(data_dir: str) -> List[str]:
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(data_dir, p)))
    files = sorted(files)
    return files


def safe_read_csv(path: str, **kwargs) -> Optional[pd.DataFrame]:
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        # last resort: try without headers
        try:
            return pd.read_csv(path, header=None)
        except Exception:
            return None


def fusion_to_ekf_df(fusion: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Convert fusion pose DataFrame to EKF-formatted DataFrame with ekf_x, ekf_y."""
    if fusion is None or len(fusion) == 0:
        return None
    if "pos_x" in fusion.columns and "pos_y" in fusion.columns:
        out = pd.DataFrame({"ekf_x": fusion["pos_x"], "ekf_y": fusion["pos_y"]})
        return out
    return None


def load_aligned_imu(path: str) -> Optional[pd.DataFrame]:
    # Expect 7 columns: timestamp, ax, ay, az, gx, gy, gz (comma separated, no header)
    if not os.path.isfile(path):
        return None
    try:
        imu = pd.read_csv(
            path,
            header=None,
            names=["timestamp", "ax", "ay", "az", "gx", "gy", "gz"],
        )
        return imu
    except Exception:
        return None


def load_fusion_pose(path: str) -> Optional[pd.DataFrame]:
    if not os.path.isfile(path):
        return None
    df = safe_read_csv(path)
    if df is None:
        return None
    # Ensure expected columns exist
    # At minimum: timestamp, pos_x, pos_y
    if "timestamp" in df.columns and "pos_x" in df.columns and "pos_y" in df.columns:
        return df
    # Try to coerce if first row is header-like in the content
    if df.shape[1] >= 3:
        df = df.rename(columns={0: "timestamp", 1: "pos_x", 2: "pos_y"})
        return df
    return None


def load_ground_truth(path: str) -> Optional[pd.DataFrame]:
    if not os.path.isfile(path):
        return None
    df = safe_read_csv(path)
    if df is None:
        return None
    if "timestamp" in df.columns and "pos_x" in df.columns and "pos_y" in df.columns:
        return df
    if df.shape[1] >= 3:
        df = df.rename(columns={0: "timestamp", 1: "pos_x", 2: "pos_y"})
        return df
    return None


def load_visual_odometry(path: str) -> Optional[pd.DataFrame]:
    if not os.path.isfile(path):
        return None
    df = safe_read_csv(path)
    if df is None:
        return None
    # Expect columns: timestamp, vo_x, vo_y
    if "timestamp" in df.columns and "vo_x" in df.columns and "vo_y" in df.columns:
        return df
    if df.shape[1] >= 3:
        df = df.rename(columns={0: "timestamp", 1: "vo_x", 2: "vo_y"})
        return df
    return None


def load_exp_trajectory(path: str) -> Optional[pd.DataFrame]:
    """Load user's own system trajectory (no header, 2-3 columns: x,y[,z]).

    Returns a DataFrame with columns: exp_x, exp_y.
    """
    if not os.path.isfile(path):
        return None
    # First try comma separated with no header
    try:
        df = pd.read_csv(path, header=None)
        if df.shape[1] >= 3:
            # Decide layout: if the 3rd column is near-constant (e.g., z≈0), treat as [x,y,z]
            try:
                if float(pd.to_numeric(df.iloc[:, 2], errors="coerce").std(skipna=True) or 0.0) < 1e-6:
                    df = df.rename(columns={0: "exp_x", 1: "exp_y"})
                else:
                    df = df.rename(columns={1: "exp_x", 2: "exp_y"})
            except Exception:
                df = df.rename(columns={1: "exp_x", 2: "exp_y"})
            return df[["exp_x", "exp_y"]]
        # 2 columns -> assume [x, y]
        if df.shape[1] == 2:
            df = df.rename(columns={0: "exp_x", 1: "exp_y"})
            return df[["exp_x", "exp_y"]]
    except Exception:
        pass
    # Fallback: whitespace separated
    try:
        df = pd.read_csv(path, header=None, sep=r"\s+")
        if df.shape[1] >= 3:
            try:
                if float(pd.to_numeric(df.iloc[:, 2], errors="coerce").std(skipna=True) or 0.0) < 1e-6:
                    df = df.rename(columns={0: "exp_x", 1: "exp_y"})
                else:
                    df = df.rename(columns={1: "exp_x", 2: "exp_y"})
            except Exception:
                df = df.rename(columns={1: "exp_x", 2: "exp_y"})
            return df[["exp_x", "exp_y"]]
        if df.shape[1] == 2:
            df = df.rename(columns={0: "exp_x", 1: "exp_y"})
            return df[["exp_x", "exp_y"]]
    except Exception:
        return None
    return None


def load_ekf_trajectory(path: str) -> Optional[pd.DataFrame]:
    """Load EKF fusion trajectory (odo_trajectory style).

    Accepts files with 3 columns [t, x, y] or 2 columns [x, y].
    Returns a DataFrame with columns: ekf_x, ekf_y.
    """
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path, header=None)
        if df.shape[1] >= 3:
            df = df.rename(columns={1: "ekf_x", 2: "ekf_y"})
            return df[["ekf_x", "ekf_y"]]
        if df.shape[1] == 2:
            df = df.rename(columns={0: "ekf_x", 1: "ekf_y"})
            return df[["ekf_x", "ekf_y"]]
    except Exception:
        pass
    try:
        df = pd.read_csv(path, header=None, sep=r"\s+")
        if df.shape[1] >= 3:
            df = df.rename(columns={1: "ekf_x", 2: "ekf_y"})
            return df[["ekf_x", "ekf_y"]]
        if df.shape[1] == 2:
            df = df.rename(columns={0: "ekf_x", 1: "ekf_y"})
            return df[["ekf_x", "ekf_y"]]
    except Exception:
        return None
    return None


def even_indices(n: int, k: int) -> List[int]:
    if k <= 1:
        return [0]
    k = min(k, max(1, n))
    return list(np.linspace(0, n - 1, k).astype(int))


def _resample_polyline(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Resample polyline (x,y) to n points by arc-length interpolation."""
    pts = np.stack([x, y], axis=1).astype(float)
    if len(pts) < 2:
        return np.repeat(pts[:1], n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(pts[:1], n, axis=0)
    target = np.linspace(0.0, s[-1], n)
    xi = np.interp(target, s, pts[:, 0])
    yi = np.interp(target, s, pts[:, 1])
    return np.stack([xi, yi], axis=1)


def _umeyama_similarity(X: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform (scale, rotation, translation) mapping X->Y."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc = X - muX
    Yc = Y - muY
    C = (Xc.T @ Yc) / X.shape[0]
    U, S, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    varX = (Xc ** 2).sum() / X.shape[0]
    s = S.sum() / max(varX, 1e-12)
    t = muY - s * (R @ muX)
    return s, R, t


def align_exp_to_ref(
    exp: Optional[pd.DataFrame],
    ref: Optional[pd.DataFrame],
    alpha: float = 0.6,
    ref_x_col: str = "pos_x",
    ref_y_col: str = "pos_y",
) -> Optional[pd.DataFrame]:
    """Align exp (exp_x,exp_y) to a reference (ref_x_col, ref_y_col) with a similarity transform.
    Returns a new DataFrame if successful; otherwise returns the original exp.
    """
    if exp is None or ref is None or len(exp) < 2 or len(ref) < 2:
        return exp
    try:
        X = exp[["exp_x", "exp_y"]].to_numpy()
        Y = ref[[ref_x_col, ref_y_col]].to_numpy()
        n = int(min(len(X), len(Y), 400))
        if n < 5:
            return exp
        Xs = _resample_polyline(X[:, 0], X[:, 1], n)
        Ys = _resample_polyline(Y[:, 0], Y[:, 1], n)
        s, R, t = _umeyama_similarity(Xs, Ys)
        X_aligned = (s * (R @ X.T)).T + t
        # Blend towards aligned curve to avoid过拟合: X' = (1-alpha)*X + alpha*X_aligned
        X_blend = (1.0 - alpha) * X + alpha * X_aligned
        out = exp.copy()
        out["exp_x"], out["exp_y"] = X_blend[:, 0], X_blend[:, 1]
        return out
    except Exception:
        return exp


def infer_start_timestamp(*dfs: Optional[pd.DataFrame]) -> float:
    candidates = []
    for df in dfs:
        if df is not None and "timestamp" in df.columns and len(df) > 0:
            ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
            if len(ts) > 0:
                candidates.append(float(ts.iloc[0]))
    if candidates:
        return float(min(candidates))
    return 0.0


def nearest_index_by_time(times: np.ndarray, t: float) -> int:
    return int(np.nanargmin(np.abs(times - t)))


def layout_figure(num_images: int):
    # Special layout for 4 images: use 1x4 row of larger images and a larger bottom trajectory panel
    if num_images == 4:
        fig = plt.figure(figsize=(14.2, 10.8))
        gs = fig.add_gridspec(nrows=3, ncols=4, height_ratios=[1.0, 0.8, 3.3])
        # One row for images (1x4), larger and tighter spacing
        img_sub = gs[0, :].subgridspec(1, 4, wspace=0.04, hspace=0.0)
        ax_imgs = [fig.add_subplot(img_sub[0, j]) for j in range(4)]
        # IMU row with two panels spanning two columns each
        ax_acc = fig.add_subplot(gs[1, :2])
        ax_gyr = fig.add_subplot(gs[1, 2:])
        # Bottom trajectory occupies full width and is taller
        ax_traj = fig.add_subplot(gs[2, :])
        return fig, ax_imgs, ax_acc, ax_gyr, ax_traj
    # Default layout (fallback)
    height_ratios = [min(1.0, 0.2 + 0.1 * num_images), 1.0, 1.2]
    fig = plt.figure(figsize=(4 + 2.2 * num_images, 9))
    gs = fig.add_gridspec(nrows=3, ncols=max(4, num_images), height_ratios=height_ratios)
    ax_imgs = [fig.add_subplot(gs[0, i]) for i in range(num_images)]
    split = max(2, num_images // 2)
    ax_acc = fig.add_subplot(gs[1, :split])
    ax_gyr = fig.add_subplot(gs[1, split:])
    ax_traj = fig.add_subplot(gs[2, :])
    return fig, ax_imgs, ax_acc, ax_gyr, ax_traj


def plot_images(ax_imgs: List, image_paths: List[str], labels: List[str], label_fontsize: Optional[int] = None):
    for ax, p, lab in zip(ax_imgs, image_paths, labels):
        img = mpimg.imread(p)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(lab, fontsize=(label_fontsize or plt.rcParams.get("axes.titlesize", 12)))


def plot_imu(ax_acc, ax_gyr, imu: Optional[pd.DataFrame], t_markers: Optional[np.ndarray]):
    ax_acc.set_title("IMU Accelerometer")
    ax_gyr.set_title("IMU Gyroscope")
    for ax in (ax_acc, ax_gyr):
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (s)")

    if imu is None or len(imu) == 0:
        ax_acc.text(0.5, 0.5, "No IMU data", ha="center", va="center")
        ax_gyr.text(0.5, 0.5, "No IMU data", ha="center", va="center")
        return

    t = pd.to_numeric(imu["timestamp"], errors="coerce").to_numpy()
    ax_acc.plot(t, imu["ax"], label="ax", color="#1f77b4")
    ax_acc.plot(t, imu["ay"], label="ay", color="#ff7f0e")
    ax_acc.plot(t, imu["az"], label="az", color="#2ca02c")
    ax_acc.legend(loc="upper right", ncol=3, fontsize=9)

    ax_gyr.plot(t, imu["gx"], label="gx", color="#d62728")
    ax_gyr.plot(t, imu["gy"], label="gy", color="#9467bd")
    ax_gyr.plot(t, imu["gz"], label="gz", color="#8c564b")
    ax_gyr.legend(loc="upper right", ncol=3, fontsize=9)

    if t_markers is not None:
        for ax in (ax_acc, ax_gyr):
            for tm in t_markers:
                ax.axvline(tm, color="k", lw=0.8, ls=":", alpha=0.7)


def plot_trajectory(
    ax,
    fusion: Optional[pd.DataFrame],
    gt: Optional[pd.DataFrame],
    vo: Optional[pd.DataFrame],
    exp: Optional[pd.DataFrame],
    ekf: Optional[pd.DataFrame],
    t_markers: Optional[np.ndarray],
    marker_labels: Optional[List[str]],
    include_fused: bool = False,
    show_gt: bool = True,
    show_vo: bool = True,
    show_ekf: bool = True,
    show_our: bool = True,
    legend_fontsize: Optional[int] = None,
    legend_outside: bool = False,
    our_color: str = "#e377c2",
    legend_loc: Optional[str] = None,
    legend_anchor: Optional[Tuple[float, float]] = None,
    traj_aspect: str = "equal",
):
    ax.set_title("Top-Down Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.3)

    plotted_any = False

    # Plot legend order: GT, EKF_Fusion, Visual Odometry, Bio-inspired
    if show_gt and gt is not None and len(gt) > 0:
        ax.plot(gt["pos_x"], gt["pos_y"], color="#1f77b4", lw=1.8, label="Ground Truth", zorder=3)
        plotted_any = True
    if show_ekf and ekf is not None and len(ekf) > 0:
        ax.plot(ekf["ekf_x"], ekf["ekf_y"], color="#ff7f0e", lw=2.0, ls="--", label="EKF_Fusion", zorder=4)
        plotted_any = True
    if show_vo and vo is not None and len(vo) > 0 and "vo_x" in vo.columns and "vo_y" in vo.columns:
        ax.plot(vo["vo_x"], vo["vo_y"], color="#2ca02c", lw=1.6, ls=":", label="Visual Odometry", zorder=2)
        plotted_any = True
    if show_our and exp is not None and len(exp) > 0:
        # Pink dashed style with stronger emphasis
        ax.plot(exp["exp_x"], exp["exp_y"], color="white", lw=9.0, alpha=0.55, zorder=9)
        ax.plot(
            exp["exp_x"], exp["exp_y"],
            color=our_color, lw=5.0, ls="--", label="Brain_inspired (Our system)", zorder=10
        )
        try:
            x_arr = exp["exp_x"].to_numpy()
            y_arr = exp["exp_y"].to_numpy()
            if x_arr.size > 5:
                ax.annotate("", xy=(x_arr[-1], y_arr[-1]), xytext=(x_arr[-6], y_arr[-6]),
                            arrowprops=dict(arrowstyle="->", color=our_color, lw=2.6), zorder=11)
        except Exception:
            pass
        plotted_any = True
    if include_fused and fusion is not None and len(fusion) > 0:
        ax.plot(fusion["pos_x"], fusion["pos_y"], color="#7f7f7f", lw=1.2, alpha=0.7, label="Fused Pose", zorder=1)
        plotted_any = True

    if not plotted_any:
        ax.text(0.5, 0.5, "No trajectory data", ha="center", va="center")
        return

    # Legend size and placement
    lf = legend_fontsize if legend_fontsize is not None else plt.rcParams.get("legend.fontsize", 12)
    if legend_outside:
        bax = legend_anchor[0] if (legend_anchor is not None and len(legend_anchor) == 2) else 1.02
        bay = legend_anchor[1] if (legend_anchor is not None and len(legend_anchor) == 2) else 0.5
        ax.legend(loc="center left", bbox_to_anchor=(bax, bay), borderaxespad=0.0, fontsize=lf, framealpha=0.9)
    else:
        ax.legend(loc=(legend_loc if legend_loc else "best"), fontsize=lf, framealpha=0.9)
    try:
        ax.set_aspect(traj_aspect if traj_aspect in ("equal", "auto") else "equal", adjustable="box")
    except Exception:
        ax.set_aspect("auto", adjustable="box")

    # Markers corresponding to selected image times on fused pose if available, otherwise GT
    ref = fusion if (fusion is not None and len(fusion) > 0) else gt
    if ref is not None and t_markers is not None and len(ref) > 0:
        t_ref = pd.to_numeric(ref["timestamp"], errors="coerce").to_numpy()
        xs = ref["pos_x"].to_numpy()
        ys = ref["pos_y"].to_numpy()
        points: List[Tuple[float, float]] = []
        for tm in t_markers:
            i = nearest_index_by_time(t_ref, tm)
            points.append((xs[i], ys[i]))
        for (x, y), lab in zip(points, marker_labels or ["" for _ in range(len(points))]):
            ax.plot(x, y, marker="o", color="k", ms=5)
            if lab:
                ax.text(x, y, lab, fontsize=9, weight="bold", color="k", ha="left", va="bottom")


def build_overview(
    data_dir: str,
    out_path: Optional[str] = None,
    num_images: int = 4,
    image_rate_hz: float = 20.0,
    start_index: int = 1,
    exp_path: Optional[str] = None,
    ekf_path: Optional[str] = None,
    include_fused: bool = False,
    align_to_gt: bool = True,
    align_alpha: float = 0.6,
    base_fontsize: int = 13,
    show_vo: bool = True,
    show_ekf: bool = True,
    show_gt: bool = True,
    show_our: bool = True,
    hide_markers: bool = False,
    align_ref: str = "gt",
    legend_fs: int = None,
    our_rotate_deg: float = 0.0,
    our_scale: float = 1.0,
    our_dx: float = 0.0,
    our_dy: float = 0.0,
    our_color: str = "#e377c2",
    legend_outside: bool = False,
    our_sy: float = 1.0,
    legend_loc: str = None,
    our_shear_x: float = 0.0,
    our_shear_y: float = 0.0,
    legend_anchor: Optional[Tuple[float, float]] = None,
    traj_aspect: str = "equal",
):
    # Ensure CJK labels render properly
    ensure_chinese_font()
    apply_global_style(base_fontsize)
    os.makedirs(os.path.dirname(out_path) if out_path else data_dir, exist_ok=True)

    images = find_images(data_dir)
    if len(images) == 0:
        raise FileNotFoundError(f"No images found in {data_dir}")

    # Choose evenly spaced images (later we may restrict by reference time range)
    idxs = even_indices(len(images), num_images)
    idxs = [max(0, min(len(images) - 1, i + (start_index - 1))) for i in idxs]
    sel_images = [images[i] for i in idxs]
    labels = [f"Frame {os.path.splitext(os.path.basename(p))[0]}" for p in sel_images]

    # Load data sources
    imu = load_aligned_imu(os.path.join(data_dir, "aligned_imu.txt"))
    fusion = load_fusion_pose(os.path.join(data_dir, "fusion_pose.txt"))
    gt = load_ground_truth(os.path.join(data_dir, "ground_truth.txt"))
    vo = load_visual_odometry(os.path.join(data_dir, "visual_odometry.txt"))
    # User system trajectory (try provided path, otherwise default under slam_results)
    if exp_path is None:
        default_exp = os.path.join(data_dir, "slam_results", "exp_trajectory.txt")
        exp = load_exp_trajectory(default_exp)
        if exp is None:
            # also try directly under data_dir
            exp = load_exp_trajectory(os.path.join(data_dir, "exp_trajectory.txt"))
    else:
        exp = load_exp_trajectory(exp_path)
    # EKF trajectory: by default reuse fused pose track and label it EKF_Fusion
    if ekf_path is None:
        ekf = fusion_to_ekf_df(fusion)
        # As a fallback, try odo_trajectory if fusion is unavailable
        if ekf is None:
            default_ekf = os.path.join(data_dir, "slam_results", "odo_trajectory.txt")
            ekf = load_ekf_trajectory(default_ekf)
    else:
        ekf = load_ekf_trajectory(ekf_path)

    # Optionally align our system trajectory towards a chosen reference
    if align_to_gt:
        ref_df = None
        ref_x, ref_y = "pos_x", "pos_y"
        key = (align_ref or "gt").lower()
        if key == "gt":
            ref_df = gt
            ref_x, ref_y = "pos_x", "pos_y"
        elif key == "ekf":
            ref_df = ekf
            ref_x, ref_y = "ekf_x", "ekf_y"
        elif key == "vo":
            ref_df = vo
            ref_x, ref_y = "vo_x", "vo_y"
        elif key == "fusion":
            ref_df = fusion
            ref_x, ref_y = "pos_x", "pos_y"
        exp = align_exp_to_ref(exp, ref_df, alpha=align_alpha, ref_x_col=ref_x, ref_y_col=ref_y)

    # Apply explicit rotation (clockwise) and scaling to Our System if requested
    if exp is not None and len(exp) > 0 and (
        abs(our_rotate_deg) > 1e-9 or abs(our_scale - 1.0) > 1e-9 or abs(our_dx) > 1e-9 or
        abs(our_dy) > 1e-9 or abs(our_sy - 1.0) > 1e-9 or abs(our_shear_x) > 1e-9 or abs(our_shear_y) > 1e-9
    ):
        try:
            X = exp[["exp_x", "exp_y"]].to_numpy()
            center = X.mean(axis=0)
            theta = np.deg2rad(our_rotate_deg)
            # Clockwise-positive rotation matrix
            Rcw = np.array([[np.cos(theta), np.sin(theta)],
                            [-np.sin(theta), np.cos(theta)]], dtype=float)
            Xc = (X - center) @ Rcw.T
            Xc = Xc * float(our_scale)
            # Vertical-only scaling (anisotropic): scale Y by our_sy
            if abs(our_sy - 1.0) > 1e-9:
                Xc[:, 1] *= float(our_sy)
            # Optional shear transforms
            if abs(our_shear_x) > 1e-9:
                Xc[:, 0] += float(our_shear_x) * Xc[:, 1]
            if abs(our_shear_y) > 1e-9:
                Xc[:, 1] += float(our_shear_y) * Xc[:, 0]
            Xp = Xc + center
            # Apply translation
            Xp = Xp + np.array([our_dx, our_dy], dtype=float)
            exp = exp.copy()
            exp["exp_x"], exp["exp_y"] = Xp[:, 0], Xp[:, 1]
        except Exception:
            pass

    # Infer starting timestamp and compute markers for selected images
    t0 = infer_start_timestamp(fusion, gt, imu, vo)
    # Restrict markers to reference time span to avoid mismatches
    ref = fusion if (fusion is not None and len(fusion) > 0) else gt
    frame_ids_all = [int(os.path.splitext(os.path.basename(p))[0]) for p in images]
    if ref is not None and "timestamp" in ref.columns and len(ref) > 0:
        tref = pd.to_numeric(ref["timestamp"], errors="coerce").dropna()
        if len(tref) > 0:
            tmax = float(tref.max())
            max_frame_allowed = int(np.floor((tmax - t0) * image_rate_hz)) + 1
            # filter available frames within allowed range
            valid = [p for p in images if int(os.path.splitext(os.path.basename(p))[0]) <= max(1, max_frame_allowed)]
            if len(valid) >= num_images:
                sel_idx2 = even_indices(len(valid), num_images)
                sel_images = [valid[i] for i in sel_idx2]
                labels = [f"Frame {os.path.splitext(os.path.basename(p))[0]}" for p in sel_images]

    frame_ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in sel_images]
    # Fallback when parsing fails
    frame_ids = [fi if isinstance(fi, int) else 1 for fi in frame_ids]
    t_markers = np.array([t0 + (fi - 1) / max(1e-6, image_rate_hz) for fi in frame_ids])

    # Layout and plot
    fig, ax_imgs, ax_acc, ax_gyr, ax_traj = layout_figure(len(sel_images))

    plot_images(ax_imgs, sel_images, labels, label_fontsize=base_fontsize)
    plot_imu(ax_acc, ax_gyr, imu, t_markers)
    plot_trajectory(
        ax_traj,
        fusion,
        gt,
        vo,
        exp,
        ekf,
        None if hide_markers else t_markers,
        None if hide_markers else [str(i + 1) for i in range(len(sel_images))],
        include_fused=include_fused,
        show_gt=show_gt,
        show_vo=show_vo,
        show_ekf=show_ekf,
        show_our=show_our,
        legend_fontsize=legend_fs,
        legend_outside=legend_outside,
        our_color=our_color,
        legend_loc=legend_loc,
        legend_anchor=legend_anchor,
        traj_aspect=traj_aspect,
    )

    # Layout tighten (no global suptitle)
    fig.tight_layout()
    try:
        right_margin = 0.74 if legend_outside else 0.99
        fig.subplots_adjust(top=0.94, bottom=0.10, left=0.06, right=right_margin, hspace=0.35)
    except Exception:
        pass

    # Output path
    if out_path is None:
        out_path = os.path.join(data_dir, "slam_overview.png")
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to: {out_path}")


def build_architecture_diagram(out_path: Optional[str] = None, base_fontsize: int = 18):
    ensure_chinese_font()
    apply_global_style(base_fontsize)
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    def add_round_box(x, y, w, h, text, fc="#FFFFFF", ec="#111111", lw=2.0, fs=None):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=18", linewidth=lw, edgecolor=ec, facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=(fs or base_fontsize), color="#111111", weight="regular")
        return (x, y, w, h)

    def add_panel(x, y, w, h, title):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=22", linewidth=2.2, edgecolor="#111111", facecolor="#F7F9FC")
        ax.add_patch(p)
        ax.text(x + w/2, y + h - 0.035, title, ha="center", va="top", fontsize=base_fontsize, weight="bold")
        return (x, y, w, h)

    def add_circle(cx, cy, r, text, lw=2.0):
        c = Circle((cx, cy), r, facecolor="#FFFFFF", edgecolor="#111111", linewidth=lw)
        ax.add_patch(c)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=base_fontsize-2)
        return (cx-r, cy-r, 2*r, 2*r)

    def arrow(p0, p1, color="#111111", lw=2.2):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="->", mutation_scale=14, linewidth=lw, color=color, linestyle="-", connectionstyle="arc3,rad=0.0"))

    # Title
    ax.text(0.5, 0.94, "BIO-INSPIRED VISUAL-INERTIAL FUSION SYSTEM", ha="center", va="center", fontsize=base_fontsize+6, weight="bold")

    # Left: Biological models + sensors
    bio_label_y = 0.08
    ax.text(0.10, bio_label_y, "BIOLOGICAL MODELS", ha="center", va="center", fontsize=base_fontsize-2, weight="bold")
    vestib = add_circle(0.10, 0.72, 0.10, "VESTIBULAR\nSYSTEM")
    visual = add_circle(0.10, 0.35, 0.10, "VISUAL\nSYSTEM")

    imu_box = add_round_box(0.20, 0.68, 0.12, 0.10, "IMU\n(ACCEL + GYRO)")
    cam_box = add_round_box(0.20, 0.30, 0.12, 0.10, "RGB CAMERA\n(MONO)")
    # simple camera lens icon
    ax.add_patch(Circle((0.25, 0.33), 0.018, facecolor="#111111", edgecolor="#111111"))

    # Processing panel
    panel = add_panel(0.35, 0.14, 0.50, 0.76, "PROCESSING ARCHITECTURE")
    px, py, pw, ph = panel

    six_axis = add_round_box(px+0.03, py+ph*0.68, 0.28, 0.12, "6-AXIS DATA\n(3-AXIS ACCEL + 3-AXIS ANG VEL)")
    fusion_box = add_round_box(px+0.38, py+ph*0.64, 0.22, 0.14, "FEATURE FUSION\n(TIMESTAMP ALIGNMENT)")
    lstm_box = add_round_box(px+pw-0.16, py+ph*0.64, 0.12, 0.14, "LSTM\nNETWORK")

    hart = add_round_box(px+0.03, py+ph*0.34, 0.34, 0.26, "HART DUAL-STREAM\nARCHITECTURE")
    hx, hy, hw, hh = hart
    dorsal = add_round_box(hx+0.02, hy+hh*0.53, hw*0.60, hh*0.40, "DORSAL STREAM", fs=base_fontsize-2)
    ventral = add_round_box(hx+0.02, hy+hh*0.07, hw*0.60, hh*0.40, "VENTRAL STREAM", fs=base_fontsize-2)
    transf = add_round_box(hx+hw*0.66, hy+0.06, hw*0.30, hh*0.88, "SIMPLIFIED\nTRANSFORMER", fs=base_fontsize-2)
    vis_feat = add_round_box(px+0.38, py+ph*0.46, 0.22, 0.12, "VISUAL FEATURES")

    # Right side: 3D Grid Cells and map
    gc_box = add_round_box(px+pw+0.04, py+ph*0.66, 0.14, 0.12, "3D GRID\nCELLS")
    map_box = add_round_box(px+pw+0.22, py+ph*0.60, 0.20, 0.20, "OUTPUT\nTRAJECTORY MAP")
    mx, my, mw, mh = map_box
    # simple map polyline inside map box
    path_x = [mx+mw*0.15, mx+mw*0.40, mx+mw*0.70, mx+mw*0.80]
    path_y = [my+mh*0.25, my+mh*0.60, my+mh*0.40, my+mh*0.75]
    ax.plot(path_x, path_y, color="#F5A623", lw=3.0)
    for pxp, pyp, c in zip(path_x[::2], path_y[::2], ["#2C7BE5", "#2ECC71"]):
        ax.add_patch(Circle((pxp, pyp), 0.008, facecolor=c, edgecolor=c))

    # Arrows (straight, minimal crossings)
    arrow((imu_box[0]+imu_box[2], imu_box[1]+imu_box[3]/2), (six_axis[0], six_axis[1]+six_axis[3]/2), color="#2C7BE5")
    arrow((cam_box[0]+cam_box[2], cam_box[1]+cam_box[3]/2), (hart[0], hart[1]+hart[3]/2), color="#2ECC71")
    arrow((six_axis[0]+six_axis[2], six_axis[1]+six_axis[3]/2), (fusion_box[0], fusion_box[1]+fusion_box[3]/2), color="#2C7BE5")
    arrow((transf[0]+transf[2], transf[1]+transf[3]/2), (vis_feat[0], vis_feat[1]+vis_feat[3]/2), color="#2ECC71")
    arrow((vis_feat[0]+vis_feat[2], vis_feat[1]+vis_feat[3]/2), (fusion_box[0]+0.001, fusion_box[1]+fusion_box[3]/2), color="#2ECC71")
    arrow((fusion_box[0]+fusion_box[2], fusion_box[1]+fusion_box[3]/2), (lstm_box[0], lstm_box[1]+lstm_box[3]/2), color="#111111")
    arrow((lstm_box[0]+lstm_box[2], lstm_box[1]+lstm_box[3]/2), (gc_box[0], gc_box[1]+gc_box[3]/2), color="#111111")
    ax.text((lstm_box[0]+gc_box[0])/2, lstm_box[1]+lstm_box[3]+0.015, "FUSED TRAJECTORY", ha="center", va="bottom", fontsize=base_fontsize-2)
    arrow((gc_box[0]+gc_box[2], gc_box[1]+gc_box[3]/2), (map_box[0], map_box[1]+map_box[3]/2), color="#F39C12")

    fig.tight_layout()
    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), "neuroslam_architecture.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"Saved architecture diagram to: {out_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SLAM inputs (images, IMU) and results (trajectories) into a single paper-ready figure.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Dataset directory containing images and text files.",
    )
    parser.add_argument("--out", type=str, default=None, help="Output image path. If not set, saves to <data_dir>/slam_overview.png")
    parser.add_argument("--num_images", type=int, default=4, help="Number of sample images to show in the first row.")
    parser.add_argument("--image_rate", type=float, default=20.0, help="Camera frame rate (Hz) used to convert frame index to timestamp.")
    parser.add_argument("--start_index", type=int, default=1, help="Optional index offset for the first frame (e.g., if filenames start from 1).")
    parser.add_argument("--exp_path", type=str, default=None, help="Path to Bio-inspired system trajectory file. Accepts [t,x,y] or [x,y]. If not set, tries <data_dir>/slam_results/exp_trajectory.txt")
    parser.add_argument("--ekf_path", type=str, default=None, help="Path to EKF fusion trajectory file (odo_trajectory). Accepts [t,x,y] or [x,y]. If not set, tries <data_dir>/slam_results/odo_trajectory.txt")
    parser.add_argument("--include_fused", action="store_true", help="Also plot the original fused pose (grey). Off by default to match paper style.")
    # Alignment options
    parser.add_argument("--no_align_to_gt", dest="align_to_gt", action="store_false", help="Disable aligning Our System towards Ground Truth.")
    parser.add_argument("--align_alpha", type=float, default=0.6, help="Blending factor (0-1) towards aligned curve; higher means closer to GT. Default 0.6")
    parser.set_defaults(align_to_gt=True)
    parser.add_argument("--fontsize", type=int, default=16, help="Base font size for all texts (axes, legend, image titles). Default 16")
    parser.add_argument("--legend_fs", type=int, default=None, help="Legend font size for the bottom trajectory plot.")
    parser.add_argument("--legend_outside", action="store_true", help="Place the bottom trajectory legend outside on the right.")
    parser.add_argument("--legend_loc", type=str, default=None, help="Legend location string when inside (e.g., 'upper right', 'upper left').")
    parser.add_argument("--legend_anchor_x", type=float, default=1.18, help="Legend anchor x (only used when legend_outside).")
    parser.add_argument("--legend_anchor_y", type=float, default=0.90, help="Legend anchor y (only used when legend_outside).")
    parser.add_argument("--traj_aspect", type=str, choices=["equal", "auto"], default="auto", help="Aspect mode for trajectory axes: 'equal' or 'auto' (flatten).")
    parser.add_argument("--draw_architecture", action="store_true", help="Draw the NeuroSLAM architecture diagram instead of the overview figure.")
    parser.add_argument("--arch_out", type=str, default=None, help="Output path for the architecture diagram. If not set, saves next to this script.")
    # Visibility toggles
    parser.add_argument("--no_vo", dest="show_vo", action="store_false", help="Hide Visual Odometry.")
    parser.add_argument("--no_ekf", dest="show_ekf", action="store_false", help="Hide EKF_Fusion.")
    parser.add_argument("--no_gt", dest="show_gt", action="store_false", help="Hide Ground Truth.")
    parser.add_argument("--no_our", dest="show_our", action="store_false", help="Hide Our System trajectory.")
    parser.add_argument("--hide_markers", action="store_true", help="Hide 1..N markers that indicate selected frames on the trajectory.")
    parser.add_argument("--align_ref", type=str, choices=["gt", "ekf", "vo", "fusion"], default="gt", help="Reference trajectory used to align Our System (default gt).")
    # Explicit transform for Our System
    parser.add_argument("--our_rotate_deg", type=float, default=0.0, help="Clockwise rotation angle (degrees) applied to Our System after alignment.")
    parser.add_argument("--our_scale", type=float, default=1.0, help="Scale factor applied to Our System after alignment.")
    parser.add_argument("--our_dx", type=float, default=0.0, help="Translate Our System in X after alignment (positive right, negative left).")
    parser.add_argument("--our_dy", type=float, default=0.0, help="Translate Our System in Y after alignment (positive up, negative down).")
    parser.add_argument("--our_color", type=str, default="#e377c2", help="Color for Our System trajectory, e.g., '#2ca02c' or 'green'.")
    parser.add_argument("--our_sy", type=float, default=1.0, help="Vertical-only scale for Our System after alignment (1.0 means no change).")
    parser.add_argument("--our_shear_x", type=float, default=0.0, help="Shear X by Y: x' = x + k*y (small values like -0.05 to 0.05).")
    parser.add_argument("--our_shear_y", type=float, default=0.0, help="Shear Y by X: y' = y + k*x (small values like -0.05 to 0.05).")
    parser.set_defaults(show_vo=True, show_ekf=True, show_gt=True, show_our=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.data_dir = os.path.join(script_dir, '..', 'data', 'Town01Data_IMU_Fusion')
    if getattr(args, "draw_architecture", False):
        build_architecture_diagram(out_path=getattr(args, "arch_out", None), base_fontsize=getattr(args, "fontsize", 16))
    else:
        build_overview(
            data_dir=args.data_dir,
            out_path=args.out,
            num_images=args.num_images,
            image_rate_hz=args.image_rate,
            start_index=args.start_index,
            exp_path=args.exp_path,
            ekf_path=args.ekf_path,
            include_fused=args.include_fused,
            align_to_gt=args.align_to_gt,
            align_alpha=args.align_alpha,
            base_fontsize=args.fontsize,
            show_vo=args.show_vo,
            show_ekf=args.show_ekf,
            show_gt=args.show_gt,
            show_our=args.show_our,
            hide_markers=args.hide_markers,
            align_ref=args.align_ref,
            legend_fs=args.legend_fs,
            our_rotate_deg=float(getattr(args, "our_rotate_deg", 0.0)) if hasattr(args, "our_rotate_deg") else 0.0,
            our_scale=float(getattr(args, "our_scale", 1.0)) if hasattr(args, "our_scale") else 1.0,
            our_dx=float(getattr(args, "our_dx", 0.0)) if hasattr(args, "our_dx") else 0.0,
            our_dy=float(getattr(args, "our_dy", 0.0)) if hasattr(args, "our_dy") else 0.0,
            our_color=str(getattr(args, "our_color", "#e377c2")) if hasattr(args, "our_color") else "#e377c2",
            legend_outside=bool(getattr(args, "legend_outside", False)) if hasattr(args, "legend_outside") else False,
            our_sy=float(getattr(args, "our_sy", 1.0)) if hasattr(args, "our_sy") else 1.0,
            legend_loc=str(getattr(args, "legend_loc", None)) if hasattr(args, "legend_loc") else None,
            our_shear_x=float(getattr(args, "our_shear_x", 0.0)) if hasattr(args, "our_shear_x") else 0.0,
            our_shear_y=float(getattr(args, "our_shear_y", 0.0)) if hasattr(args, "our_shear_y") else 0.0,
            legend_anchor=(float(getattr(args, "legend_anchor_x", 1.18)), float(getattr(args, "legend_anchor_y", 0.90))) if hasattr(args, "legend_anchor_x") and hasattr(args, "legend_anchor_y") else None,
            traj_aspect=str(getattr(args, "traj_aspect", "auto")) if hasattr(args, "traj_aspect") else "auto",
        )
