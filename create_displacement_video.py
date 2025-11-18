#!/usr/bin/env python3
"""
位移场动画视频合成脚本 V5 - 向量插值 + 可变速度
--------------------------------------------------
本版本直接读取每个 migration_data_anim_*.txt 文件中的矢量场数据，
以 g100 规格构建 100×100 网格的平均箭头，并在段落之间进行线性插值，
从而实现“箭头旋转/伸缩”的连续过渡效果。图像外观（轴标签、颜色条、网格）
与原始 matplotlib 输出保持一致，仅在画面顶部保留实时物理时间标注。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib

# 使用无窗口后端，避免在服务器/命令行环境中报错
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation, ticker

# ---------------------------
# 常量配置
# ---------------------------
GRID_SIZE = 100  # g100
PLOT_SIZE = 12.0  # p12
COLORBAR_MAX = 1.2  # c1.2
ARROW_WIDTH = 0.003
ARROW_HEADWIDTH = 3.0
ARROW_HEADLENGTH = 5.0
ARROW_HEADAXIS = 4.5
ARROW_SCALE = 1.0  # a1.0
FPS = 30

# ---------------------------
# 文件解析与数据加载
# ---------------------------
def parse_data_filename(filename: str) -> Tuple[int, str, float, float]:
    pattern = r"migration_data_anim_(\d+)_(.+?)_(\d+\.?\d*)ps-(\d+\.?\d*)ps(?:_[^.]*)?\.txt"
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"无法从文件名解析时间信息: {filename}")
    seq_num = int(match.group(1))
    stage = match.group(2)
    start_time = float(match.group(3))
    end_time = float(match.group(4))
    return seq_num, stage, start_time, end_time


def find_data_files(input_dir: Path) -> List[Dict]:
    data_files: List[Dict] = []
    for path in input_dir.glob("migration_data_anim_*.txt"):
        seq_num, stage, start_time, end_time = parse_data_filename(path.name)
        data_files.append({
            "path": path,
            "seq_num": seq_num,
            "stage": stage,
            "start_time": start_time,
            "end_time": end_time,
        })
    data_files.sort(key=lambda info: info["seq_num"])
    if not data_files:
        raise FileNotFoundError(f"在 {input_dir} 中未找到 migration_data_anim_*.txt 文件")
    return data_files


def read_metadata(filepath: Path) -> Dict[str, float]:
    metadata: Dict[str, float] = {}
    with filepath.open("r") as fh:
        for line in fh:
            if line.startswith("# x_min x_max:"):
                parts = line.split(":")[1].strip().split()
                metadata["x_min"], metadata["x_max"] = map(float, parts)
            elif line.startswith("# y_min y_max:"):
                parts = line.split(":")[1].strip().split()
                metadata["y_min"], metadata["y_max"] = map(float, parts)
            elif not line.startswith("#"):
                # 读取到第一行数据即可停止
                break
    if len(metadata) != 4:
        raise ValueError(f"文件 {filepath} 缺少边界元数据")
    return metadata


def read_displacement_data(filepath: Path) -> np.ndarray:
    """读取 x, y, dx, dy 数据 (float32)。"""
    return np.loadtxt(filepath, comments="#", dtype=np.float32)


def bin_vector_field(data: np.ndarray, metadata: Dict[str, float], grid_size: int) -> np.ndarray:
    """将粒子级数据平均到规则网格上，返回 shape=(grid, grid, 2) 的矢量场。"""
    x_min, x_max = metadata["x_min"], metadata["x_max"]
    y_min, y_max = metadata["y_min"], metadata["y_max"]

    # 网格步长（grid_size-1 个间隔）
    dx = (x_max - x_min) / (grid_size - 1)
    dy = (y_max - y_min) / (grid_size - 1)

    x_idx = np.clip(((data[:, 0] - x_min) / dx).astype(np.int32), 0, grid_size - 1)
    y_idx = np.clip(((data[:, 1] - y_min) / dy).astype(np.int32), 0, grid_size - 1)

    vectors = np.zeros((grid_size, grid_size, 2), dtype=np.float32)
    counts = np.zeros((grid_size, grid_size), dtype=np.int32)

    np.add.at(vectors[..., 0], (y_idx, x_idx), data[:, 2])
    np.add.at(vectors[..., 1], (y_idx, x_idx), data[:, 3])
    np.add.at(counts, (y_idx, x_idx), 1)

    mask = counts > 0
    vectors[..., 0][mask] /= counts[mask]
    vectors[..., 1][mask] /= counts[mask]

    return vectors


def load_vector_field(info: Dict, cache_dir: Path, grid_size: int) -> Tuple[np.ndarray, Dict[str, float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"vector_field_{info['seq_num']:03d}.npz"

    if cache_path.exists():
        with np.load(cache_path) as npz:
            vectors = npz["vectors"]
            metadata = {
                "x_min": float(npz["x_min"]),
                "x_max": float(npz["x_max"]),
                "y_min": float(npz["y_min"]),
                "y_max": float(npz["y_max"]),
            }
        return vectors, metadata

    print(f"  解析矢量场: {info['path'].name}")
    metadata = read_metadata(info["path"])
    raw_data = read_displacement_data(info["path"])
    vectors = bin_vector_field(raw_data, metadata, grid_size)

    np.savez_compressed(
        cache_path,
        vectors=vectors,
        x_min=metadata["x_min"],
        x_max=metadata["x_max"],
        y_min=metadata["y_min"],
        y_max=metadata["y_max"],
    )
    print(f"    ✓ 已缓存 {cache_path.name}")
    return vectors, metadata


# ---------------------------
# 时间轴 & 速度设置
# ---------------------------
def get_speed_multiplier(time_ps: float) -> float:
    """根据物理时间返回段落时长倍数（基准 1 秒）。"""
    if time_ps <= 20.0:
        return 2.0
    if time_ps <= 70.0:
        return 4.0
    if time_ps <= 120.0:
        return 2.0
    return 1.0


def build_segments(vectors: List[np.ndarray], infos: List[Dict], fps: int) -> Tuple[List[Dict], List[Tuple[int, int]]]:
    segments: List[Dict] = []

    zero_field = np.zeros_like(vectors[0])
    first_info = infos[0]
    first_mult = get_speed_multiplier(first_info["end_time"])
    first_frames = max(2, int(round(fps * first_mult)))
    segments.append({
        "start": zero_field,
        "end": vectors[0],
        "delta": vectors[0] - zero_field,
        "start_time": first_info["start_time"],
        "end_time": first_info["end_time"],
        "frames": first_frames,
        "label": f"growth_{first_info['seq_num']:03d}",
    })

    for idx in range(len(vectors) - 1):
        current_info = infos[idx]
        next_info = infos[idx + 1]
        multiplier = get_speed_multiplier((next_info["start_time"] + next_info["end_time"]) / 2)
        frame_count = max(2, int(round(fps * multiplier)))
        start_field = vectors[idx]
        end_field = vectors[idx + 1]
        segments.append({
            "start": start_field,
            "end": end_field,
            "delta": end_field - start_field,
            "start_time": next_info["start_time"],
            "end_time": next_info["end_time"],
            "frames": frame_count,
            "label": f"transition_{current_info['seq_num']:03d}_{next_info['seq_num']:03d}",
        })

    frame_map: List[Tuple[int, int]] = []
    for seg_idx, seg in enumerate(segments):
        for local_idx in range(seg["frames"]):
            frame_map.append((seg_idx, local_idx))

    return segments, frame_map


# ---------------------------
# 动画生成
# ---------------------------
def setup_figure(metadata: Dict[str, float]):
    fig, ax = plt.subplots(figsize=(12.0, 10.0), dpi=150)
    ax.set_xlim(metadata["x_min"], metadata["x_max"])
    ax.set_ylim(metadata["y_min"], metadata["y_max"])
    ax.set_xlabel("X Position (nm)", fontsize=12)
    ax.set_ylabel("Y Position (nm)", fontsize=12)
    ax.set_title("")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    Xi = np.linspace(metadata["x_min"], metadata["x_max"], GRID_SIZE)
    Yi = np.linspace(metadata["y_min"], metadata["y_max"], GRID_SIZE)
    mesh_X, mesh_Y = np.meshgrid(Xi, Yi)

    base = np.zeros_like(mesh_X)
    quiver = ax.quiver(
        mesh_X,
        mesh_Y,
        base,
        base,
        base,
        scale=1,
        scale_units="xy",
        angles="xy",
        cmap="plasma",
        alpha=0.9,
        width=ARROW_WIDTH,
        headwidth=ARROW_HEADWIDTH,
        headlength=ARROW_HEADLENGTH,
        headaxislength=ARROW_HEADAXIS,
    )

    cbar = fig.colorbar(quiver, ax=ax, shrink=0.8, pad=0.02)
    cbar.mappable.set_clim(vmin=0, vmax=COLORBAR_MAX)
    cbar.set_label("Displacement Magnitude (nm)", fontsize=12, labelpad=15)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    time_text = fig.text(
        0.5,
        0.96,
        "",
        ha="center",
        va="bottom",
        fontsize=34,
        color="#FFD700",
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, ax, quiver, time_text, mesh_X, mesh_Y


def create_video(segments: List[Dict], frame_map: List[Tuple[int, int]], metadata: Dict[str, float], output_video: Path):
    fig, ax, quiver, time_text, mesh_X, mesh_Y = setup_figure(metadata)

    work_u = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    work_v = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    work_mag = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    total_frames = len(frame_map)

    def update(frame_idx: int):
        nonlocal work_u, work_v, work_mag
        seg_idx, local_idx = frame_map[frame_idx]
        seg = segments[seg_idx]
        frames = max(seg["frames"] - 1, 1)
        alpha = local_idx / frames

        np.copyto(work_u, seg["start"][..., 0])
        np.copyto(work_v, seg["start"][..., 1])
        work_u[...] += seg["delta"][..., 0] * alpha
        work_v[...] += seg["delta"][..., 1] * alpha
        np.hypot(work_u, work_v, out=work_mag)
        np.clip(work_mag, 0, COLORBAR_MAX, out=work_mag)

        quiver.set_UVC(work_u * ARROW_SCALE, work_v * ARROW_SCALE, work_mag)

        current_time = seg["start_time"] + alpha * (seg["end_time"] - seg["start_time"])
        time_text.set_text(f"{current_time:.1f} ps")
        return quiver, time_text

    print(f"\n开始生成动画，共 {total_frames} 帧 ({total_frames / FPS:.2f} 秒 @ {FPS} fps)")
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000 / FPS,
        blit=False,
    )

    writer = animation.FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    anim.save(str(output_video), writer=writer)
    plt.close(fig)
    print(f"✓ 视频已输出: {output_video}")


# ---------------------------
# 主流程
# ---------------------------
def main():
    input_dir = Path("/Users/jiangly/MY_WorkSpace/Migration_Analysis/Results_of_all/g100_a1.0_c1.2_p12/original")
    output_root = Path("/Users/jiangly/MY_WorkSpace/Migration_Analysis/Results_of_all/g100_a1.0_c1.2_p12")
    output_dir = output_root / "video_output_v5_vector"
    frames_cache = output_dir / "vector_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("位移场视频制作 - V5 向量插值 & 可变速度")
    print("=" * 70)
    print(f"输入数据目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"网格规格: g{GRID_SIZE}")

    data_infos = find_data_files(input_dir)
    print(f"共找到 {len(data_infos)} 组数据")

    vector_fields: List[np.ndarray] = []
    metadata: Dict[str, float] | None = None
    for info in data_infos:
        vectors, meta = load_vector_field(info, frames_cache, GRID_SIZE)
        vector_fields.append(vectors)
        if metadata is None:
            metadata = meta
        else:
            for key in ("x_min", "x_max", "y_min", "y_max"):
                if not np.isclose(metadata[key], meta[key], atol=1e-6):
                    raise ValueError(f"文件 {info['path']} 的边界与前一文件不一致")

    assert metadata is not None

    segments, frame_map = build_segments(vector_fields, data_infos, FPS)
    total_frames = len(frame_map)
    print(f"构建时间轴完成，共 {len(segments)} 个段落，{total_frames} 帧")

    output_video = output_dir / "Displacement_field_video_vector_variable_speed.mp4"
    create_video(segments, frame_map, metadata, output_video)

    print("\n总结:")
    print(f"  输出视频: {output_video}")
    print(f"  帧率: {FPS} fps")
    print(f"  总帧数: {total_frames}")
    print(f"  总时长: {total_frames / FPS:.2f} 秒")


if __name__ == "__main__":
    main()
