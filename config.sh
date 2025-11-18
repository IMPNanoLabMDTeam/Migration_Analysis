#!/bin/bash
# 配置文件 - 所有参数在此设置

# ============== 基本路径配置 ==============
# 模型文件所在文件夹
MODEL_DIR="/Users/jiangly/MY_WorkSpace/Migration_Analysis/raw_data/Cu60_ds=5"

# 指定的model1文件名（不带.data扩展名）
MODEL1="KAPTON10_3168_CHON2019_irradiated2_ds=5"

# 输出目录
OUTPUT_DIR="results"

# ============== Python脚本路径 ==============
PYTHON_SCRIPT="migration_analysis.py"

# ============== 绘图参数配置 ==============
# 网格大小
GRID_SIZE=100

# 是否创建动画 (true/false)
CREATE_ANIMATION=true

# 动画时长（秒）
ANIMATION_DURATION=3.0

# 动画帧率
ANIMATION_FPS=30

# 是否使用平滑动画效果 (true/false)
# true: 使用平滑效果, false: 使用--no-smooth参数
SMOOTH_ANIMATION=true

# Colorbar最大值
COLORBAR_MAX=1.2

# GIF标题
GIF_TITLE="2D Displacement Vector Field - Growing Animation"

# ============== 并行配置 ==============
# 最大并行进程数
MAX_PARALLEL_JOBS=10
