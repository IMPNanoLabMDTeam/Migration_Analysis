#!/bin/bash

# ============================================================
# 批量迁移分析脚本
# 功能：自动寻找文件夹中除model1外的所有model作为model2运行分析
# 特性：
#   1. 支持最多10个并行Python进程
#   2. 所有参数通过config.sh配置文件设置
#   3. 自动根据输入文件名确定输出文件名
# ============================================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载配置文件
CONFIG_FILE="${SCRIPT_DIR}/config.sh"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] 配置文件不存在: $CONFIG_FILE"
    echo "[ERROR] 请先创建config.sh配置文件"
    exit 1
fi

echo "============================================================"
echo "加载配置文件: $CONFIG_FILE"
source "$CONFIG_FILE"
echo "配置加载完成"
echo "============================================================"

# 验证必要的配置参数
if [ -z "$MODEL_DIR" ]; then
    echo "[ERROR] MODEL_DIR未设置，请在config.sh中配置"
    exit 1
fi

if [ -z "$MODEL1" ]; then
    echo "[ERROR] MODEL1未设置，请在config.sh中配置"
    exit 1
fi

if [ -z "$PYTHON_SCRIPT" ]; then
    echo "[ERROR] PYTHON_SCRIPT未设置，请在config.sh中配置"
    exit 1
fi

# 检查目录和文件是否存在
if [ ! -d "$MODEL_DIR" ]; then
    echo "[ERROR] 模型目录不存在: $MODEL_DIR"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "[ERROR] Python脚本不存在: $PYTHON_SCRIPT"
    exit 1
fi

# 检查MODEL1文件是否存在
MODEL1_PATH="${MODEL_DIR}/${MODEL1}"
if [ ! -f "${MODEL1_PATH}.data" ]; then
    echo "[ERROR] MODEL1数据文件不存在: ${MODEL1_PATH}.data"
    exit 1
fi

# 创建输出目录和日志目录
mkdir -p "$OUTPUT_DIR"
LOGS_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOGS_DIR"

echo ""
echo "============================================================"
echo "配置信息："
echo "  模型目录: $MODEL_DIR"
echo "  MODEL1: $MODEL1"
echo "  输出目录: $OUTPUT_DIR"
echo "  日志目录: $LOGS_DIR"
echo "  网格大小: $GRID_SIZE"
echo "  创建动画: $CREATE_ANIMATION"
if [ "$CREATE_ANIMATION" = true ]; then
    echo "  动画时长: ${ANIMATION_DURATION}秒"
    echo "  动画帧率: ${ANIMATION_FPS}fps"
    echo "  平滑动画: $SMOOTH_ANIMATION"
fi
echo "  Colorbar最大值: $COLORBAR_MAX"
echo "  最大并行数: $MAX_PARALLEL_JOBS"
echo "============================================================"
echo ""

# 查找所有.data文件（排除MODEL1）
echo "[LOG] 正在搜索模型文件..."
MODEL2_LIST=()
while IFS= read -r -d '' file; do
    # 获取不带路径和扩展名的文件名
    basename=$(basename "$file" .data)
    
    # 排除MODEL1
    if [ "$basename" != "$MODEL1" ]; then
        MODEL2_LIST+=("$basename")
    fi
done < <(find "$MODEL_DIR" -maxdepth 1 -name "*.data" -print0)

# 检查是否找到了model2文件
if [ ${#MODEL2_LIST[@]} -eq 0 ]; then
    echo "[ERROR] 在目录 $MODEL_DIR 中未找到除 $MODEL1 外的其他.data文件"
    exit 1
fi

echo "[LOG] 找到 ${#MODEL2_LIST[@]} 个MODEL2文件："
for model in "${MODEL2_LIST[@]}"; do
    echo "  - $model"
done
echo ""

echo "============================================================"
echo "开始批量处理..."
echo "============================================================"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 初始化计数器
total_tasks=${#MODEL2_LIST[@]}
completed_tasks=0
failed_tasks=0

# 使用pueue进行任务管理和并行处理
if command -v pueue &> /dev/null; then
    echo "[LOG] 使用pueue进行任务管理"
    
    # 检查pueue守护进程是否运行
    if ! pueue status &> /dev/null; then
        echo "[ERROR] pueue守护进程未运行，请先启动: brew services start pueue"
        exit 1
    fi
    
    # 创建一个专门的组用于本次批量分析
    GROUP_NAME="migration_analysis_$(date +%s)"
    pueue group add "$GROUP_NAME"
    
    # 设置该组的并行任务数
    pueue parallel $MAX_PARALLEL_JOBS -g "$GROUP_NAME"
    
    echo "[LOG] 创建任务组: $GROUP_NAME (并行数: $MAX_PARALLEL_JOBS)"
    echo "[LOG] 开始添加 ${#MODEL2_LIST[@]} 个任务到队列..."
    
    # 存储所有任务ID
    TASK_IDS=()
    
    # 为每个model2添加任务到pueue队列
    for model2 in "${MODEL2_LIST[@]}"; do
        file_id="${MODEL1}_to_${model2}"
        log_file="${LOGS_DIR}/log_${file_id}.txt"
        
        # 构建完整命令 - 使用绝对路径
        cmd="cd '$SCRIPT_DIR' && uv run '$PYTHON_SCRIPT'"
        cmd+=" --output-dir '$OUTPUT_DIR'"
        cmd+=" --grid-size $GRID_SIZE"
        cmd+=" --colorbar-max $COLORBAR_MAX"
        cmd+=" --gif-title '$GIF_TITLE'"
        
        if [ "$CREATE_ANIMATION" = true ]; then
            cmd+=" --animation"
            cmd+=" --duration $ANIMATION_DURATION"
            cmd+=" --fps $ANIMATION_FPS"
        fi
        
        if [ "$SMOOTH_ANIMATION" = false ]; then
            cmd+=" --no-smooth"
        fi
        
        cmd+=" '${MODEL_DIR}/${MODEL1}.data'"
        cmd+=" '${MODEL_DIR}/${model2}.data'"
        cmd+=" --file-id '$file_id'"
        cmd+=" >> '$log_file' 2>&1"
        
        # 创建任务日志头部
        {
            echo "============================================================"
            echo "批量分析任务日志"
            echo "============================================================"
            echo "任务ID: $file_id"
            echo "MODEL1: $MODEL1"
            echo "MODEL2: $model2"
            echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "============================================================"
            echo ""
        } > "$log_file"
        
        # 添加任务到pueue（重定向已包含在命令中）
        task_output=$(pueue add -g "$GROUP_NAME" -l "$file_id" -- bash -c "$cmd")
        task_id=$(echo "$task_output" | grep -oE '[0-9]+' | head -1)
        TASK_IDS+=("$task_id")
        
        echo "[LOG] 已添加任务 #$task_id: $MODEL1 -> $model2"
    done
    
    echo ""
    echo "============================================================"
    echo "所有任务已添加到队列"
    echo "任务组: $GROUP_NAME"
    echo "总任务数: ${#TASK_IDS[@]}"
    echo "============================================================"
    echo ""
    echo "[LOG] 监控任务执行状态 (按 Ctrl+C 退出监控，任务会继续在后台运行)..."
    echo "[LOG] 使用 'pueue status -g $GROUP_NAME' 查看任务状态"
    echo "[LOG] 使用 'pueue log <task_id>' 查看任务日志"
    echo ""
    
    # 等待所有任务完成
    echo "[LOG] 等待任务完成..."
    while true; do
        # 获取该组的任务状态
        status_output=$(pueue status -g "$GROUP_NAME" --json)
        
        # 检查是否所有任务都已完成
        running_count=$(echo "$status_output" | grep -c '"status":"Running"' || true)
        queued_count=$(echo "$status_output" | grep -c '"status":"Queued"' || true)
        
        if [ "$running_count" -eq 0 ] && [ "$queued_count" -eq 0 ]; then
            echo ""
            echo "[LOG] 所有任务已完成！"
            break
        fi
        
        # 显示当前状态
        success_count=$(echo "$status_output" | grep -c '"status":"Success"' || true)
        failed_count=$(echo "$status_output" | grep -c '"status":"Failed"' || true)
        echo -ne "\r[进度] 运行中: $running_count | 队列中: $queued_count | 成功: $success_count | 失败: $failed_count | 总计: ${#TASK_IDS[@]}"
        
        sleep 2
    done
    
    echo ""
    echo ""
    
    # 统计结果
    echo "============================================================"
    echo "检查处理结果..."
    echo "============================================================"
    
    completed_tasks=0
    failed_tasks=0
    
    for model2 in "${MODEL2_LIST[@]}"; do
        file_id="${MODEL1}_to_${model2}"
        log_file="${LOGS_DIR}/log_${file_id}.txt"
        
        # 在日志文件末尾添加完成时间和状态
        if [ -f "$log_file" ]; then
            if grep -q "\[LOG\] 所有步骤完成！" "$log_file"; then
                {
                    echo ""
                    echo "============================================================"
                    echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
                    echo "状态: 成功"
                    echo "============================================================"
                } >> "$log_file"
                ((completed_tasks++))
            else
                {
                    echo ""
                    echo "============================================================"
                    echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
                    echo "状态: 失败"
                    echo "============================================================"
                } >> "$log_file"
                ((failed_tasks++))
                echo "[失败] $file_id - 详见: $log_file"
            fi
        else
            ((failed_tasks++))
            echo "[失败] $file_id - 日志文件不存在"
        fi
    done
    
    # 计算总耗时
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(((ELAPSED % 3600) / 60))
    SECONDS=$((ELAPSED % 60))
    
    # 输出汇总信息
    echo ""
    echo "============================================================"
    echo "批量处理完成！"
    echo "============================================================"
    echo "总任务数: ${#MODEL2_LIST[@]}"
    echo "成功: $completed_tasks"
    echo "失败: $failed_tasks"
    printf "总耗时: %02d:%02d:%02d\n" $HOURS $MINUTES $SECONDS
    echo ""
    echo "输出文件位置: $(cd "$OUTPUT_DIR" && pwd)"
    echo "日志文件位置: $(cd "$LOGS_DIR" && pwd)"
    echo "任务组: $GROUP_NAME"
    echo ""
    echo "[提示] 使用 'pueue status -g $GROUP_NAME' 查看详细任务状态"
    echo "[提示] 使用 'pueue clean -g $GROUP_NAME' 清理已完成的任务"
    echo "[提示] 使用 'pueue group remove $GROUP_NAME' 删除任务组"
    echo "============================================================"
    
    # 返回适当的退出码
    if [ $failed_tasks -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
    
else
    echo "[ERROR] 未找到pueue命令"
    echo "[ERROR] 请安装pueue: brew install pueue"
    echo "[ERROR] 然后启动服务: brew services start pueue"
    exit 1
fi
