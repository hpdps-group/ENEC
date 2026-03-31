# 设置变量方便脚本使用
BASE_PATH="/data/yjw/kvtensors/kvtensors/Qwen3-8B-bf16/Qwen3-8B"
OUTPUT_DIR="eneclog"
TMP_FILE="tmp.bin"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 查找BASE_PATH下所有的.bin文件
bin_files=($(find "$BASE_PATH" -name "k*.bin" -type f))

# 使用循环执行所有找到的.bin文件
device_id=4
for input_file in "${bin_files[@]}"; do
    # 从文件名提取基本名称（不含路径和扩展名）
    base_name=$(basename "$input_file" .bin)
    output_log="${OUTPUT_DIR}/${base_name}.log"
    
    # ASCEND_RT_VISIBLE_DEVICES=$device_id python model.py "$input_file" > "$output_log" 2>&1 &

    ASCEND_RT_VISIBLE_DEVICES=$device_id ./build/compress "$input_file" "$TMP_FILE" $(stat -c%s "$input_file") > "$output_log" 2>&1 &
    
    # 递增设备ID，确保不同任务使用不同设备
    ((device_id++))
done