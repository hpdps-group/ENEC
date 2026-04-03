#!/bin/bash

# 1. 执行数据准备脚本
echo "正在运行 download_data.sh..."
bash download_data.sh

# 2. 运行 Python 切分/工具脚本
echo "正在运行 python/utils.py..."
python3 python/utils.py

echo "全部任务已完成！"