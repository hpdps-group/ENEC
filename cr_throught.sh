#!/bin/bash

# 确保脚本在遇到错误时立即停止
set -e

echo "开始执行全流程数据压缩与分析脚本..."

# 1. 执行模型压缩
echo "步骤 1/5: 正在进行模型压缩 (enec_model_compress.py)..."
python python/enec_model_compress.py

# 2. 执行模型解压
echo "步骤 2/5: 正在进行模型解压 (enec_model_decompress.py)..."
python python/enec_model_decompress.py

# 3. 压缩阶段全局分析
echo "步骤 3/5: 正在分析压缩性能 (global_analysis_comp.py)..."
python python/global_analysis_comp.py

# 4. 解压阶段全局分析
echo "步骤 4/5: 正在分析解压性能 (global_analysis_decomp.py)..."
python python/global_analysis_decomp.py

# 5. 生成最终总结报告
echo "步骤 5/5: 正在生成实验总结 (summarization.py)..."
python python/summarization.py

echo "-------------------------------------------"
echo "所有任务已成功完成！"