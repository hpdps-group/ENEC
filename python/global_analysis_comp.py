# import pandas as pd
# import os

# def calculate_single_pass_metrics(csv_path):
#     if not os.path.exists(csv_path):
#         print(f"文件未找到: {csv_path}")
#         return

#     # 读取 CSV 数据
#     df = pd.read_csv(csv_path)

#     # 1. 核心过滤：只保留 OP Type 包含 'comp' 的行
#     df_comp = df[df['OP Type'].str.contains('comp', case=False, na=False)].copy()

#     if df_comp.empty:
#         print("未在 CSV 中找到 'comp' 相关算子数据。")
#         return

#     # 2. 计算各项指标
#     total_original_size_mb = df_comp['datasize_MB'].sum()
#     total_original_size_gb = total_original_size_mb / 1024.0
#     total_avg_time_s = df_comp['Avg Time(us)'].sum() / 1e6
    
#     # 计算加权压缩率
#     df_comp['compressed_size_mb'] = df_comp['datasize_MB'] / df_comp['cr'].replace(0, 1)
#     total_compressed_size_mb = df_comp['compressed_size_mb'].sum()
#     global_cr = total_original_size_mb / total_compressed_size_mb

#     # 计算全局吞吐量
#     global_throughput = total_original_size_gb / total_avg_time_s

#     # 3. 准备输出文本
#     output_lines = [
#         "="*45,
#         f" 模型全局性能简报 (基于单次 Avg Time) ",
#         "-" * 45,
#         f" 参与计算的权重层数 : {len(df_comp)}",
#         f" 模型原始总大小    : {total_original_size_gb:.3f} GB",
#         f" 压缩后总大小      : {(total_compressed_size_mb/1024.0):.3f} GB",
#         f" 全局压缩率 (CR)   : {global_cr:.4f}",
#         f" 全局吞吐量 (Speed) : {global_throughput:.2f} GB/s",
#         "="*45
#     ]
#     output_text = "\n".join(output_lines)

#     # 4. 打印到屏幕
#     print(output_text)

#     # 5. 保存到 TXT 文件
#     # 自动在 CSV 同级目录下生成同名的 .txt 文件
#     txt_path = csv_path.replace('.csv', '_summary.txt')
#     try:
#         with open(txt_path, 'w', encoding='utf-8') as f:
#             f.write(output_text)
#         print(f"\n[INFO] 结果已成功保存至: {txt_path}")
#     except Exception as e:
#         print(f"\n[ERROR] 文件保存失败: {e}")

# if __name__ == '__main__':
#     # 替换为你实际的 CSV 路径
#     csv_file = './results/deepseek-llm-7b-base/models/BF16/deepseek-llm-7b-base/split/deepseek-llm-7b-base_compress.csv'
#     calculate_single_pass_metrics(csv_file)

import pandas as pd
import os
from pathlib import Path

def calculate_single_pass_metrics(csv_path, output_txt_path):
    if not os.path.exists(csv_path):
        print(f"文件未找到: {csv_path}")
        return False

    df = pd.read_csv(csv_path)

    # 过滤 OP Type 包含 'comp' 的行
    df_comp = df[df['OP Type'].str.contains('comp', case=False, na=False)].copy()

    if df_comp.empty:
        print(f"未在 CSV 中找到 'comp' 相关算子数据: {csv_path}")
        return False

    total_original_size_mb = df_comp['datasize_MB'].sum()
    total_original_size_gb = total_original_size_mb / 1024.0
    total_avg_time_s = df_comp['Avg Time(us)'].sum() / 1e6

    df_comp['compressed_size_mb'] = df_comp['datasize_MB'] / df_comp['cr'].replace(0, 1)
    total_compressed_size_mb = df_comp['compressed_size_mb'].sum()
    global_cr = total_original_size_mb / total_compressed_size_mb
    global_throughput = total_original_size_gb / total_avg_time_s

    output_lines = [
        "=" * 45,
        f" 模型全局性能简报 (基于单次 Avg Time) ",
        "-" * 45,
        f" 参与计算的权重层数 : {len(df_comp)}",
        f" 模型原始总大小    : {total_original_size_gb:.3f} GB",
        f" 压缩后总大小      : {(total_compressed_size_mb / 1024.0):.3f} GB",
        f" 全局压缩率 (CR)   : {global_cr:.4f}",
        f" 全局吞吐量 (Speed) : {global_throughput:.2f} GB/s",
        "=" * 45
    ]
    output_text = "\n".join(output_lines)

    print(f"\n处理文件: {csv_path}")
    print(output_text)

    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"[INFO] 结果已保存至: {output_txt_path}")
    except Exception as e:
        print(f"[ERROR] 文件保存失败: {e}")
        return False
    return True

def main():
    results_root = './results'
    if not os.path.isdir(results_root):
        print(f"结果根目录不存在: {results_root}")
        return

    for dtype in os.listdir(results_root):
        dtype_path = os.path.join(results_root, dtype)
        if not os.path.isdir(dtype_path):
            continue
        for model_name in os.listdir(dtype_path):
            model_path = os.path.join(dtype_path, model_name)
            if not os.path.isdir(model_path):
                continue
            csv_file = os.path.join(model_path, f"{model_name}_compress.csv")
            if os.path.exists(csv_file):
                output_txt = os.path.join(model_path, f"{model_name}_compress_summary.txt")
                calculate_single_pass_metrics(csv_file, output_txt)
            else:
                print(f"跳过 {model_path}，未找到 {model_name}_compress.csv")

if __name__ == '__main__':
    main()