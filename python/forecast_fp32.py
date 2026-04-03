import pandas as pd
import sys
import argparse
import os

def calculate_model_compression_stats(file_path):
    # 1. 读取表格文件
    # 如果你的文件是用制表符(Tab)分隔的，请使用 sep='\t'
    # 如果是标准 CSV，请使用 sep=','
    try:
        # 先尝试读取制表符分隔（符合你之前贴出的格式）
        df = pd.read_csv(file_path, sep='\t')
        if len(df.columns) < 8:
            # 如果列数不对，尝试用逗号分隔读取
            df = pd.read_csv(file_path, sep=',')
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 2. 核心计算逻辑
    # 计算每一层的总比特数：num_elements * average_bit_length
    df['layer_total_bits'] = df['num_elements'] * df['average_bit_length']
    
    # 计算原始 FP32 总比特数 (16 bits/element)
    df['original_bf16_bits'] = df['num_elements'] * 32

    # 3. 汇总结果
    total_compressed_bits = df['layer_total_bits'].sum()
    total_original_bits = df['original_bf16_bits'].sum()
    total_elements = df['num_elements'].sum()

    # 单位换算 (Bits -> MB)
    compressed_size_mb = total_compressed_bits / (8 * 1024 * 1024)
    original_size_mb = total_original_bits / (8 * 1024 * 1024)
    
    # 计算整体压缩率 (CR)
    compression_ratio = total_original_bits / total_compressed_bits if total_compressed_bits > 0 else 0
    # 计算整体平均位宽
    overall_avg_bit = total_compressed_bits / total_elements if total_elements > 0 else 0

    avg_compression_ratio_formula = 32 / (24 + overall_avg_bit) if (8 + overall_avg_bit) > 0 else 0

    # 4. 打印输出
    print("="*50)
    print(f"{'模型压缩统计结果':^46}")
    print("="*50)
    print(f"处理文件:           {os.path.basename(file_path)}")
    print(f"总元素个数:         {total_elements:,}")
    print("-" * 50)
    print(f"原始 FP32 总容量:    {original_size_mb:>10.2f} MB")
    print(f"ENEC 压缩后总容量:   {compressed_size_mb:>10.2f} MB")
    print(f"整体压缩率 (CR):     {compression_ratio:>10.2f}x")
    print(f"全模型平均位宽:      {overall_avg_bit:>10.4f} bits/element")
    print(f"公式平均压缩率*:     {avg_compression_ratio_formula:>12.2f} x")
    print("="*50)

    # 可选：将计算明细保存到新文件
    # df.to_csv("compression_details.csv", index=False)

if __name__ == "__main__":
    # 1. 创建解析器
    parser = argparse.ArgumentParser(description="计算模型压缩统计数据")
    
    # 2. 添加位置参数 (positional argument)
    # help 是提示信息，nargs='?' 表示该参数可选（如果没传则使用 default）
    parser.add_argument("file_path", 
                        nargs='?',
                        default='../param_after_search/hyperparams_results.csv', 
                        help="CSV 结果文件的路径")

    # 3. 解析命令行参数
    args = parser.parse_args()

    # 4. 检查文件是否存在
    if not os.path.exists(args.file_path):
        print(f"Error: 找不到文件 '{args.file_path}'，请确认路径是否正确。")
        sys.exit(1)

    # 执行主函数
    calculate_model_compression_stats(args.file_path)