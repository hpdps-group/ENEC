import pandas as pd
import os

def calculate_global_decompression_metrics(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    # 读取 CSV 数据
    df = pd.read_csv(csv_path)

    # 1. 过滤：只保留 OP Type 包含 'decomp' 的解压核心算子
    df_decomp = df[df['OP Type'].str.contains('decomp', case=False, na=False)].copy()

    if df_decomp.empty:
        print("未在 CSV 中找到 'decomp' 相关算子数据。")
        return

    # 2. 计算指标
    # datasize_MB 代表解压后的原始数据大小
    total_output_size_mb = df_decomp['datasize_MB'].sum()
    total_output_size_gb = total_output_size_mb / 1024.0

    # 使用 Avg Time(us) 计算单次顺序执行的总耗时
    total_avg_time_s = df_decomp['Avg Time(us)'].sum() / 1e6

    # 计算全局解压吞吐量 (GB/s)
    global_decomp_throughput = total_output_size_gb / total_avg_time_s

    # 3. 构造输出文本
    output_lines = [
        "="*45,
        f" 模型全局解压性能汇总 (基于单次 Avg Time) ",
        "-" * 45,
        f" 解压权重总层数   : {len(df_decomp)}",
        f" 解压后总数据量   : {total_output_size_gb:.3f} GB",
        f" 全局总耗时 (Sum) : {total_avg_time_s:.4f} 秒",
        f" 全局解压吞吐量   : {global_decomp_throughput:.2f} GB/s",
        "="*45
    ]
    output_text = "\n".join(output_lines)

    # 4. 打印到屏幕
    print(output_text)

    # 5. 保存到 TXT 文件
    # 自动生成与 CSV 同名但以 _summary.txt 结尾的文件
    txt_path = csv_path.replace('.csv', '_summary.txt')
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"\n[INFO] 解压分析结果已保存至: {txt_path}")
    except Exception as e:
        print(f"\n[ERROR] 文件保存失败: {e}")

if __name__ == '__main__':
    # 你的实际解压结果 CSV 路径
    csv_file = './results/deepseek-llm-7b-base/models/BF16/deepseek-llm-7b-base/split/deepseek-llm-7b-base_decompress.csv'
    calculate_global_decompression_metrics(csv_file)