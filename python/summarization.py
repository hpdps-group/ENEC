import os
import re
import pandas as pd
from pathlib import Path

def extract_compress_metrics(filepath):
    """从 compress summary txt 中提取 CR 和 压缩吞吐量 (GB/s)"""
    cr = None
    throughput = None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        cr_match = re.search(r'全局压缩率 \(CR\)\s*:\s*([0-9.]+)', content)
        if cr_match:
            cr = float(cr_match.group(1))
        thr_match = re.search(r'全局吞吐量 \(Speed\)\s*:\s*([0-9.]+)\s*GB/s', content)
        if thr_match:
            throughput = float(thr_match.group(1))
    except Exception as e:
        print(f"读取压缩摘要失败 {filepath}: {e}")
    return cr, throughput

def extract_decompress_metrics(filepath):
    """从 decompress summary txt 中提取 解压缩吞吐量 (GB/s)"""
    throughput = None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        thr_match = re.search(r'全局解压吞吐量\s*:\s*([0-9.]+)\s*GB/s', content)
        if thr_match:
            throughput = float(thr_match.group(1))
    except Exception as e:
        print(f"读取解压摘要失败 {filepath}: {e}")
    return throughput

def main():
    results_root = './results'
    if not os.path.isdir(results_root):
        print(f"结果根目录不存在: {results_root}")
        return

    rows = []
    for dtype in os.listdir(results_root):
        dtype_path = os.path.join(results_root, dtype)
        if not os.path.isdir(dtype_path):
            continue
        for model_name in os.listdir(dtype_path):
            model_path = os.path.join(dtype_path, model_name)
            if not os.path.isdir(model_path):
                continue
            compress_summary = os.path.join(model_path, f"{model_name}_compress_summary.txt")
            decompress_summary = os.path.join(model_path, f"{model_name}_decompress_summary.txt")
            
            cr = None
            compress_throughput = None
            decompress_throughput = None
            
            if os.path.exists(compress_summary):
                cr, compress_throughput = extract_compress_metrics(compress_summary)
            else:
                print(f"警告: 缺失压缩摘要 {compress_summary}")
            
            if os.path.exists(decompress_summary):
                decompress_throughput = extract_decompress_metrics(decompress_summary)
            else:
                print(f"警告: 缺失解压摘要 {decompress_summary}")
            
            rows.append({
                'model_name': model_name,
                'dtype': dtype,
                'compression_ratio_CR': cr,
                'compress_throughput_GBps': compress_throughput,
                'decompress_throughput_GBps': decompress_throughput
            })
    
    if not rows:
        print("未找到任何摘要文件。")
        return
    
    df = pd.DataFrame(rows)
    output_csv = 'summary_all.csv'
    df.to_csv(output_csv, index=False)
    print(f"汇总表格已保存至: {output_csv}")
    print(df)

if __name__ == '__main__':
    main()