import pandas as pd
import os
from pathlib import Path

def calculate_single_pass_metrics(csv_path, output_txt_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return False

    df = pd.read_csv(csv_path)

    # 1. Filter: Only keep rows where 'OP Type' contains 'comp'
    df_comp = df[df['OP Type'].str.contains('comp', case=False, na=False)].copy()

    if df_comp.empty:
        print(f"No 'comp' related operator data found in CSV: {csv_path}")
        return False

    # --- Core Logic: Adjusted Time Calculation ---
    # Logic: Assuming 4 test runs, exclude the maximum value and average the remaining 3.
    # Formula: Adjusted_Time_us = (Avg * 4 - Max) / 3
    df_comp['adjusted_time_us'] = (df_comp['Avg Time(us)'] * 4 - df_comp['Max Time(us)']) / 3
    
    # Calculate total adjusted execution time for the model (seconds)
    total_avg_time_s = df_comp['adjusted_time_us'].sum() / 1e6
    # --------------------------------------------

    # Basic physical metric calculations
    total_original_size_mb = df_comp['datasize_MB'].sum()
    total_original_size_gb = total_original_size_mb / 1024.0

    # Calculate compressed size for each layer (to derive Global CR)
    df_comp['compressed_size_mb'] = df_comp['datasize_MB'] / df_comp['cr'].replace(0, 1)
    total_compressed_size_mb = df_comp['compressed_size_mb'].sum()
    
    # Global metrics summary
    global_cr = total_original_size_mb / total_compressed_size_mb if total_compressed_size_mb != 0 else 0
    # Throughput = Total Original Size / Total Adjusted Time
    global_throughput = total_original_size_gb / total_avg_time_s if total_avg_time_s != 0 else 0

    # Generate report content
    output_lines = [
        "=" * 45,
        " Model Global Performance Summary (Adj. Time) ",
        "-" * 45,
        f" Number of Layers Processed : {len(df_comp)}",
        f" Total Original Model Size  : {total_original_size_gb:.3f} GB",
        f" Total Compressed Model Size: {(total_compressed_size_mb / 1024.0):.3f} GB",
        f" Global Compression Ratio (CR) : {global_cr:.4f}",
        f" Global Throughput (Speed-Adj) : {global_throughput:.2f} GB/s",
        f" Total Execution Time (Sum-Adj): {total_avg_time_s:.4f} s",
        "=" * 45
    ]
    output_text = "\n".join(output_lines)

    print(f"\nProcessing file: {csv_path}")
    print(output_text)

    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"[INFO] Adjusted result saved to: {output_txt_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save file: {e}")
        return False
    return True

def main():
    results_root = './results_enec'
    if not os.path.isdir(results_root):
        print(f"Results root directory not found: {results_root}")
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
                print(f"Skipping {model_path}, could not find {model_name}_compress.csv")

if __name__ == '__main__':
    main()