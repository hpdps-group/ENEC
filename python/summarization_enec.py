import os
import re
import pandas as pd
from pathlib import Path

def extract_compress_metrics(filepath):
    """Extract CR and Compression Throughput (GB/s) from compress summary txt"""
    cr = None
    throughput = None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        # Search for Global Compression Ratio (CR)
        cr_match = re.search(r'Global Compression Ratio \(CR\)\s*:\s*([0-9.]+)', content)
        if cr_match:
            cr = float(cr_match.group(1))
        # Search for Global Throughput (supports "Speed" or "Speed-Adj")
        thr_match = re.search(r'Global Throughput \(Speed(?:-Adj)?\)\s*:\s*([0-9.]+)\s*GB/s', content)
        if thr_match:
            throughput = float(thr_match.group(1))
    except Exception as e:
        print(f"Failed to read compression summary {filepath}: {e}")
    return cr, throughput

def extract_decompress_metrics(filepath):
    """Extract Decompression Throughput (GB/s) from decompress summary txt"""
    throughput = None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        # Search for Global Decomp Throughput
        thr_match = re.search(r'Global Decomp Throughput\s*:\s*([0-9.]+)\s*GB/s', content)
        if thr_match:
            throughput = float(thr_match.group(1))
    except Exception as e:
        print(f"Failed to read decompression summary {filepath}: {e}")
    return throughput

def main():
    results_root = './results_enec'
    if not os.path.isdir(results_root):
        print(f"Results root directory not found: {results_root}")
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
                cr_raw, thr_raw = extract_compress_metrics(compress_summary)
                cr = round(cr_raw, 2) if cr_raw is not None else None
                compress_throughput = round(thr_raw, 1) if thr_raw is not None else None
            
            if os.path.exists(decompress_summary):
                dthr_raw = extract_decompress_metrics(decompress_summary)
                decompress_throughput = round(dthr_raw, 1) if dthr_raw is not None else None
            
            rows.append({
                'model_name': model_name,
                'dtype': dtype,
                'compression_ratio_CR': cr,
                'compress_throughput_GBps': compress_throughput,
                'decompress_throughput_GBps': decompress_throughput
            })
    
    if not rows:
        print("No summary files found.")
        return
    
    df = pd.DataFrame(rows)

    # --- Logic: Define custom sort order for dtype ---
    dtype_order = ['BF16', 'FP16', 'FP32']
    df['dtype'] = pd.Categorical(df['dtype'], categories=dtype_order, ordered=True)
    
    # Sort priority: first by dtype custom order, then by model_name alphabetically
    df = df.sort_values(by=['dtype', 'model_name'])
    
    output_csv = 'summary_enec.csv'
    df.to_csv(output_csv, index=False)
    print(f"Summary table saved to: {output_csv}")
    print("\n--- Summary Data Preview ---")
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()