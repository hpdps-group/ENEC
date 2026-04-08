import pandas as pd
import os
from pathlib import Path

def calculate_global_decompression_metrics(csv_path, output_txt_path):
    """
    Analyzes decompression performance from CSV.
    Returns: (Success, Global Throughput GB/s, Total Output Size GB)
    """
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return False, 0.0, 0.0

    df = pd.read_csv(csv_path)

    # Filter for decompression operators
    df_decomp = df[df['OP Type'].str.contains('decomp', case=False, na=False)].copy()

    if df_decomp.empty:
        print(f"No 'decomp' related operator data found in CSV: {csv_path}")
        return False, 0.0, 0.0

    # Ensure numeric types for calculation
    df_decomp['Avg Time(us)'] = df_decomp['Avg Time(us)'].astype(float)
    df_decomp['Max Time(us)'] = df_decomp['Max Time(us)'].astype(float)
    df_decomp['datasize_MB'] = df_decomp['datasize_MB'].astype(float)

    # Adjusted time calculation (Floating point)
    df_decomp['adjusted_time_us'] = (df_decomp['Avg Time(us)'] * 4.0 - df_decomp['Max Time(us)']) / 3.0
    
    total_avg_time_s = float(df_decomp['adjusted_time_us'].sum()) / 1e6
    total_output_size_gb = float(df_decomp['datasize_MB'].sum()) / 1024.0
    
    global_decomp_throughput = total_output_size_gb / total_avg_time_s if total_avg_time_s > 0 else 0.0

    output_lines = [
        "=" * 50,
        " GLOBAL DECOMPRESSION PERFORMANCE SUMMARY ",
        "-" * 50,
        f" Total Layers Processed    : {len(df_decomp)}",
        f" Total Output Data Size    : {total_output_size_gb:.6f} GB",
        f" Total Execution Time (Adj): {total_avg_time_s:.6f} seconds",
        f" Global Decomp Throughput  : {global_decomp_throughput:.4f} GB/s",
        "=" * 50
    ]
    output_text = "\n".join(output_lines)

    print(f"\n[Analysing] {csv_path}")
    print(output_text)

    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"[INFO] Performance summary saved to: {output_txt_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save summary file: {e}")
        return False, 0.0, 0.0

    return True, float(global_decomp_throughput), float(total_output_size_gb)

def compute_inference_latency(model_name, dtype, decomp_throughput_gbps, model_size_gb, baseline_data, time_per_layer, num_of_layer, cal_time):
    """
    Computes estimated inference latency and speedup using ENEC.
    Ensures all calculations use floating-point precision.
    """
    model_size_gb = float(model_size_gb)
    if model_size_gb <= 0:
        print(f"Warning: Model size is zero for {model_name}. Skipping calculation.")
        return

    all_layer, npu_layer = [float(x) for x in num_of_layer]
    baseline_ttft, baseline_tpot = [float(x) for x in baseline_data]
    base_comm_ttft, base_comm_tpot = [float(x) for x in time_per_layer]

    model_size_per_layer = model_size_gb / all_layer
    new_decomp_time_per_layer = model_size_per_layer / float(decomp_throughput_gbps)

    cal_time_ttft, cal_time_tpot = cal_time

    enec_latency_ttft = cal_time_ttft + ((new_decomp_time_per_layer - base_comm_ttft) * all_layer)
    enec_latency_tpot = cal_time_tpot + ((new_decomp_time_per_layer - base_comm_tpot) * all_layer)

    speedup_ttft = baseline_ttft / enec_latency_ttft if enec_latency_ttft > 0 else 0.0
    speedup_tpot = baseline_tpot / enec_latency_tpot if enec_latency_tpot > 0 else 0.0

    result = {
        'model': model_name,
        'dtype': dtype,
        'batch_size': 1,
        'baseline_TTFT_s': round(baseline_ttft, 6),
        'baseline_TPOT_s': round(baseline_tpot, 6),
        'ENEC_TTFT_s': round(enec_latency_ttft, 6),
        'ENEC_TPOT_s': round(enec_latency_tpot, 6),
        'speedup_TTFT': round(speedup_ttft, 4),
        'speedup_TPOT': round(speedup_tpot, 4)
    }

    output_csv = f'Latency_{model_name}_{dtype}.csv'
    df_out = pd.DataFrame([result])
    df_out.to_csv(output_csv, index=False)

    print(f"\n[Inference: {model_name}]")
    print(f"  Configuration: size={model_size_gb:.2f} GB, throughput={decomp_throughput_gbps:.2f} GB/s")
    print(f"  baseline TTFT: {baseline_ttft} s")
    print(f"  baseline TPOT: {baseline_tpot} s")
    print(f"  ENEC TTFT: {enec_latency_ttft:.6f} s (Speedup: {speedup_ttft:.2f}x)")
    print(f"  ENEC TPOT: {enec_latency_tpot:.6f} s (Speedup: {speedup_tpot:.2f}x)")
    print(f"  Result saved to: {output_csv}")

def main():

    BASELINE = {
        'Qwen3-32B': (2.36064, 1.1951),
        'falcon-40b': (2.608317, 1.150164),
    }

    TIME_PER_LAYER = {
        'Qwen3-32B': (0.0032, 0.0025),
        'falcon-40b': (0.0027, 0.0025)
    }

    NUM_OF_LAYER = {
        'Qwen3-32B': (64, 54),
        'falcon-40b': (60, 40)
    }
    CAL_TIME  = {
        'Qwen3-32B': (0.48, 0.24),
        'falcon-40b': (0.29, 0.12)
    }

    results_root = './results_enec'
    if not os.path.isdir(results_root):
        print(f"Error: Results root directory not found: {results_root}")
        return

    for dtype in os.listdir(results_root):
        dtype_path = os.path.join(results_root, dtype)
        if not os.path.isdir(dtype_path):
            continue
            
        for model_name in os.listdir(dtype_path):
            model_path = os.path.join(dtype_path, model_name)
            if not os.path.isdir(model_path):
                continue

            if model_name not in BASELINE:
                continue

            csv_file = os.path.join(model_path, f"{model_name}_decompress.csv")
            if not os.path.exists(csv_file):
                print(f"Skipping {model_name}: decompress.csv not found in {model_path}")
                continue

            # Summary file path
            summary_path = os.path.join(model_path, f"{model_name}_decompress_summary.txt")
            
            success, decomp_throughput, total_output_size_gb = calculate_global_decompression_metrics(
                csv_file, summary_path
            )

            if success and decomp_throughput > 0:
                compute_inference_latency(
                    model_name, dtype, decomp_throughput, total_output_size_gb,
                    BASELINE[model_name], TIME_PER_LAYER[model_name], NUM_OF_LAYER[model_name],
                    CAL_TIME[model_name]
                )
            else:
                print(f"Error: Could not calculate valid throughput for {model_name}")

if __name__ == '__main__':
    main()