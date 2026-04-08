import warnings
import os
import torch
import torch_npu
import time
import pandas as pd
import subprocess
from pathlib import Path
from logger import LoggerGenerator

# --- Configuration ---
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*was not compiled with torchair.*")
data_type = 'models'  
operation = 'compress'  # Options: compress, decompress
log_directory = './logs/comp'
logger = LoggerGenerator.get_logger(log_directory, name=f"{operation}_{data_type}", console_output=True)

# Global Cache: Stores hyperparameter mapping for each model to avoid redundant disk I/O
# Structure: {(dtype, model_name): DataFrame}
CSV_CACHE = {}

def get_hyperparams(dtype, model_name, param_name):
    """Load and retrieve hyperparameters from cache or disk"""
    key = (dtype, model_name)
    if key not in CSV_CACHE:
        csv_path = f'./param_search_enec/{dtype}/{model_name}/hyperparams_results.csv'
        if os.path.exists(csv_path):
            logger.info(f"Initial loading of parameter table: {csv_path}")
            CSV_CACHE[key] = pd.read_csv(csv_path)
        else:
            logger.error(f"Hyperparameter file not found: {csv_path}")
            return None
    
    df = CSV_CACHE[key]
    match = df[df['parameter_name'] == param_name]
    return match.iloc[0] if not match.empty else None

def get_file(path, file_list):
    """Recursively retrieve profiling result files"""
    try:
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if not os.path.isdir(file_path) and "op_statistic" in file:
                file_list.append(file_path)
            else:
                get_file(file_path, file_list)
    except: 
        pass

def prof_print(profiler_result_path, output_file_dir, datasize_GB, test_times, model_name, param_name, cr):
    """Parse and record profiling performance data (Optimized by excluding max execution time)"""
    file_list = []
    get_file(profiler_result_path, file_list)
    if not file_list:
        with open(f'{output_file_dir}/{model_name}_{operation}_error.log', 'a') as f:
            f.write(f'No profiling result for {model_name} {param_name} in {profiler_result_path}\n')
        return

    # Read the first relevant profiling CSV file
    results = pd.read_csv(file_list[0])
    
    # Filter core operator types
    if 'OP Type' in results.columns:
        # Match 'comp' for compress or decompress
        results = results[results['OP Type'].str.contains('comp', na=False, case=False)]
        if results.empty: 
            logger.warning(f"[{param_name}] No core operator timing data found in profiling")
            return

    # --- Logic: Calculate average throughput excluding the maximum value ---
    # Formula: Adjusted_Time = (Avg_Time * Count - Max_Time) / (Count - 1)
    if test_times > 1:
        avg_time_us = results['Avg Time(us)']
        max_time_us = results['Max Time(us)']
        
        # Calculate single iteration average time (microseconds) excluding the maximum
        adjusted_avg_time_us = (avg_time_us * test_times - max_time_us) / (test_times - 1)
        # Convert to seconds (s)
        duration_s = adjusted_avg_time_us / 1000 / 1000
    else:
        # If test_times is 1, use raw average time
        duration_s = results['Avg Time(us)'] / 1000 / 1000

    results['Parameter_Name'] = param_name
    results['datasize_MB'] = datasize_GB * 1024
    
    # Calculate adjusted Speed (GB/S)
    results['Speed(GB/S)'] = datasize_GB / duration_s
    results['cr'] = cr

    # Write results to CSV
    output_file = f'{output_file_dir}/{model_name}_{operation}.csv'
    results.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    
    # Log the result with two decimal places
    current_speed = results["Speed(GB/S)"].iloc[0]
    logger.info(f'[{param_name}] Speed(Adj): {current_speed:.2f} GB/s | CR: {cr:.2f}')

def enec_test(model_name, file_path, dtype, results_dir, operation):
    file_name_stem = Path(file_path).stem
    file_size = os.path.getsize(file_path)
    
    # 1. Quick hyperparameter lookup (via memory cache)
    row = get_hyperparams(dtype, model_name, file_name_stem)
    if row is None:
        logger.error(f"Parameter {file_name_stem} not matched in CSV")
        return

    # Extract hyperparams
    b_val, n_val, m_val, L_val = int(row['b']), int(row['n']), int(row['m']), int(row['L'])
    dtype_flag = {'BF16': 0, 'FP16': 1, 'FP32': 2}.get(dtype.upper(), 0)
    
    # 2. Locate the renamed executable in csrc/exec
    # Corresponds to naming format: compress_ENEC-BF16-121-6-3-16
    suffix = f"ENEC-{dtype.upper()}-{b_val}-{n_val}-{m_val}-{L_val}"
    exec_filename = f"{operation}_{suffix}"
    exec_path = f"./csrc/exec/{exec_filename}"

    if not os.path.exists(exec_path):
        logger.error(f"Executable not found: {exec_path} (Verify shell extraction script)")
        return

    # 3. Prepare paths
    result_path_dir = Path(results_dir) / dtype / model_name
    os.makedirs(result_path_dir / 'compressed', exist_ok=True)
    os.makedirs(result_path_dir / 'decompressed', exist_ok=True)
    
    compressed_file = result_path_dir / 'compressed' / f'{file_name_stem}.compressed'
    decompressed_file = result_path_dir / 'decompressed' / f'{file_name_stem}.decompressed'
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    msprof_out = f'./prof_path_enec/tmp/model/prof_{model_name}/{operation}/{file_name_stem}/{timestamp}'

    # 4. Construct Command
    if operation == 'compress':
        command = ["msprof", "--output=" + msprof_out, exec_path, file_path, str(compressed_file), str(file_size), str(L_val), str(dtype_flag)]
    else:
        command = ["msprof", "--output=" + msprof_out, exec_path, str(compressed_file), str(decompressed_file), file_path]

    # 5. Execute and Parse
    result = subprocess.run(command, capture_output=True, text=True)
    cr_float = -1.0
    for line in result.stdout.splitlines():
        if "cr:" in line:
            cr_float = float(line.split("cr:", 1)[1].strip())
            break

    prof_print(msprof_out, result_path_dir, file_size / (1024**3), 4, model_name, file_name_stem, cr_float)

def main():
    dtypes = ['FP32', 'FP16', 'BF16']
    base_models_dir = './models'
    results_base_dir = './results_enec'
    MIN_FILE_SIZE = 32768  # 32KB

    for dtype in dtypes:
        dtype_path = os.path.join(base_models_dir, dtype)
        if not os.path.isdir(dtype_path): 
            continue

        for model_name in os.listdir(dtype_path):
            split_dir = os.path.join(dtype_path, model_name, 'split')
            if not os.path.isdir(split_dir): 
                continue

            logger.info(f">>> Processing Model: {model_name} ({dtype})")
            
            # Traverse weight files
            for root, _, files in os.walk(split_dir):
                for file in files:
                    if file.endswith(('.bin', '.weight')):
                        file_path = os.path.join(root, file)
                        param_name = Path(file).stem
                        
                        # Skip logic
                        if (Path(results_base_dir)/dtype/model_name/'compressed'/f'{param_name}.compressed').exists():
                            continue
                        if os.path.getsize(file_path) < MIN_FILE_SIZE:
                            continue

                        try:
                            enec_test(model_name, file_path, dtype, results_base_dir, operation)
                        except Exception as e:
                            logger.error(f"Failed to process {file}: {e}")

if __name__ == '__main__':
    main()