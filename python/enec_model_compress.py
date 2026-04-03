import warnings
import os
import torch
import torch_npu
import time
import pandas as pd
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
data_type = 'models'  
operation = 'compress'  # optional：compress, decompress
from logger import LoggerGenerator
log_directory = './logs/comp'
logger = LoggerGenerator.get_logger(log_directory, name=f"{operation} {data_type}", console_output=True)

warnings.filterwarnings("ignore", message=".*was not compiled with torchair.*")

def get_file(path, file_name):
    try:
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if not os.path.isdir(file_path) and "op_statistic" in file:
                file_name.append(file_path)
            else:
                get_file(file_path, file_name)
    except:
        pass

def prof_print(profiler_result_path, output_file_dir, datasize_GB, test_times, model_name, param_name, cr):
    file_name = []
    get_file(profiler_result_path, file_name)
    if len(file_name) == 0:
        with open(f'{output_file_dir}/{model_name}_{operation}_error.log', 'a') as f:
            f.write(f'No profiling result found for {model_name} {param_name} in {profiler_result_path}\n')
        return

    results = pd.read_csv(file_name[0])
    
    if 'OP Type' in results.columns:
        results = results[results['OP Type'].str.contains('comp', na=False, case=False)]
        if results.empty:
            logger.warning(f"No 'comp' operations found for {param_name}, skipping.")
            return
    else:
        logger.warning(f"Column 'OP Type' not found in profiling results for {param_name}, skipping filter.")
    
    results['Parameter_Name'] = param_name
    results['datasize_MB'] = datasize_GB * 1024
    results['Speed(GB/S)'] = datasize_GB * test_times / (results['Avg Time(us)'] / 1000 / 1000)
    results['cr'] = cr

    output_file = f'{output_file_dir}/{model_name}_{operation}.csv'
    if os.path.exists(output_file):
        results.to_csv(output_file, mode='a', header=False, index=False)
    else:
        results.to_csv(output_file, mode='w', header=True, index=False)
    logger.info(f'results: \n{results}')

def enec_test(model_name, file_path, dtype, results_dir, operation):
    import subprocess

    # 结果根目录: results/{dtype}/{model_name}
    result_path_dir = Path(results_dir) / dtype / model_name
    os.makedirs(result_path_dir, exist_ok=True)
    file_name = Path(file_path).stem
    file_size = os.path.getsize(file_path)

    compressed_file_path = result_path_dir / 'compressed' / f'{file_name}.compressed'
    decompressed_file_path = result_path_dir / 'decompressed' / f'{file_name}.decompressed'
    os.makedirs(result_path_dir / 'compressed', exist_ok=True)
    os.makedirs(result_path_dir / 'decompressed', exist_ok=True)

    times = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    msprof_path = f'./prof_path_enec/tmp/model/prof_{model_name}/{operation}/{file_name}/{times}'

    csv_param_name = file_name

    # 超参数 CSV 路径: param_search/{dtype}/{model_name}/hyperparams_results.csv
    # hyperparams_csv_path = f'./param_search/{dtype}/{model_name}/hyperparams_results.csv'
    hyperparams_csv_path = f'/data/yjw/results_data/param_search/{dtype}/{model_name}/hyperparams_results.csv'
    if not os.path.exists(hyperparams_csv_path):
        logger.error(f"Hyperparams CSV not found: {hyperparams_csv_path}")
        return

    df = pd.read_csv(hyperparams_csv_path)
    matching_rows = df[df['parameter_name'] == csv_param_name]

    if not matching_rows.empty:
        row = matching_rows.iloc[0]
        b_val = int(row['b'])
        n_val = int(row['n'])
        m_val = int(row['m'])
        L_val = int(row['L'])
        dtype_flag = {'BF16': 0, 'FP16': 1, 'FP32': 2}.get(dtype.upper(), 0)
        folder_name = f"ENEC-{dtype.upper()}-{b_val}-{n_val}-{m_val}-{L_val}"
        exec_file_dir = f"./csrc/{dtype}/{folder_name}/build"

        if not os.path.exists(exec_file_dir):
            logger.error(f"Compressor directory not found: {exec_file_dir}")
            return
        logger.info(f"Matched param {csv_param_name} -> {exec_file_dir}, dtype_flag={dtype_flag}")
    else:
        logger.error(f"Parameter {csv_param_name} not found in {hyperparams_csv_path}")
        return

    if operation == 'compress':
        command = [
            "msprof",
            "--output=" + msprof_path,
            f"{exec_file_dir}/compress",
            file_path,
            str(compressed_file_path),
            str(file_size),
            str(L_val),
            str(dtype_flag)
        ]
    elif operation == 'decompress':
        command = [
            "msprof",
            "--output=" + msprof_path,
            f"{exec_file_dir}/decompress",
            str(compressed_file_path),
            str(decompressed_file_path),
            file_path
        ]
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout

    cr_value = None
    for line in output.splitlines():
        if "cr:" in line:
            parts = line.split("cr:", 1)
            if len(parts) > 1:
                cr_value = parts[1].strip()
                break
    cr_float = float(cr_value) if cr_value is not None else -1.0

    prof_print(msprof_path, result_path_dir, file_size / (1024 * 1024 * 1024), 1, model_name, file_name, cr_float)

    if result.returncode != 0:
        logger.error(f"Command {' '.join(command)} failed with return code {result.returncode}")

def main():
    from tqdm import tqdm
    dtypes = ['FP32', 'FP16', 'BF16']
    base_models_dir = '/data/yjw/models'
    results_base_dir = './results'
    error_files = []
    MIN_FILE_SIZE = 32768  # 32KB

    for dtype in dtypes:
        dtype_path = os.path.join(base_models_dir, dtype)
        if not os.path.isdir(dtype_path):
            logger.warning(f"Directory {dtype_path} not found, skipping.")
            continue

        for model_name in os.listdir(dtype_path):
            split_dir = os.path.join(dtype_path, model_name, 'split')
            if not os.path.isdir(split_dir):
                logger.info(f"No split directory for {model_name} ({dtype}), skipping.")
                continue

            logger.info(f"Processing model {model_name} ({dtype})")
            for root, dirs, files in os.walk(split_dir):
                for file in files:
                    if file.endswith(('.dat', '.bin', '.weight')):
                        file_path = os.path.join(root, file)
                        param_name = Path(file).stem
                        # 检查是否已压缩
                        result_subdir = Path(results_base_dir) / dtype / model_name
                        compressed_file = result_subdir / 'compressed' / f'{param_name}.compressed'
                        if compressed_file.exists():
                            logger.info(f"Skipping {file_path}, already compressed.")
                            continue

                        # 检查文件大小，小于阈值则跳过
                        file_size = os.path.getsize(file_path)
                        if file_size < MIN_FILE_SIZE:
                            logger.info(f"Skipping {file_path} (size={file_size} bytes < {MIN_FILE_SIZE} bytes) - too small to compress.")
                            continue

                        try:
                            logger.info(f"Processing {file_path} with dtype {dtype}")
                            enec_test(model_name, file_path, dtype, results_base_dir, operation)
                        except Exception as e:
                            error_msg = f"Error processing {file_path}: {e}"
                            logger.error(error_msg)
                            error_files.append((file_path, dtype, str(e)))

    if error_files:
        logger.error("Some files failed during testing:")
        for file_path, dtype, err in error_files:
            logger.error(f"File: {file_path}, Dtype: {dtype}, Error: {err}")
            
if __name__ == '__main__':
    main()