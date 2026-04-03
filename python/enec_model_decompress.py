# import torch
# import torch_npu
# import time
# import os
# import pandas as pd
# from transformers import AutoModelForCausalLM,BertModel,CLIPModel,LlamaForCausalLM,Wav2Vec2ForCTC
# from pathlib import Path
# from utils import get_result_path


# import warnings
# warnings.filterwarnings("ignore")
# # from test_case import setup_seed
# data_type = 'models'  
# operation = 'decompress'  #optional：compress, decompress
# from logger import LoggerGenerator
# log_directory = './logs/decomp'
# logger = LoggerGenerator.get_logger(log_directory, name=f"{operation} {data_type}", console_output=True)


# def get_file(path,file_name):
#     try:
#         files = os.listdir(path)
#         for file in files:
#             file_path = os.path.join(path,file)
#             if not os.path.isdir(file_path) and "op_statistic" in file:
#                 file_name.append(file_path)
#             else:
#                 get_file(file_path,file_name)
#     except:
#         pass

# # test_times: no use,set to 1
# def prof_print(profiler_result_path, output_file_dir, datasize_GB, test_times, model_name, param_name,cr):
#     import pandas as pd

#     file_name = []
#     get_file(profiler_result_path, file_name)
#     if(len(file_name) == 0):
#         with open(f'{output_file_dir}/{model_name}_{operation}_error.log', 'a') as f:
#             f.write(f'No profiling result found for {model_name} {param_name} in {profiler_result_path}\n')    
#         return
#     results = pd.read_csv(file_name[0])
#     results['Parameter_Name'] = param_name
#     results['datasize_MB'] = datasize_GB * 1024
#     results['Speed(GB/S)'] = datasize_GB * test_times / (results['Avg Time(us)'] / 1000 / 1000)
#     results['cr'] = cr 
#     # Append to CSV (create if doesn't exist)
#     output_file = f'{output_file_dir}/{model_name}_{operation}.csv'
#     if os.path.exists(output_file):
#         results.to_csv(output_file, mode='a', header=False, index=False)
#     else:
#         results.to_csv(output_file, mode='w', header=True, index=False)
#     logger.info(f'results: \n{results}')


# def enec_test(model_name,file_path,dtype,results_dir,operation):
#     import subprocess

#     # logger.info(f"Processing {file_path} ")
    
#     result_path_dir = Path(get_result_path(file_path,results_dir)).parent  # final result path
#     os.makedirs(result_path_dir, exist_ok=True)
#     file_name = Path(file_path).stem  # final file name
#     file_size = os.path.getsize(file_path)
#     # 执行可执行文件并记录结果到结果文件
#     # execute the program and record the results
#     input_file_path = file_path
#     compressed_file_path = f'{result_path_dir}/compressed/{file_name}.compressed'  # compressed file path
#     decompressed_file_path = f'{result_path_dir}/decompressed/{file_name}.decompressed'  # decompressed file path
#     os.makedirs(f'{result_path_dir}/compressed', exist_ok=True)
#     os.makedirs(f'{result_path_dir}/decompressed', exist_ok=True)
#     import time
#     times = time.strftime('%Y%m%d_%H%M%S', time.localtime())
#     msprof_path = f'./prof_path_enec/tmp/model/prof_{model_name}/{operation}/{file_name}/{times}'
#     # Construction Command
#     # Extract parameter names from the file path
#     file_name_with_ext = os.path.basename(input_file_path)
#     if file_name_with_ext.endswith('.bin'):
#         csv_param_name = file_name_with_ext[:-4]
#     else:
#         csv_param_name = file_name_with_ext
    
#     # 1. 检查 hyperparams_results.csv 是否存在
#     hyperparams_csv_path = f'./param_search/BF16/deepseek-llm-7b-base/hyperparams_results.csv'
#     if not os.path.exists(hyperparams_csv_path):
#         logger.error(f"Critical Error: Hyperparams CSV file not found: {hyperparams_csv_path}")
#         return  # 直接返回，防止后续逻辑报错

#     # 2. 读取并匹配参数
#     df = pd.read_csv(hyperparams_csv_path)
#     matching_rows = df[df['parameter_name'] == csv_param_name]

#     if not matching_rows.empty:
#         row = matching_rows.iloc[0]
#         # 提取四个核心超参数并强制转为 int，确保路径格式正确 (如 ENEC-125-6-3-16)
#         b_val = int(row['b'])
#         n_val = int(row['n'])
#         m_val = int(row['m'])
#         L_val = int(row['L'])
        
#         folder_name = f"ENEC-{dtype.upper()}-{b_val}-{n_val}-{m_val}-{L_val}"
#         exec_file_dir = f"./csrc/{folder_name}/build"
        
#         # 3. 核心检查：如果对应的参数文件夹不存在，报错并返回
#         if not os.path.exists(exec_file_dir):
#             logger.error(f"Directory not found for parameters b={b_val}, n={n_val}, m={m_val}, L={L_val}: {exec_file_dir}")
#             return  # 匹配不到实际文件夹，直接终止
        
#         logger.info(f"Successfully matched and verified directory: {exec_file_dir}")
#     else:
#         # 4. 如果 CSV 里连参数名都没找到，也报错返回
#         logger.error(f"Parameter {csv_param_name} not found in hyperparams CSV. Cannot proceed.")
#         return
            
#     if operation == 'compress':
#         command = [
#             "msprof",
#             "--output=" + msprof_path,
#             f"{exec_file_dir}/compress",
#             input_file_path,
#             compressed_file_path,  
#             str(file_size),
#             "16",
#             "0"
#         ]
#     elif operation == 'decompress':
#         command = [
#             "msprof",
#             "--output=" + msprof_path,
#             f"{exec_file_dir}/decompress",
#             compressed_file_path,
#             decompressed_file_path,  
#             input_file_path
#         ]
#     else:
#         raise ValueError(f"Unsupported operation: {operation}. Supported operations are 'compress' and 'decompress'.")

#     result = subprocess.run(command, capture_output=True, text=True)
#     output = result.stdout
#     lines = output.splitlines()
#     cr_value = None
#     for line in lines:
#         if "cr:" in line:
#             parts = line.split("cr:", 1)
#             if len(parts) > 1:
#                 cr_value = parts[1].strip() 
#                 break 

#     cr_float = 0.0
#     if cr_value is not None:
#         try:
#             cr_float = float(cr_value)
#             print(f"Compression ratio cr is: {cr_float}")
#         except ValueError:
#             print("Can not cast to float!")
#     else:
#         print("Something wrong! No cr ,default set to -1")
#     prof_print(msprof_path, result_path_dir, file_size / (1024 * 1024 * 1024), 1, model_name, file_name,cr_float)

#     if result.returncode != 0:
#         logger.error(f"Command:{' '.join(command)} execute fail，return code: {result.returncode}")




# def main():
#     from tqdm import tqdm
#     dtypes = ['FP32', 'FP16', 'BF16']
#     dtype = 'BF16'
#     assert dtype in dtypes, f"Unsupported dtype: {dtype}"
#     # BF16
#     model_name = 'deepseek-llm-7b-base'
#     model_data_dir = f'./models/{dtype}/{model_name}/split'

#     results_dir = f'./results/{model_name}'
#     error_files = []
#     new_dirs = set()
#     for root, dirs, files in tqdm(os.walk(model_data_dir)):
#         for file in files:
#             if file.endswith('.dat') or file.endswith('.bin') or file.endswith('.weight'):
#                 file_path = os.path.join(root, file)
#                 try:
#                     logger.info(f"Processing {file_path} with dtype {dtype}")
#                     final_result_path = Path(get_result_path(file_path,results_dir)).parent
#                     final_result_csv_path = f'{final_result_path}/{model_name}_{operation}.csv'
#                     if os.path.exists(final_result_path) and final_result_path not in new_dirs and os.path.exists(final_result_csv_path):
#                         logger.info(f"File {final_result_csv_path} already exists, skipping.")
#                         continue
#                     new_dirs.add(final_result_path)
#                     if 'fp16' in str(final_result_path).lower():
#                         dtype = 'FP16'
#                     elif 'fp32' in str(final_result_path).lower():
#                         dtype = 'FP32'
#                     elif 'bf16' in str(final_result_path).lower():
#                         dtype = 'BF16'
#                     else:
#                         logger.warning(f"Unsupported dtype in file name {file}, defaulting to {dtype}")
#                     enec_test(model_name,file_path, dtype, results_dir,operation)
#                 except Exception as e:
#                     logger.error(f"Error processing {file_path}, dtype: {dtype}, Error: {e}")
#                     raise e
    
#     if error_files:
#         logger.error("Some file failed during testing:")
#         for file_path, dtype, error in error_files:
#             logger.error(f"File: {file_path}, Dtype: {dtype}, Error: {error}")

# if __name__ == '__main__':
#     main()

import torch
import torch_npu
import time
import os
import pandas as pd
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
data_type = 'models'  
operation = 'decompress'  # optional: compress, decompress
from logger import LoggerGenerator
log_directory = './logs/decomp'
logger = LoggerGenerator.get_logger(log_directory, name=f"{operation} {data_type}", console_output=True)

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

        folder_name = f"ENEC-{dtype.upper()}-{b_val}-{n_val}-{m_val}-{L_val}"
        exec_file_dir = f"./csrc/{dtype}/{folder_name}/build"

        if not os.path.exists(exec_file_dir):
            logger.error(f"Decompressor directory not found: {exec_file_dir}")
            return
        logger.info(f"Matched param {csv_param_name} -> {exec_file_dir}")
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
            str({'BF16':0, 'FP16':1, 'FP32':2}.get(dtype.upper(), 0))
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
    base_models_dir = '/data/yjw/models'
    results_base_dir = './results'
    dtypes = ['FP32', 'FP16', 'BF16']
    MIN_FILE_SIZE = 32768  # 32KB

    error_files = []

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
                        # 检查原始文件大小，小于阈值则跳过（因为压缩也会跳过）
                        file_size = os.path.getsize(file_path)
                        if file_size < MIN_FILE_SIZE:
                            logger.info(f"Skipping {file_path} (size={file_size} bytes < {MIN_FILE_SIZE} bytes) - too small to decompress.")
                            continue
                        # 检查是否已解压
                        result_subdir = Path(results_base_dir) / dtype / model_name
                        decompressed_file = result_subdir / 'decompressed' / f'{param_name}.decompressed'
                        if decompressed_file.exists():
                            logger.info(f"Skipping {file_path}, already decompressed.")
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