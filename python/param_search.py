import torch
import math
import os
from utils import load
from load_safetensors import find_weights_files, load_weights
import time

import warnings
warnings.filterwarnings("ignore")
# from test_case import setup_seed
from logger import LoggerGenerator
log_directory = './logs/param_search'
logger = LoggerGenerator.get_logger(log_directory, name="param search", console_output=True)

def find_hyperparams(tensor: torch.Tensor) -> dict:
    """
    Finds the optimal hyperparameters b, n, m, L for the given tensor based on the described algorithm.
    
    Args:
        tensor (torch.Tensor): Input tensor of dtype torch.bfloat16, torch.float16, or torch.float32.
    
    Returns:
        dict: Dictionary containing 'b', 'n', 'm', 'L', and 'average_bit_length'.
    """
    if tensor.dtype not in [torch.bfloat16, torch.float16, torch.float32]:
        raise ValueError("Input tensor must be of dtype torch.bfloat16, torch.float16, or torch.float32")
    
    # Extract exponents based on dtype
    if tensor.dtype == torch.bfloat16:
        # BF16: 16-bit, exponent 8 bits at positions [15:7]
        exps = ((tensor.view(torch.uint16).long() >> 7) & 0xFF).to(torch.int64)
        max_exp_value = 255
    elif tensor.dtype == torch.float16:
        # FP16: 16-bit, exponent 5 bits at positions [15:10]
        exps = ((tensor.view(torch.uint16).long() >> 10) & 0x1F).to(torch.int64)
        max_exp_value = 31
    elif tensor.dtype == torch.float32:
        # FP32: 32-bit, exponent 8 bits at positions [31:23]
        exps = ((tensor.view(torch.uint32).long() >> 23) & 0xFF).to(torch.int64)
        max_exp_value = 255
    
    exps = exps.flatten()
    
    if exps.numel() == 0:
        raise ValueError("Input tensor is empty")
    
    # Compute frequencies
    unique, counts = torch.unique(exps, return_counts=True)
    total = exps.numel()
    p = torch.zeros(max_exp_value + 1, dtype=torch.float)
    p[unique] = (counts.float() / total)
    
    minnum = unique.min().item()
    maxnum = unique.max().item()
    
    if minnum == maxnum:
        # Special case: all exponents are the same
        return {'b': minnum, 'n': 1, 'm': 1, 'L': 1, 'average_bit_length': 1.0}
    
    # Search for b and n to minimize h
    min_h = float('inf')
    best_b = None
    best_n = None
    
    for b in range(minnum + 1, maxnum):
        left_val = b - minnum
        right_val = maxnum - b
        if left_val < 1 or right_val < 1:
            continue
        
        left = math.floor(math.log2(left_val)) + 1
        right = math.ceil(math.log2(right_val))
        n = max(left, right) + 1

        if n > 8:
            continue
        
        # Compute h
        two_n = 1 << n  # 2**n
        r = (two_n + b - unique) % two_n
        h = (p[unique] * r.float()).sum().item()
        
        if h < min_h:
            min_h = h
            best_b = b
            best_n = n
    
    if best_b is None or best_n is None:
        raise ValueError("No valid b and n found. Check the exponent distribution.")
    
    # Now, based on best_b and best_n, compute r(x) for all x
    two_n = 1 << best_n
    r_all = (two_n + best_b - unique) % two_n
    
    # Now search for m and L to minimize average bit length
    # L_options = [1, 2, 4, 8, 16, 32, 64]
    L_options = [ 16 ]
    min_avg = float('inf')
    best_m = None
    best_L = None
    
    for m in range(0, best_n + 1):  # 0 <= m <= n
        two_m = 1 << m
        # p(m) = prob(r(x) <= 2^m - 1)
        mask = r_all <= (two_m - 1)
        p_m = p[unique[mask]].sum().item()
        
        for L in L_options:
            if p_m == 0:
                avg = 1.0 / L + best_n
            else:
                p_m_L = p_m ** L
                avg = 1.0 / L + best_n + (m - best_n) * p_m_L
            
            if avg < min_avg:
                min_avg = avg
                best_m = m
                best_L = L
            print(f"b={best_b}, n={best_n}, m={m}, L={L}, avg={avg:.6f}")
    
    return {
        'b': best_b,
        'n': best_n,
        'm': best_m,
        'L': best_L,
        'average_bit_length': min_avg
    }



def search_param_model(model_name,dtype,model_path,results_dir):
    """
        对每个模型，进行如下处理：
        搜索超参数b,n,m,L
    """
    if model_name in ['stable-video-diffusion-img2vid-fp16']:
        # 通过加载单独的safetensor文件进行测试
        weights_files = find_weights_files(model_path)
        weights = load_weights(weights_files)
        
        # 创建结果目录
        result_path = os.path.join(results_dir, dtype, model_name)
        os.makedirs(result_path, exist_ok=True)
        
        # 准备保存所有结果
        all_results = []
        param_combinations = {}  # 统计参数组合频率
        
        # 创建CSV文件并写入表头
        csv_file = os.path.join(result_path, f"hyperparams_results.csv")
        with open(csv_file, 'w') as f:
            f.write("parameter_name,shape,num_elements,b,n,m,L,average_bit_length\n")
        
        # 遍历所有权重张量
        total_tensors = len(weights)
        processed_count = 0
        
        for name, tensor in weights.items():
            # 只处理多维度且支持的数据类型的张量
            if tensor.ndim > 1 and tensor.dtype in [torch.bfloat16, torch.float16, torch.float32]:
                try:
                    hyperparams = find_hyperparams(tensor)
                    logger.info(f"Model: {model_name}, Tensor: {name}, Dtype: {tensor.dtype}, Hyperparams: {hyperparams}")
                    
                    # 保存到CSV文件
                    shape_str = "x".join(map(str, tensor.shape))
                    num_elements = tensor.numel()
                    
                    with open(csv_file, 'a') as f:
                        f.write(f"{name},{shape_str},{num_elements},{hyperparams['b']},{hyperparams['n']},{hyperparams['m']},{hyperparams['L']},{hyperparams['average_bit_length']:.6f}\n")
                    
                    # 统计参数组合频率
                    combo_key = f"({hyperparams['b']},{hyperparams['n']},{hyperparams['m']},{hyperparams['L']})"
                    if combo_key in param_combinations:
                        param_combinations[combo_key] += 1
                    else:
                        param_combinations[combo_key] = 1
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing tensor {name} in model {model_name}: {e}")
                    continue
            else:
                if tensor.ndim <= 1:
                    logger.debug(f"Skipping tensor {name} in model {model_name} as it has {tensor.ndim} dimensions")
                else:
                    logger.warning(f"Skipping tensor {name} in model {model_name} as it has unsupported dtype: {tensor.dtype}")
        
        # 生成统计信息文件
        stats_file = os.path.join(result_path, "param_combinations_stats.txt")
        with open(stats_file, 'w') as f:
            f.write("Parameter Combinations Statistics\n")
            f.write("=" * 40 + "\n")
            f.write("Format: (b,n,m,L) -> frequency\n")
            f.write("-" * 40 + "\n")
            
            if param_combinations:
                # 按频率降序排列
                sorted_combinations = sorted(param_combinations.items(), key=lambda x: x[1], reverse=True)
                total_params = sum(param_combinations.values())
                
                for combo, count in sorted_combinations:
                    percentage = (count / total_params) * 100
                    f.write(f"{combo}: {count} ({percentage:.2f}%)\n")
                
                f.write("-" * 40 + "\n")
                f.write(f"Total parameters processed: {total_params}\n")
                f.write(f"Unique combinations: {len(param_combinations)}\n")
            else:
                f.write("No valid parameters found for processing.\n")
            
            f.write(f"Total tensors in model: {total_tensors}\n")
            f.write(f"Processed tensors: {processed_count}\n")

        logger.info(f"Successfully completed hyperparameter search for {model_name} ({dtype})")
        logger.info(f"Results saved to: {csv_file}")
        logger.info(f"Statistics saved to: {stats_file}")
        logger.info(f"Processed {processed_count} out of {total_tensors} tensors")
        
        return


    try:
        # 获取模型参数分布数据
        model,model_name = load(model_path)

        # 创建结果目录
        result_path = os.path.join(results_dir, dtype, model_name)
        os.makedirs(result_path, exist_ok=True)
        
        # 准备保存所有结果
        all_results = []
        param_combinations = {}  # 统计参数组合频率
        
        # 创建CSV文件并写入表头
        csv_file = os.path.join(result_path, "hyperparams_results.csv")
        with open(csv_file, 'w') as f:
            f.write("parameter_name,shape,num_elements,b,n,m,L,average_bit_length\n")

        # 统计参数搜索总的时间(包含文件写入以及总的频率统计的时间)
        start_time = time.time()

        for name, param in model.cpu().named_parameters():
            # if param.requires_grad and param.ndim > 1 and 'model.layers.0.self_attn.k_proj' in name:
            # if param.requires_grad and param.ndim > 1 and 'embed' not in name:
            if param.requires_grad and param.ndim > 1:
                if param.dtype in [torch.bfloat16, torch.float16, torch.float32]:
                    hyperparams = find_hyperparams(param.data)
                    logger.info(f"Model: {model_name}, Param: {name}, Dtype: {param.dtype}, Hyperparams: {hyperparams}")
                    
                    # 保存到CSV文件
                    shape_str = "x".join(map(str, param.shape))
                    num_elements = param.numel()
                    
                    with open(csv_file, 'a') as f:
                        f.write(f"{name},{shape_str},{num_elements},{hyperparams['b']},{hyperparams['n']},{hyperparams['m']},{hyperparams['L']},{hyperparams['average_bit_length']:.6f}\n")
                    
                    # 统计参数组合频率
                    combo_key = f"({hyperparams['b']},{hyperparams['n']},{hyperparams['m']},{hyperparams['L']})"
                    if combo_key in param_combinations:
                        param_combinations[combo_key] += 1
                    else:
                        param_combinations[combo_key] = 1
                    
                else:
                    logger.warning(f"Skipping param {name} in model {model_name} as it is not supported dtype: {param.dtype}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Total time for hyperparameter search and saving results: {elapsed_time:.2f} seconds")
        # 生成统计信息文件
        stats_file = os.path.join(result_path, "param_combinations_stats.txt")
        with open(stats_file, 'w') as f:
            f.write("Parameter Combinations Statistics\n")
            f.write("Time Total: {:.2f} seconds\n".format(elapsed_time))
            f.write("=" * 40 + "\n")
            f.write("Format: (b,n,m,L) -> frequency\n")
            f.write("-" * 40 + "\n")
            
            # 按频率降序排列
            sorted_combinations = sorted(param_combinations.items(), key=lambda x: x[1], reverse=True)
            total_params = sum(param_combinations.values())
            
            for combo, count in sorted_combinations:
                percentage = (count / total_params) * 100
                f.write(f"{combo}: {count} ({percentage:.2f}%)\n")
            
            f.write("-" * 40 + "\n")
            f.write(f"Total parameters processed: {total_params}\n")
            f.write(f"Unique combinations: {len(param_combinations)}\n")

        logger.info(f"Successfully completed hyperparameter search for {model_name} ({dtype})")
        logger.info(f"Results saved to: {csv_file}")
        logger.info(f"Statistics saved to: {stats_file}")

    except Exception as e:
        logger.error(f"Error in search_param_model for {model_name} ({dtype}): {e}")
        raise

def main():
    model_dir = './models'

    # 遍历所有
    """
    ├── BF16
    │   ├── deepseek-llm-7b-base
    │   ├── falcon-7b
    │   ├── Qwen3-32B
    │   └── Qwen3-8B
    │   └── Meta-Llama-3.1-8B-Instruct
    ├── FP16
    │   ├── CapybaraHermes-2.5-Mistral-7B
    │   └── stable-video-diffusion-img2vid-fp16  # currently not supported
    │   └── Llama-2-7b-ms
    └── FP32
        ├── bert-base-uncased
        ├── OLMo-1B-hf
        └── wav2vec2-large-xlsr-53-english
    
    """
    # dtypes = ['FP32', 'FP16', 'BF16']
    dtypes = ['FP32', 'FP16', 'BF16']
    FP32_models = [
                #     'bert-base-uncased', 
                #    'OLMo-1B-hf', 
                #    'wav2vec2-large-xlsr-53-english'
                   ]
    FP16_models = [
                # 'CapybaraHermes-2.5-Mistral-7B', 
                #    'stable-video-diffusion-img2vid-fp16',
                #    'Llama-2-7b-ms'
                   ]
    BF16_models = [
                    'deepseek-llm-7b-base', 
                    # 'falcon-7b', 
                    # 'Qwen3-32B', 
                    # 'Meta-Llama-3.1-8B-Instruct', 
                    # 'Qwen3-8B'
                ]
    models = {
        'FP32': FP32_models,
        'FP16': FP16_models,
        'BF16': BF16_models
    }
    error_models = []
    results_dir = './param_search'
    for dtype in dtypes:
        for model_name in models[dtype]:
            # 检查结果目录中该模型是否已经存在
            model_result_path = f'{results_dir}/{dtype}/{model_name}/hyperparams_results.csv'
            if os.path.exists(model_result_path):
                logger.info(f"Skipping {model_name} with {dtype}, results already exist.")
                continue
            logger.info(f"Processing {model_name} with {dtype}")
            try:
                search_param_model(model_name, dtype, f'{model_dir}/{dtype}/{model_name}', results_dir)
            except Exception as e:
                error_models.append((model_name, dtype, str(e)))
                logger.error(f"Error processing {model_name} with {dtype}: {e}")
                continue
    if error_models:
        logger.error("Some models failed during testing:")
        for model_name, dtype, error in error_models:
            logger.error(f"Model: {model_name}, Dtype: {dtype}, Error: {error}")


if __name__ == "__main__":
    main()