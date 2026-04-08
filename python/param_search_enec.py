import os
import sys

# ========== 1. Must be set BEFORE importing torch ==========
os.environ["TORCH_NPU_DYNAMO_ENABLE"] = "0"
os.environ["DISABLE_TORCH_NPU_DYNAMO"] = "1"
os.environ["TORCH_NPU_DYNAMO_OPT"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # Disable all GPU / NPU

import torch
import math
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import mmap

# ========== 2. Force CPU Usage ==========
if hasattr(torch, 'set_default_device'):
    torch.set_default_device('cpu')
else:
    torch.set_default_tensor_type(torch.FloatTensor)

# ========== 3. Suppress all warnings completely ==========
warnings.filterwarnings("ignore", message="Register eager implementation for the 'npu' backend")
warnings.filterwarnings("ignore", message="since the loaded file is not a zipfile")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.serialization")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
warnings.filterwarnings("ignore")

# ========== 4. Logging Module (Fallback to print if import fails) ==========
try:
    from logger import LoggerGenerator
    log_directory = './logs/param_search'
    logger = LoggerGenerator.get_logger(log_directory, name="param search", console_output=True)
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("param search")
    logger.info("Using fallback logging (logger.py not found)")


def find_hyperparams(tensor: torch.Tensor) -> dict:
    """ENEC Hyperparameter Search (High-speed version, forced CPU)"""
    # Ensure tensor is on CPU
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Unsupported dtype. Use BF16, FP16, or FP32.")
    tensor = tensor.contiguous()

    # Exponent extraction
    if tensor.dtype == torch.bfloat16:
        exps = ((tensor.view(torch.uint16).long() >> 7) & 0xFF).to(torch.int64)
        max_exp = 255
    elif tensor.dtype == torch.float16:
        exps = ((tensor.view(torch.uint16).long() >> 10) & 0x1F).to(torch.int64)
        max_exp = 31
    else:  # float32
        exps = ((tensor.view(torch.int32).long() >> 23) & 0xFF).to(torch.int64)
        max_exp = 255

    exps = exps.flatten()
    bin_counts = torch.bincount(exps, minlength=max_exp + 1)
    total = exps.numel()
    p = bin_counts.float() / total

    nonzero = (bin_counts > 0).nonzero(as_tuple=True)[0]
    minnum = nonzero[0].item()
    maxnum = nonzero[-1].item()
    if minnum == maxnum:
        return {'b': minnum, 'n': 1, 'm': 1, 'L': 1, 'average_bit_length': 1.0}

    # Peak detection
    is_multimodal = False
    peaks = 0
    for i in range(minnum, maxnum):
        if p[i] > p[i+1] + 1e-7:
            peaks += 1
            if peaks >= 2:
                is_multimodal = True
                break

    idx = nonzero.float()
    prob = p[nonzero]
    best_b, best_n, min_h = None, None, float('inf')

    if not is_multimodal:
        mode_exp = nonzero[torch.argmax(bin_counts[nonzero])].item()
        b_low = max(mode_exp, minnum + 1)
        b_high = min(mode_exp + 7, maxnum - 1)
        if b_low > b_high:
            b_low = b_high = (minnum + maxnum) // 2
        for b in range(b_low, b_high + 1):
            n = max((b - minnum).bit_length(),
                    (maxnum - b - 1).bit_length() if maxnum > b else 0) + 1
            if n > 8:
                continue
            two_n = 1 << n
            h = (prob * ((b - idx) % two_n)).sum().item()
            if h < min_h:
                min_h, best_b, best_n = h, b, n
            elif h > min_h:
                break
    else:
        for b in range(minnum + 1, maxnum):
            n = max((b - minnum).bit_length(),
                    (maxnum - b - 1).bit_length() if maxnum > b else 0) + 1
            if n > 8:
                continue
            two_n = 1 << n
            h = (prob * ((b - idx) % two_n)).sum().item()
            if h < min_h:
                min_h, best_b, best_n = h, b, n

    if best_b is None:
        raise ValueError(f"Search failed for range [{minnum}, {maxnum}]")

    two_n = 1 << best_n
    r_all = (best_b - idx) % two_n
    best_m, min_avg = 1, float('inf')
    for m in range(1, best_n):
        p_m = p[nonzero[r_all <= ((1 << m) - 1)]].sum().item()
        avg = 0.0625 + best_n + (m - best_n) * (p_m ** 16)
        if avg < min_avg:
            min_avg, best_m = avg, m
        elif not is_multimodal:
            break

    return {
        'b': int(best_b),
        'n': int(best_n),
        'm': int(best_m),
        'L': 16,
        'average_bit_length': float(min_avg)
    }


def load_tensor_from_file(file_path: str, target_dtype_str: str):
    """Ultrafast file loading, forced CPU usage"""
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    # PyTorch format
    if ext in ('.pt', '.pth'):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.serialization")
            warnings.filterwarnings("ignore", message="since the loaded file is not a zipfile")
            data = torch.load(file_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, torch.Tensor):
                    return v.cpu()
            raise ValueError("No tensor found in dict")
        if isinstance(data, torch.Tensor):
            return data.cpu()
        raise ValueError("Unexpected data type")

    # NumPy format
    if ext == '.npy':
        return torch.from_numpy(np.load(file_path)).cpu()

    # Raw Binary (using mmap)
    if ext in ('.bin', '.dat', '.weight'):
        target_dtype_str = target_dtype_str.upper()
        if target_dtype_str == 'FP32':
            expected_dtype = torch.float32
            elem_size = 4
        elif target_dtype_str == 'FP16':
            expected_dtype = torch.float16
            elem_size = 2
        elif target_dtype_str == 'BF16':
            expected_dtype = torch.bfloat16
            elem_size = 2
        else:
            raise RuntimeError(f"Unknown dtype: {target_dtype_str}")

        file_size = file_path.stat().st_size
        if file_size % elem_size != 0:
            raise RuntimeError(f"File size {file_size} is not a multiple of {elem_size}")
        numel = file_size // elem_size

        with open(file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                buffer = np.frombuffer(mm, dtype=np.uint8).copy()
        tensor = torch.frombuffer(buffer, dtype=torch.uint8).view(expected_dtype)[:numel].clone()
        return tensor.cpu()

    raise RuntimeError(f"Unsupported file format: {ext}")


def process_single_file(file_path: str, param_name: str, target_dtype_str: str):
    """Multiprocessing worker: Force CPU and suppress warnings"""
    # Set environment variables for sub-process
    os.environ["TORCH_NPU_DYNAMO_ENABLE"] = "0"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if hasattr(torch, 'set_default_device'):
        torch.set_default_device('cpu')
    warnings.filterwarnings("ignore", message="Register eager implementation for the 'npu' backend")
    warnings.filterwarnings("ignore", message="since the loaded file is not a zipfile")
    torch.set_num_threads(1)

    try:
        tensor = load_tensor_from_file(file_path, target_dtype_str)
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return {'success': False, 'name': param_name, 'reason': f"unsupported dtype {tensor.dtype}"}
        hp = find_hyperparams(tensor)
        return {
            'success': True,
            'name': param_name,
            'shape_str': "x".join(map(str, tensor.shape)),
            'num_elements': tensor.numel(),
            'hyperparams': hp,
        }
    except Exception as e:
        return {'success': False, 'name': param_name, 'reason': str(e)}


def calculate_model_compression_stats(csv_file: str, dtype: str, stats_file: str = None):
    """Compression statistics calculation"""
    if not os.path.exists(csv_file):
        logger.error(f"Statistics file does not exist: {csv_file}")
        return
    try:
        df = pd.read_csv(csv_file, sep=',')
        if len(df.columns) < 8:
            df = pd.read_csv(csv_file, sep='\t')
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return
    dtype_upper = dtype.upper()
    if dtype_upper == 'BF16':
        orig_bits = 16; mantissa = 7; const = 8; name = "BF16"
    elif dtype_upper == 'FP16':
        orig_bits = 16; mantissa = 10; const = 11; name = "FP16"
    elif dtype_upper == 'FP32':
        orig_bits = 32; mantissa = 23; const = 24; name = "FP32"
    else:
        logger.error(f"Unknown data type: {dtype}")
        return
    df['compressed_bits_per_elem'] = 1 + mantissa + df['average_bit_length']
    df['compressed_bits'] = df['num_elements'] * df['compressed_bits_per_elem']
    df['original_bits'] = df['num_elements'] * orig_bits
    total_comp = df['compressed_bits'].sum()
    total_orig = df['original_bits'].sum()
    total_elems = df['num_elements'].sum()
    comp_mb = total_comp / (8 * 1024 * 1024)
    orig_mb = total_orig / (8 * 1024 * 1024)
    cr = total_orig / total_comp if total_comp > 0 else 0
    avg_bits = total_comp / total_elems if total_elems > 0 else 0
    total_exp_bits = (df['num_elements'] * df['average_bit_length']).sum()
    avg_exp_bits = total_exp_bits / total_elems if total_elems > 0 else 0
    cr_formula = orig_bits / (const + avg_exp_bits) if (const + avg_exp_bits) > 0 else 0

    print("\n" + "="*60)
    print(f"{name} Model Compression Results".center(58))
    print("="*60)
    print(f"File Processed:      {os.path.basename(csv_file)}")
    print(f"Total Elements:      {total_elems:,}")
    print("-" * 60)
    print(f"Original {name} Size:   {orig_mb:>10.2f} MB")
    print(f"ENEC Compressed Size:  {comp_mb:>10.2f} MB")
    print(f"Compression Ratio (CR):{cr:>10.2f}x")
    print(f"Model Avg Bit-width:   {avg_bits:>10.4f} bits/element")
    print(f"Exponent Avg Bit-width:{avg_exp_bits:>10.4f} bits/element")
    print(f"Formula Avg CR*:       {cr_formula:>12.2f} x")
    print("="*60 + "\n")
    if stats_file:
        with open(stats_file, 'a') as f:
            f.write("\n" + "="*60 + "\n")
            f.write(f"{name} Model Compression Results\n")
            f.write("="*60 + "\n")
            f.write(f"File Processed: {os.path.basename(csv_file)}\n")
            f.write(f"Total Elements: {total_elems:,}\n")
            f.write("-"*60 + "\n")
            f.write(f"Original {name} Size: {orig_mb:.2f} MB\n")
            f.write(f"ENEC Compressed Size: {comp_mb:.2f} MB\n")
            f.write(f"Compression Ratio (CR): {cr:.2f}x\n")
            f.write(f"Model Avg Bit-width: {avg_bits:.4f} bits/element\n")
            f.write(f"Exponent Avg Bit-width: {avg_exp_bits:.4f} bits/element\n")
            f.write(f"Formula Avg CR*: {cr_formula:.2f} x\n")
            f.write("="*60 + "\n")


def search_param_model(model_name, dtype, split_dir, results_dir):
    """Process all parameter files for a single model using multiprocessing"""
    file_list = []
    for root, _, files in os.walk(split_dir):
        for file in files:
            if file.endswith(('.dat', '.bin', '.weight', '.pt', '.pth', '.npy')):
                file_list.append((os.path.join(root, file), Path(file).stem))
    if not file_list:
        logger.warning(f"No tensor files found in {split_dir}")
        return

    result_path = os.path.join(results_dir, dtype, model_name)
    os.makedirs(result_path, exist_ok=True)
    csv_file = os.path.join(result_path, "hyperparams_results.csv")
    with open(csv_file, 'w') as f:
        f.write("parameter_name,shape,num_elements,b,n,m,L,average_bit_length\n")

    param_combos = {}
    total_files = len(file_list)
    processed = 0
    num_workers = min(multiprocessing.cpu_count(), total_files)
    logger.info(f"Processing {total_files} files from {split_dir} using {num_workers} workers")

    try:
        mp_ctx = multiprocessing.get_context('spawn')
    except:
        mp_ctx = multiprocessing

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as executor:
        future_to_name = {
            executor.submit(process_single_file, fp, name, dtype): name
            for fp, name in file_list
        }
        for future in as_completed(future_to_name):
            res = future.result()
            if res['success']:
                name = res['name']
                hp = res['hyperparams']
                with open(csv_file, 'a') as f:
                    f.write(f"{name},{res['shape_str']},{res['num_elements']},"
                            f"{hp['b']},{hp['n']},{hp['m']},{hp['L']},{hp['average_bit_length']:.6f}\n")
                combo = f"({hp['b']},{hp['n']},{hp['m']},{hp['L']})"
                param_combos[combo] = param_combos.get(combo, 0) + 1
                processed += 1
                logger.info(f"Completed {name} ({dtype}/{model_name})")
            else:
                logger.warning(f"Skipping {res['name']}: {res.get('reason', 'unknown')}")

    stats_file = os.path.join(result_path, "param_combinations_stats.txt")
    with open(stats_file, 'w') as f:
        f.write("Parameter Combinations Statistics\n")
        f.write("="*40 + "\nFormat: (b,n,m,L) -> frequency\n" + "-"*40 + "\n")
        if param_combos:
            sorted_c = sorted(param_combos.items(), key=lambda x: x[1], reverse=True)
            total_params = sum(param_combos.values())
            for combo, cnt in sorted_c:
                f.write(f"{combo}: {cnt} ({cnt/total_params*100:.2f}%)\n")
            f.write("-"*40 + f"\nTotal params: {total_params}\nUnique combos: {len(param_combos)}\n")
        else:
            f.write("No valid parameters found.\n")
        f.write(f"Total files scanned: {total_files}\nProcessed: {processed}\n")

    logger.info(f"Completed {model_name} ({dtype})")
    logger.info(f"Results: {csv_file}, Stats: {stats_file}")
    calculate_model_compression_stats(csv_file, dtype, stats_file)


def discover_models(base_dir='./models'):
    """Quickly discover models and return a list of (dtype, model_name, split_dir)"""
    discovered = []
    if not os.path.isdir(base_dir):
        # Use print to ensure the error is seen (in case logger is unconfigured)
        print(f"ERROR: Models directory not found: {base_dir}")
        logger.error(f"Models directory not found: {base_dir}")
        return discovered
    for dtype in ('FP32', 'FP16', 'BF16'):
        dtype_path = os.path.join(base_dir, dtype)
        if not os.path.isdir(dtype_path):
            continue
        with os.scandir(dtype_path) as it:
            for entry in it:
                if entry.is_dir():
                    split_dir = os.path.join(dtype_path, entry.name, 'split')
                    if os.path.isdir(split_dir):
                        discovered.append((dtype, entry.name, split_dir))
                    else:
                        logger.info(f"No split dir for {entry.name} ({dtype})")
    return discovered


def main():
    print("=== Starting ENEC hyperparameter search (CPU only) ===", flush=True)
    model_dir = './models'
    results_dir = './param_search_enec'
    error_models = []

    # Check model directory
    if not os.path.isdir(model_dir):
        print(f"FATAL: Model directory '{model_dir}' does not exist. Please create it or adjust the path.")
        logger.error(f"Model directory '{model_dir}' not found.")
        return

    models = discover_models(model_dir)
    print(f"Found {len(models)} model(s) to process.", flush=True)
    if not models:
        print("No models with split directories found. Ensure structure: ./models/{FP32,FP16,BF16}/<model_name>/split/")
        logger.error("No models with split directories found.")
        return

    for dtype, model_name, split_dir in models:
        out_csv = os.path.join(results_dir, dtype, model_name, "hyperparams_results.csv")
        if os.path.exists(out_csv):
            logger.info(f"Skipping {model_name} ({dtype}) - results already exist.")
            continue
        logger.info(f"Processing {model_name} ({dtype}) from {split_dir}")
        try:
            search_param_model(model_name, dtype, split_dir, results_dir)
        except Exception as e:
            error_models.append((model_name, dtype, str(e)))
            logger.error(f"Error processing {model_name}: {e}")

    if error_models:
        logger.error("Failed models:")
        for m, d, e in error_models:
            logger.error(f"{m} ({d}): {e}")
    print("=== All done ===", flush=True)


if __name__ == "__main__":
    # Set multiprocessing start method (avoid fork memory issues)
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()