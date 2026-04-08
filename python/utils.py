import argparse
import os
import torch
import numpy as np
import glob
import gc
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from safetensors.torch import load_file

def get_torch_dtype(dtype_str):
    mapping = {
        "BF16": torch.bfloat16,
        "FP16": torch.float16,
        "FP32": torch.float32
    }
    return mapping.get(dtype_str.upper(), torch.float32)

def save_tensor_to_bin(name, param, save_dir, dtype_str):
    """Extract weights and save as binary files"""
    if param.dim() < 2:
        return False
    
    # Unified view conversion logic
    if dtype_str.upper() == 'FP32':
        param_np = param.detach().view(torch.float32).cpu().numpy()
    else:
        # Use uint16 view for BF16/FP16 to preserve original bit information
        param_np = param.detach().view(torch.uint16).cpu().numpy()

    param_path = os.path.join(save_dir, f"{name.replace('/', '.')}.bin")
    param_np.tofile(param_path)
    return True

def split_model(model_path, dtype_str, force=False):
    target_dtype = get_torch_dtype(dtype_str)
    save_dir = os.path.join(model_path, 'split')
    
    # Checkpoint/Resume check
    if not force and os.path.exists(save_dir):
        bin_files = glob.glob(os.path.join(save_dir, "*.bin"))
        if len(bin_files) > 0:
            print(f"  [Skip] {model_path} ({len(bin_files)} weights already exist)")
            return False
    
    print(f"\n[Processing] Target: {model_path} ({dtype_str})")
    os.makedirs(save_dir, exist_ok=True)

    try:
        loaded_params = {}
        
        # Strategy A: Attempt to load as Diffusion model (look for unet directory)
        unet_path = os.path.join(model_path, "unet")
        if os.path.exists(unet_path):
            print("  -> Diffusion model structure detected. Extracting UNet weights...")
            from diffusers import UNet2DConditionModel, UNetSpatioTemporalConditionModel
            try:
                # For spatio-temporal models like SVD
                m = UNetSpatioTemporalConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=target_dtype)
            except:
                # For standard SD models
                m = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=target_dtype)
            loaded_params = dict(m.named_parameters())
            del m

        # Strategy B: Attempt to load as standard Transformers model
        if not loaded_params:
            try:
                print("  -> Attempting to load via Transformers...")
                try:
                    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=target_dtype, device_map="cpu", low_cpu_mem_usage=True)
                except:
                    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=target_dtype, device_map="cpu", low_cpu_mem_usage=True)
                loaded_params = dict(model.named_parameters())
                del model
            except Exception as e:
                print(f"  -> Class-based loading failed. Attempting direct file read... ({e})")

        # Strategy C: Universal Read (direct access to safetensors/bin files without model definition)
        if not loaded_params:
            # Prioritize safetensors, then fallback to bin
            weight_files = glob.glob(os.path.join(model_path, "**/*.safetensors"), recursive=True) + \
                           glob.glob(os.path.join(model_path, "**/*.bin"), recursive=True)
            
            for wf in weight_files:
                if "split" in wf: continue # Avoid processed directory
                try:
                    if wf.endswith(".safetensors"):
                        weights = load_file(wf, device="cpu")
                    else:
                        weights = torch.load(wf, map_location="cpu")
                    
                    if isinstance(weights, dict):
                        for k, v in weights.items():
                            if isinstance(v, torch.Tensor):
                                loaded_params[f"{os.path.basename(wf)}_{k}"] = v.to(target_dtype)
                except:
                    continue

        # Save weights
        if not loaded_params:
            raise ValueError("No weight parameters could be extracted from the directory")

        count = 0
        for name, param in loaded_params.items():
            if save_tensor_to_bin(name, param, save_dir, dtype_str):
                count += 1
        
        print(f"  ✅ Success: Exported {count} weights (2D+)")
        
    except Exception as e:
        print(f"  ❌ Failed: {model_path} | Error: {e}")
        return False
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return True

def main():
    parser = argparse.ArgumentParser(description="ENEC Weight Splitting Tool (Multi-model Compatible Version)")
    parser.add_argument("--root_dir", type=str, default="models", help="Path to the models root directory")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing results")
    args = parser.parse_args()

    if not os.path.exists(args.root_dir):
        print(f"Error: Root directory not found: {args.root_dir}")
        return

    dtypes = ["BF16", "FP16", "FP32"]
    stats = {"processed": 0, "skipped": 0}

    for dtype in dtypes:
        dtype_path = os.path.join(args.root_dir, dtype)
        if not os.path.exists(dtype_path): continue
        
        print(f"\n{'#'*40}\n## Precision Directory: {dtype}\n{'#'*40}")
        
        for model_name in os.listdir(dtype_path):
            model_full_path = os.path.join(dtype_path, model_name)
            if not os.path.isdir(model_full_path): continue
            
            # Verify if it's a valid model directory
            config_files = ["config.json", "configuration.json", "model_index.json", "hash.txt"]
            is_model = any(os.path.exists(os.path.join(model_full_path, c)) for c in config_files) or \
                       glob.glob(os.path.join(model_full_path, "*.safetensors"))
            
            if is_model:
                if split_model(model_full_path, dtype, force=args.force):
                    stats["processed"] += 1
                else:
                    stats["skipped"] += 1

    print(f"\nSummary: Processed {stats['processed']}, Skipped {stats['skipped']}.")

if __name__ == "__main__":
    main()