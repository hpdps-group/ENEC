import os
import argparse
import torch
import numpy as np
from pathlib import Path
from safetensors import safe_open

def find_weights_files(model_path):
    """
    在模型目录中查找所有权重文件（.safetensors 或 .bin）。
    优先选择 .safetensors 文件。
    """
    safetensors_files = []
    bin_files = []

    for filename in os.listdir(model_path):
        if filename.endswith(".safetensors"):
            safetensors_files.append(os.path.join(model_path, filename))
        elif filename.endswith(".bin"):
            bin_files.append(os.path.join(model_path, filename))

    # 优先返回 safetensors 文件列表
    if safetensors_files:
        return sorted(safetensors_files)
    elif bin_files:
        return sorted(bin_files)
    else:
        raise FileNotFoundError(f"在 {model_path} 中未找到 .safetensors 或 .bin 文件")

def load_weights(weights_files):
    """从多个 .safetensors 或 .bin 文件加载权重。"""
    all_weights = {}
    print(f"找到 {len(weights_files)} 个权重文件，开始加载...")
    
    for weights_file in weights_files:
        print(f"  > 正在加载: {os.path.basename(weights_file)}")
        if weights_file.endswith(".safetensors"):
            # 使用 safetensors 库加载
            with safe_open(weights_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    all_weights[key] = f.get_tensor(key)
        elif weights_file.endswith(".bin"):
            # 使用 PyTorch 加载
            weights = torch.load(weights_file, map_location="cpu")
            all_weights.update(weights)

    if not all_weights:
        raise ValueError("未能从任何文件中加载权重。")

    print(f"成功从所有文件中加载了总共 {len(all_weights)} 个张量。")
    return all_weights

def save_tensors(weights, output_dir, output_format):
    """将每个张量保存到单独的文件中。"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"正在将张量保存到 {output_dir}...")
    total_tensors = len(weights)
    for i, (key, tensor) in enumerate(weights.items()):
        # 将键名转换为安全的文件名
        safe_key = key.replace('/', '_').replace('.', '_')
        output_path = os.path.join(output_dir, f"{safe_key}.{output_format}")

        if output_format == "pt":
            torch.save(tensor, output_path)
        elif output_format == "bin":
            if tensor.dtype == torch.bfloat16:
                # numpy 不支持 bfloat16，因此我们将其视图转换为 uint16 以保存原始字节
                tensor.view(torch.uint16).cpu().numpy().tofile(output_path)
            else:
                tensor.cpu().numpy().tofile(output_path)
        
        print(f"[{i+1}/{total_tensors}] 已保存 {key} -> {output_path}")

def main():
    parser = argparse.ArgumentParser(description="将模型的权重文件转换为单独的张量文件。")
    parser.add_argument("--model_name", 
                        type=str, 
                        help="位于 'models' 目录下的模型目录名称。",
                        default="stable-video-diffusion")
    parser.add_argument(
        "--output_format",
        type=str,
        choices=['pt', 'bin'],
        default='bin',
        help="输出张量文件的格式 ('pt' 或 'bin')。'pt' 保存 PyTorch 张量对象, 'bin' 保存原始二进制数据。"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='/root/workspaces/datasets/weights_data',
        help="保存张量文件的根目录。"
    )
    args = parser.parse_args()

    # 脚本位于 models/ 目录中，因此模型目录是其同级目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, args.model_name)

    if not os.path.isdir(model_path):
        print(f"错误：在 {model_path} 未找到模型目录")
        return

    try:
        weights_files = find_weights_files(model_path)
        weights = load_weights(weights_files)
        
        # 从第一个张量确定数据类型以用于命名
        first_tensor_dtype = next(iter(weights.values())).dtype
        dtype_str_map = {
            torch.bfloat16: "bf16",
            torch.float16: "fp16",
            torch.float32: "fp32",
        }
        dtype_str = dtype_str_map.get(first_tensor_dtype, str(first_tensor_dtype).replace("torch.", ""))

        # 在输出路径中包含带有数据类型的模型名称和格式
        model_name_with_dtype = f"{args.model_name}_{dtype_str}"
        output_dir = Path(args.output_dir) / model_name_with_dtype
        save_tensors(weights, output_dir, args.output_format)
        print("\n转换完成。")
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
