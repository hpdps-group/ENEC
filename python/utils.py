# import argparse
# import os
# from typing import Tuple
# import numpy as np
# import torch
# from transformers import AutoModelForCausalLM,BertModel,CLIPModel,LlamaForCausalLM,Wav2Vec2ForCTC


# def load(model_path):
#     if 'deepseek-llm-7b-base' in model_path:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             trust_remote_code=True,
#             torch_dtype=torch.bfloat16
#         ).npu()
#         model_name = 'deepseek-llm-7b-base'
#     else:
#         raise ValueError(f"Unsupported model type or path: {model_path}")

#     return model,model_name

# def get_result_path(processed_file_path,result_dir):
#     common = os.path.commonpath([processed_file_path, result_dir])
#     relative = os.path.relpath(processed_file_path, common)
#     return os.path.join(result_dir, relative)


# def split_model(model_path,dtype='BF16'):
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         trust_remote_code=True,
#         torch_dtype=torch.bfloat16
#     )
#     model_name = 'deepseek-llm-7b-base'
#     # 在model_path下创建一个split目录
#     save_dir = os.path.join(model_path, 'split')
#     if os.path.exists(save_dir):
#         return
#     os.makedirs(save_dir, exist_ok=True)
#     print(f"正在将模型拆分并保存到目录: {save_dir}")
#     model.eval()
#     dtype = dtype.lower()
#     with torch.no_grad():
#         for name, param in model.named_parameters():
#             if  param.dim() == 1:
#                 continue
#             if dtype == 'fp32':
#                 param_np = param.view(torch.uint32).cpu().numpy()
#             else:
#                 param_np = param.view(torch.uint16).cpu().numpy()
#             param_path = os.path.join(save_dir, f"{name}.bin")
#             param_np.tofile(param_path)
#             print(f"已保存: {param_path}, 形状: {param_np.shape}, 数据类型: {dtype}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="ENEC Model Splitter")
    
#     # 添加命令行参数
#     parser.add_argument("--model_path", type=str, required=True, help="模型权重的存放路径")
#     parser.add_argument("--data_type", type=str, default="BF16", choices=["BF16", "FP16", "FP32"], help="数据类型 (默认: BF16)")
    
#     args = parser.parse_args()

#     # 检查路径是否存在
#     if not os.path.exists(args.model_path):
#         print(f"Error: 路径 {args.model_path} 不存在")
#     else:
#         print(f"正在处理模型路径: {args.model_path}，类型: {args.data_type}")
#         split_model(args.model_path, args.data_type)

import argparse
import os
import torch
import numpy as np
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

def get_torch_dtype(dtype_str):
    """映射字符串到 torch 精度类型"""
    mapping = {
        "BF16": torch.bfloat16,
        "FP16": torch.float16,
        "FP32": torch.float32
    }
    return mapping.get(dtype_str.upper(), torch.float32)

def split_model(model_path, dtype_str):
    # 1. 自动推断模型类型并加载
    print(f"\n[Processing] 正在加载模型: {model_path}")
    target_dtype = get_torch_dtype(dtype_str)
    
    save_dir = os.path.join(model_path, 'split')
    if os.path.exists(save_dir):
        print(f"跳过: {save_dir} 已存在")
        return

    try:
        # 使用 AutoModelForCausalLM 加载，如果失败则尝试通用 AutoModel (针对 BERT/Wav2Vec2)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                torch_dtype=target_dtype,
                device_map="cpu" # 拆分通常在 CPU 内存完成即可
            )
        except Exception:
            model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                torch_dtype=target_dtype,
                device_map="cpu"
            )

        os.makedirs(save_dir, exist_ok=True)
        model.eval()

        # 2. 遍历参数并保存为二进制
        with torch.no_grad():
            for name, param in model.named_parameters():
                # 排除 1D 参数（如 bias, LayerNorm），通常只压缩 2D 及以上的权重矩阵
                if param.dim() < 2:
                    continue
                
                # 转换为对应的 numpy 类型
                if dtype_str.upper() == 'FP32':
                    # FP32 视图转换
                    param_np = param.detach().view(torch.float32).cpu().numpy()
                else:
                    # BF16/FP16 在内存中通常是 uint16 存储
                    param_np = param.detach().view(torch.uint16).cpu().numpy()

                param_path = os.path.join(save_dir, f"{name}.bin")
                param_np.tofile(param_path)
                # print(f"  -> 已保存: {name}.bin | Shape: {param_np.shape}")

        print(f"✅ 成功: {model_path} 权重已拆分至 {save_dir}")
        
        # 释放内存防止 OOM
        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ 失败: 处理 {model_path} 时发生错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="ENEC 批量模型权重拆分工具")
    parser.add_argument("--root_dir", type=str, default="models", help="models 根目录路径")
    args = parser.parse_args()

    if not os.path.exists(args.root_dir):
        print(f"错误: 找不到根目录 {args.root_dir}")
        return

    # 定义要扫描的子目录（精度）
    dtypes = ["BF16", "FP16", "FP32"]

    for dtype in dtypes:
        dtype_path = os.path.join(args.root_dir, dtype)
        if not os.path.exists(dtype_path):
            continue
        
        print(f"\n{'='*20} 开始处理 {dtype} 目录 {'='*20}")
        
        # 遍历精度目录下的各个模型文件夹
        for model_name in os.listdir(dtype_path):
            model_full_path = os.path.join(dtype_path, model_name)
            
            # 确保是文件夹且包含配置文件
            if os.path.isdir(model_full_path) and \
               (os.path.exists(os.path.join(model_full_path, "config.json")) or \
                os.path.exists(os.path.join(model_full_path, "configuration.json"))):
                
                split_model(model_full_path, dtype)

if __name__ == "__main__":
    main()