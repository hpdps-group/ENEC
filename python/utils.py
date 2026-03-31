import argparse
import os
from typing import Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM,BertModel,CLIPModel,LlamaForCausalLM,Wav2Vec2ForCTC


def load(model_path):
    if 'deepseek-llm-7b-base' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).npu()
        model_name = 'deepseek-llm-7b-base'
    else:
        raise ValueError(f"Unsupported model type or path: {model_path}")

    return model,model_name

def get_result_path(processed_file_path,result_dir):
    common = os.path.commonpath([processed_file_path, result_dir])
    relative = os.path.relpath(processed_file_path, common)
    return os.path.join(result_dir, relative)


def split_model(model_path,dtype='BF16'):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model_name = 'deepseek-llm-7b-base'
    # 在model_path下创建一个split目录
    save_dir = os.path.join(model_path, 'split')
    if os.path.exists(save_dir):
        return
    os.makedirs(save_dir, exist_ok=True)
    print(f"正在将模型拆分并保存到目录: {save_dir}")
    model.eval()
    dtype = dtype.lower()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if  param.dim() == 1:
                continue
            if dtype == 'fp32':
                param_np = param.view(torch.uint32).cpu().numpy()
            else:
                param_np = param.view(torch.uint16).cpu().numpy()
            param_path = os.path.join(save_dir, f"{name}.bin")
            param_np.tofile(param_path)
            print(f"已保存: {param_path}, 形状: {param_np.shape}, 数据类型: {dtype}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ENEC Model Splitter")
    
    # 添加命令行参数
    parser.add_argument("--model_path", type=str, required=True, help="模型权重的存放路径")
    parser.add_argument("--data_type", type=str, default="BF16", choices=["BF16", "FP16", "FP32"], help="数据类型 (默认: BF16)")
    
    args = parser.parse_args()

    # 检查路径是否存在
    if not os.path.exists(args.model_path):
        print(f"Error: 路径 {args.model_path} 不存在")
    else:
        print(f"正在处理模型路径: {args.model_path}，类型: {args.data_type}")
        split_model(args.model_path, args.data_type)