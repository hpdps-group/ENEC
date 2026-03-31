import torch
import torch_npu
import numpy as np
import time

def get_compression_info(input_tensor, compress_header):
    """
    根据压缩头部信息计算原始大小、压缩后大小和压缩率。
    """
    tensor_numel = input_tensor.numel()
    tensor_element_size = input_tensor.element_size()
    original_size = tensor_numel * tensor_element_size
    
    # 压缩后的大小由多个部分组成
    fixed_compressed_size = sum(compress_header[8:8+48])
    var_compressed_size = sum(compress_header[64:64+48])
    mantissa_size = tensor_numel * (tensor_element_size - 1)
    pdf_size = 256 * 4  # 256个int32
    header_size = 512 # 128个int32

    compressed_size = fixed_compressed_size + var_compressed_size + mantissa_size + pdf_size + header_size
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    return original_size, compressed_size, compression_ratio


def test_hans_on_tensor(input_tensor):
    """
    对给定的tensor测试HANS压缩和解压缩，并评估其性能。
    """
    # HANS压缩的可选配置
    statistics = [True, False]
    reshuffs = [True, False]
    
    # 用于性能测试的迭代次数
    warmup_runs = 5
    test_runs = 50

    for statistic in statistics:
        for reshuff in reshuffs:
            # 跳过无效组合 (reshuff=True 必须搭配 statistic=True)
            if not statistic and reshuff:
                continue

            print(f"\n===== 测试配置: statistic={statistic}, reshuff={reshuff} =====\n")
            
            # 准备HANS API所需的输出张量
            recover = torch.zeros_like(input_tensor)
            pdf = torch.zeros(256, dtype=torch.int32).npu()
            mantissa_numel = input_tensor.numel() * (input_tensor.element_size() - 1)
            mantissa = torch.zeros(mantissa_numel // input_tensor.element_size(), dtype=input_tensor.dtype).npu()
            fixed = torch.zeros(input_tensor.numel(), dtype=input_tensor.dtype).npu()
            var = torch.zeros(input_tensor.numel(), dtype=input_tensor.dtype).npu()

            # --- 预热 ---
            # 在正式计时前执行几次操作，以确保性能稳定
            for _ in range(warmup_runs):
                _, _, _, _ = torch_npu.npu_hans_encode(input_tensor, statistic, reshuff, out=(pdf, mantissa, fixed, var))
                _ = torch_npu.npu_hans_decode(mantissa, fixed, var, pdf, reshuff, out=recover)

            # --- 吞吐率测试 ---
            data_size_gb = input_tensor.numel() * input_tensor.element_size() / (1024**3)
            total_data_processed_gb = data_size_gb * test_runs

            # 1. 测试压缩吞吐率
            torch.npu.synchronize()
            start_time_encode = time.time()
            for _ in range(test_runs):
                _, _, _, _ = torch_npu.npu_hans_encode(input_tensor, statistic, reshuff, out=(pdf, mantissa, fixed, var))
            torch.npu.synchronize()
            end_time_encode = time.time()
            total_time_encode = end_time_encode - start_time_encode
            encode_throughput_gb_s = total_data_processed_gb / total_time_encode

            # 2. 测试解压吞吐率
            torch.npu.synchronize()
            start_time_decode = time.time()
            for _ in range(test_runs):
                _ = torch_npu.npu_hans_decode(mantissa, fixed, var, pdf, reshuff, out=recover)
            torch.npu.synchronize()
            end_time_decode = time.time()
            total_time_decode = end_time_decode - start_time_decode
            decode_throughput_gb_s = total_data_processed_gb / total_time_decode
            
            # --- 压缩率计算 ---
            compress_header = fixed.view(torch.int32)[:128].cpu()
            original_size, compressed_size, compression_ratio = get_compression_info(input_tensor, compress_header)

            # --- 打印结果 ---
            print("--- 性能结果 ---")
            print(f"原始大小: {original_size / (1024**2):.2f} MB")
            print(f"压缩后大小: {compressed_size / (1024**2):.2f} MB")
            print(f"压缩率: {compression_ratio:.4f}")
            print(f"压缩吞吐率: {encode_throughput_gb_s:.2f} GB/s")
            print(f"解压吞吐率: {decode_throughput_gb_s:.2f} GB/s")
            print("-" * 20)
            
            # --- 验证正确性 ---
            are_equal = torch.allclose(input_tensor, recover, rtol=0, atol=0)
            if are_equal:
                print("验证通过: 解压缩后的数据与原始数据完全一致。")
            else:
                diff_count = torch.sum(input_tensor != recover)
                print(f"验证失败: 存在 {diff_count} 个不同的元素。")


if __name__ == '__main__':
    # 设置目标文件路径和数据类型
    import sys
    if len(sys.argv) < 2:
        print("错误: 请提供文件路径作为参数")
        sys.exit(1)
    file_path = sys.argv[1]
    dtype = torch.bfloat16
    numpy_dtype = np.float16 # numpy没有bfloat16，用float16读取，再转为torch的bfloat16

    print(f"正在从 '{file_path}' 加载数据...")

    try:
        # 从二进制文件加载数据
        data = np.fromfile(file_path, dtype=numpy_dtype)
        
        # 将numpy数组转换为torch张量
        input_tensor_cpu = torch.from_numpy(data).to(dtype)
        
        # HANS压缩要求元素数量是64的倍数，这里进行裁剪
        num_elements = input_tensor_cpu.numel()
        valid_elements = (num_elements // 64) * 64
        
        if valid_elements == 0:
            raise ValueError("数据量过小，无法满足HANS压缩要求 (元素数量需 >= 64)。")

        if valid_elements < num_elements:
             print(f"原始元素数量 {num_elements}，不满足64的倍数要求。已裁剪为 {valid_elements}。")

        # 裁剪并移动到NPU
        input_tensor_npu = input_tensor_cpu[:valid_elements].npu()

        print(f"数据加载完成。张量形状: {input_tensor_npu.shape}, 数据类型: {input_tensor_npu.dtype}")

        # 执行压缩和解压缩测试
        test_hans_on_tensor(input_tensor_npu)

    except FileNotFoundError:
        print(f"错误: 文件未找到 '{file_path}'")
    except Exception as e:
        print(f"发生错误: {e}")