# ENEC
- 目录结构
```
- csrc:npu kernel 实现
- param_search:参数搜索结果
- python:数据处理，参数搜索，压缩测试和profile结果收集
- results:最终结果
```
- 系统和软件库要求
```
  1. 推荐平台：Linux（Ubuntu22.04），aarch64
  2. 推荐python版本：python 3.9
  3. NPU：Ascend 910B2
  4. 推荐CANN版本：8.2.RC1.alpha002
  5. 推荐cann kernels版本：8.2.RC1.alpha002
  6. 推荐atb库版本：8.0.0
```

- 配置cann的基本环境
```
  从[社区版资源中心-昇腾社区](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1.alpha002)下载Ascend-cann-kernels-910b_8.2.RC1.alpha002_linux-aarch64.run和 Ascend-cann-toolkit_8.2.RC1.alpha002_linux-aarch64.run（注意，如果是x86_64平台请下载对应的x86版本）并上传到linux服务器。
  
  ```shell
  # 增加可执行权限
  chmod +x Ascend-cann-toolkit_<version>_linux-<arch>.run
  chmod +x Ascend-cann-kernels-<chip_type>_<version>_linux-<arch>.run
  # 校验
  ./Ascend-cann-toolkit_<version>_linux-<arch>.run --check
  ./Ascend-cann-kernels-<chip_type>_<version>_linux-<arch>.run --check
  # 安装
  ./Ascend-cann-toolkit_<version>_linux-<arch>.run --install
  ./Ascend-cann-kernels-<chip_type>_<version>_linux-<arch>.run --install
  # 使用cann相关内容需要设置对应环境变量
  # 非root安装：
  source ${HOME}/Ascend/ascend-toolkit/set_env.sh
  # root安装：
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```
```

- 配置conda环境

```shell
conda create -n enec python=3.9 -y
conda activate enec
wget https://download.pytorch.org/whl/cpu/torch-2.1.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install torch-2.1.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
wget https://gitcode.com/Ascend/pytorch/releases/download/v7.1.0-pytorch2.1.0/torch_npu-2.1.0.post13-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install torch_npu-2.1.0.post13-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install numpy==1.24.3
pip3 install decorator attrs psutil absl-py cloudpickle ml-dtypes scipy tornado pyyaml
```

- 验证环境是否正常
```shell
# 正常输出结果则环境正常
python3 -c "import torch;import torch_npu; a = torch.randn(3, 4).npu(); print(a + a);"
```

- 启动环境并安装
```shell
conda activate enec
bash build_csrc.sh
```
- 数据准备
```python
# 1. 模型下载
您可以根据当前服务器的网络连通性，选择以下任一工具进行下载：
选项 A：使用 ModelScope 命令行
pip install modelscope
modelscope download --model deepseek-ai/deepseek-llm-7b-base --local_dir models/BF16/deepseek-llm-7b-base
选项 B：使用 Hugging Face 命令行
pip install --upgrade huggingface_hub
hf download deepseek-ai/deepseek-llm-7b-base --local-dir models/BF16/deepseek-llm-7b-base
# 2. split model
python python/utils.py --model_path models/BF16/deepseek-llm-7b-base --data_type BF16
# 3. param_search
python python/param_search.py
```
- 运行与测试
```shell
source ${HOME}/Ascend/ascend-toolkit/set_env.sh
# source /data/wja/ascend/ascend-toolkit/set_env.sh

# 1. 压缩
python python/enec_model_compress.py 
# 2. 压缩结果分析
python python/global_analysis_comp.py
# 3. 解压缩
python python/enec_model_decompress.py
# 4. 解压缩结果分析
python python/global_analysis_decomp.py
```