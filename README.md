# ENEC
- Directory Structure
```
- csrc: NPU kernel implementation
- param_search_enec: Parameter search results
- python: Data processing, parameter searching, compression testing, and profiling result collection
- results_enec: Final output results
```
- System and Software Requirements
```
  1. Recommended Platform: Linux (Ubuntu 22.04), aarch64
  2. Recommended Python Version: Python 3.9
  3. NPU：Ascend 910B2
  4. Recommended CANN Version: 8.2.RC1.alpha002
  5. Recommended CANN Kernels Version: 8.2.RC1.alpha002
  6. Recommended ATB Library Version: 8.0.0
```

- Basic CANN Environment Configuration
```
  From the [community resource center - Ascend](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1.alpha002), download the Ascend-cann-kernels-910b_8.2.RC1.alpha002_linux-aarch64.run and Ascend-cann-toolkit_8.2.RC1.alpha002_linux-aarch64.run files (note: for x86_64, download the x86 version) and upload them to your Linux server.
  
  ```shell
  # Add executable permissions
  chmod +x Ascend-cann-toolkit_<version>_linux-<arch>.run
  chmod +x Ascend-cann-kernels-<chip_type>_<version>_linux-<arch>.run
  # Verification
  ./Ascend-cann-toolkit_<version>_linux-<arch>.run --check
  ./Ascend-cann-kernels-<chip_type>_<version>_linux-<arch>.run --check
  # Installation
  ./Ascend-cann-toolkit_<version>_linux-<arch>.run --install
  ./Ascend-cann-kernels-<chip_type>_<version>_linux-<arch>.run --install
  # Using cann related content requires setting corresponding environment variables
  # Non-root installation：
  source ${HOME}/Ascend/ascend-toolkit/set_env.sh
  # Root Installation：
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```
```

- Configure the conda environment

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

- Verify that the environment is working
```shell
# If the output is normal, the environment is normal
python3 -c "import torch;import torch_npu; a = torch.randn(3, 4).npu(); print(a + a);"
```

- Start the environment and install it
```shell
conda activate enec
git clone https://github.com/hpdps-group/ENEC.git
bash build_csrc.sh
```
- Data preparation
```shell
bash data_prepare.sh
```
- Running and Testing
```shell
# 1. Compression ratio and compression throughput
bash compressor_test.sh
# 2. Inference
bash run_inference.sh
```