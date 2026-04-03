#!/bin/bash

# 1. 检查 huggingface_hub 是否安装（提供 hf 命令）
if ! command -v hf &> /dev/null; then
    echo "❌ huggingface_hub CLI 未安装，请先安装：pip install -U huggingface_hub[cli]"
    exit 1
fi

# 设置 Hugging Face 镜像源（国内加速）
export HF_ENDPOINT=https://hf-mirror.com

# 2. 模型映射表 (原ID -> Hugging Face ID)
declare -A MODEL_MAP
MODEL_MAP["LLM-Research/OLMo-1B-hf"]="allenai/OLMo-1B-hf"
MODEL_MAP["AI-ModelScope/bert-base-uncased"]="google-bert/bert-base-uncased"
MODEL_MAP["AI-ModelScope/wav2vec2-large-xlsr-53-english"]="jonatasgrosman/wav2vec2-large-xlsr-53-english"
MODEL_MAP["LLM-Research/CapybaraHermes-2.5-Mistral-7B"]="argilla/CapybaraHermes-2.5-Mistral-7B"
MODEL_MAP["AI-ModelScope/stable-video-diffusion-img2vid-fp16"]="stabilityai/stable-video-diffusion-img2vid"
MODEL_MAP["AI-ModelScope/falcon-7b"]="tiiuae/falcon-7b"
MODEL_MAP["AI-ModelScope/falcon-40b"]="tiiuae/falcon-40b"
MODEL_MAP["deepseek-ai/deepseek-llm-7b-base"]="deepseek-ai/deepseek-llm-7b-base"
MODEL_MAP["qwen/Qwen3-8B"]="Qwen/Qwen3-8B"
MODEL_MAP["qwen/Qwen3-32B"]="Qwen/Qwen3-32B"
MODEL_MAP["meta-llama/Llama-3.1-8B-Instruct"]="meta-llama/Llama-3.1-8B-Instruct"

# 精度分组
declare -A PRECISION_GROUPS
PRECISION_GROUPS["FP32"]="LLM-Research/OLMo-1B-hf AI-ModelScope/bert-base-uncased AI-ModelScope/wav2vec2-large-xlsr-53-english"
PRECISION_GROUPS["FP16"]="LLM-Research/CapybaraHermes-2.5-Mistral-7B AI-ModelScope/stable-video-diffusion-img2vid-fp16"
PRECISION_GROUPS["BF16"]="AI-ModelScope/falcon-7b AI-ModelScope/falcon-40b deepseek-ai/deepseek-llm-7b-base qwen/Qwen3-8B qwen/Qwen3-32B meta-llama/Llama-3.1-8B-Instruct"

# 3. 验证阶段（只检查是否可下载，不计算大小）
echo "============================================================"
echo "🔍 正在验证模型可用性..."
echo "============================================================"

python3 << 'EOF'
import os
import requests

endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip('/')
api_url_base = f"{endpoint}/api/models"

model_map = {
    "LLM-Research/OLMo-1B-hf": "allenai/OLMo-1B-hf",
    "AI-ModelScope/bert-base-uncased": "google-bert/bert-base-uncased",
    "AI-ModelScope/wav2vec2-large-xlsr-53-english": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "LLM-Research/CapybaraHermes-2.5-Mistral-7B": "argilla/CapybaraHermes-2.5-Mistral-7B",
    "AI-ModelScope/stable-video-diffusion-img2vid-fp16": "stabilityai/stable-video-diffusion-img2vid",
    "AI-ModelScope/falcon-7b": "tiiuae/falcon-7b",
    "AI-ModelScope/falcon-40b": "tiiuae/falcon-40b",
    "deepseek-ai/deepseek-llm-7b-base": "deepseek-ai/deepseek-llm-7b-base",
    "qwen/Qwen3-8B": "Qwen/Qwen3-8B",
    "qwen/Qwen3-32B": "Qwen/Qwen3-32B",
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
}

print(f"{'ModelScope ID':<50} {'HF ID':<35} {'状态'}")
print("-" * 90)

for ms_id, hf_id in model_map.items():
    url = f"{api_url_base}/{hf_id}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            status = "✅ 可下载"
        elif resp.status_code in (401, 403):
            status = "🔐 需要登录授权"
        elif resp.status_code == 404:
            status = "❌ 仓库不存在"
        else:
            status = f"⚠️ HTTP {resp.status_code}"
    except Exception as e:
        status = f"⚠️ 请求失败: {str(e)}"
    
    print(f"{ms_id:<50} {hf_id:<35} {status}")

print("=" * 90)
EOF

# 4. 等待 10 秒后自动开始下载（用户可按任意键跳过等待）
echo ""
echo "10 秒后自动开始下载可用的模型..."
echo "按任意键立即开始，或按 Ctrl+C 取消。"
read -t 10 -n 1

# 5. 下载阶段
echo ""
echo "============================================================"
echo "🚀 开始下载模型..."
echo "============================================================"

for PRECISION in FP32 FP16 BF16; do
    echo "------------------------------------------------"
    echo "📁 目标精度目录：models/$PRECISION"
    echo "------------------------------------------------"

    for MS_ID in ${PRECISION_GROUPS[$PRECISION]}; do
        HF_ID="${MODEL_MAP[$MS_ID]}"
        if [[ -z "$HF_ID" ]]; then
            echo "⚠️ 未找到映射: $MS_ID，跳过"
            continue
        fi

        LOCAL_DIR_NAME="${HF_ID##*/}"
        LOCAL_PATH="models/${PRECISION}/${LOCAL_DIR_NAME}"

        echo "⬇️ 正在下载: $HF_ID"
        echo "📍 存储位置: $LOCAL_PATH"

        # 移除 --resume 参数（新版 hf download 不支持）
        hf download "$HF_ID" --local-dir "$LOCAL_PATH"

        if [ $? -eq 0 ]; then
            echo "✅ $LOCAL_DIR_NAME 下载/校验成功"
        else
            echo "❌ $LOCAL_DIR_NAME 下载失败，请检查网络或模型权限"
        fi
    done
done

echo "============================================================"
echo "🎉 所有任务处理完毕"