#!/bin/bash

# NOTE: 'set -e' is removed so the script continues checking remaining files 
# and counts errors even if one comparison fails.
echo "Starting full process: data compression, verification, and analysis..."

# Non-root installation: 
# source ${HOME}/Ascend/ascend-toolkit/set_env.sh
# Root Installation:
# source /usr/local/Ascend/ascend-toolkit/set_env.sh

export TORCH_NPU_DISABLED_WARNING=1
export TORCHDYNAMO_DISABLE=1

# 0. Search Hyperparameters
python python/param_search_enec.py

# Activate environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate enec

# 1. Execute Model Compression
echo "Step 1/5: Compressing models..."
python python/enec_model_compress_compressor.py

# 2. Execute Model Decompression
echo "Step 2/5: Decompressing models..."
python python/enec_model_decompress_compressor.py

# --- Step 2.5: Automated Lossless Verification ---
echo "Step 2.5: Running data consistency check (cmp)..."

RESULTS_ROOT="./results_enec"
MODELS_ROOT="./models"
FAILED_LOG="verification_failures.csv"

# Initialize CSV file with headers
echo "model_name,dtype,filename,status" > "$FAILED_LOG"

FAILED_COUNT=0
SUCCESS_COUNT=0

while read -r decomp_file; do
    # Extract path information
    rel_path=${decomp_file#$RESULTS_ROOT/}
    dtype=$(echo "$rel_path" | cut -d'/' -f1)
    model_name=$(echo "$rel_path" | cut -d'/' -f2)
    filename=$(basename "$decomp_file" .decompressed)
    
    original_file="$MODELS_ROOT/$dtype/$model_name/split/$filename.bin"

    if [ -f "$original_file" ]; then
        if cmp -s "$decomp_file" "$original_file"; then
            ((SUCCESS_COUNT++))
        else
            echo -e "\033[0;31m❌ [DIFF] Inconsistency: $filename in $model_name ($dtype)\033[0m"
            # Log failure to CSV
            echo "$model_name,$dtype,$filename,mismatch" >> "$FAILED_LOG"
            ((FAILED_COUNT++))
        fi
    else
        echo -e "\033[1;33m⚠️ [MISS] Missing original: $original_file\033[0m"
        echo "$model_name,$dtype,$filename,missing_original" >> "$FAILED_LOG"
    fi

done < <(find "$RESULTS_ROOT" -name "*.decompressed")

echo "-------------------------------------------"
echo "Verification Summary: Success: $SUCCESS_COUNT, Failed: $FAILED_COUNT"
if [ $FAILED_COUNT -ne 0 ]; then
    echo -e "\033[0;31mWARNING: $FAILED_COUNT files failed. Details saved to $FAILED_LOG\033[0m"
else
    echo -e "\033[0;32mCongratulations: All files passed lossless verification.\033[0m"
    # Optional: remove the log if empty
    rm -f "$FAILED_LOG"
fi
echo "-------------------------------------------"

# 3. Global Analysis
echo "Step 3/5: Analyzing compression performance..."
python python/global_analysis_comp_enec.py

# 4. Global Analysis
echo "Step 4/5: Analyzing decompression performance..."
python python/global_analysis_decomp_enec.py

# 5. Summary Report
echo "Step 5/5: Generating experiment summary..."
python python/summarization_enec.py

echo "-------------------------------------------"
echo "All tasks completed successfully!"