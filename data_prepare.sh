#!/bin/bash

# 1. Execute data preparation script
echo "Running download_data.sh..."
bash download_data.sh

# 2. Run Python splitting/utility script
echo "Running python/utils.py..."
python3 python/utils.py

echo "All tasks completed!"