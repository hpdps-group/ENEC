#!/bin/bash

# Set target directory
TARGET_DIR="./csrc"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR not found"
    exit 1
fi

# Get absolute path
ABS_TARGET_DIR=$(realpath "$TARGET_DIR")

echo "Starting to clean all 'build' directories under $ABS_TARGET_DIR..."
echo "------------------------------------------------"

# Use find command to locate all directories named 'build'
# -type d specifies directories
# -name "build" matches the name
find "$ABS_TARGET_DIR" -type d -name "build" | while read -r build_path; do
    echo "Cleaning: $build_path"
    
    # Force removal
    rm -rf "$build_path"
    
    if [ $? -eq 0 ]; then
        echo "✅ Deleted"
    else
        echo "❌ Delete failed (Please check permissions)"
    fi
done

echo "------------------------------------------------"
echo "All 'build' directories have been cleaned."