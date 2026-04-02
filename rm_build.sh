#!/bin/bash

# 设置目标目录
TARGET_DIR="./csrc"

if [ ! -d "$TARGET_DIR" ]; then
    echo "错误: 找不到目录 $TARGET_DIR"
    exit 1
fi

# 获取绝对路径
ABS_TARGET_DIR=$(realpath "$TARGET_DIR")

echo "开始清理所有项目中的 build 目录..."
echo "------------------------------------------------"

# 遍历 csrc 下的一级子目录
for dir in "$ABS_TARGET_DIR"/*/; do
    dir=${dir%/}
    
    # 检查是否存在 build 目录
    if [ -d "$dir/build" ]; then
        echo "正在清理: $(basename "$dir")/build"
        # 强制删除 build 文件夹
        rm -rf "$dir/build"
        
        if [ $? -eq 0 ]; then
            echo "✅ 已删除"
        else
            echo "❌ 删除失败 (请检查权限)"
        fi
    else
        echo "跳过: $(basename "$dir") (无 build 目录)"
    fi
done

echo "------------------------------------------------"
echo "所有 build 目录清理完毕。"