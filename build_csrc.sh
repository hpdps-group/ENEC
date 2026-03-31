#!/bin/bash

# 1. 加载昇腾开发环境（建议在编译前加载，确保 cmake 能找到 NPU 相关库）
if [ -f "/data/wja/ascend/ascend-toolkit/set_env.sh" ]; then
    source /data/wja/ascend/ascend-toolkit/set_env.sh
    echo "已加载 Ascend 环境。"
else
    echo "警告: 未找到 set_env.sh，编译可能会失败。"
fi

# 2. 设置目标目录
TARGET_DIR="./csrc"

if [ ! -d "$TARGET_DIR" ]; then
    echo "错误: 找不到目录 $TARGET_DIR"
    exit 1
fi

# 获取绝对路径
ABS_TARGET_DIR=$(realpath "$TARGET_DIR")

# 3. 遍历 csrc 下的一级子目录
for dir in "$ABS_TARGET_DIR"/*/; do
    dir=${dir%/}
    
    if [ -f "$dir/CMakeLists.txt" ]; then
        echo "================================================"
        echo "正在处理项目: $(basename "$dir")"
        echo "================================================"

        # 进入项目目录
        pushd "$dir" > /dev/null || continue

        # 4. 修改点：如果不存在 build 则创建，不再执行删除操作
        if [ ! -d "build" ]; then
            echo "创建新的 build 目录..."
            mkdir build
        fi

        # 进入 build 目录
        pushd build > /dev/null || continue
        
        # 5. 执行编译
        # 注意：如果不删除 build，cmake 会利用之前的缓存进行增量配置
        make clean
        cmake ..
        make -j 32
        
        # 检查编译状态
        if [ $? -eq 0 ]; then
            echo "✅ $(basename "$dir") 编译/更新成功！"
        else
            echo "❌ $(basename "$dir") 编译发生错误！"
        fi

        # 6. 回到初始目录
        popd > /dev/null # 退出 build
        popd > /dev/null # 退出项目目录
        
    else
        echo "跳过 $dir (未发现 CMakeLists.txt)"
    fi
done

echo "------------------------------------------------"
echo "所有项目处理完毕。"