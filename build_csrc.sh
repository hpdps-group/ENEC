#!/bin/bash

# 定义颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # 无颜色

# 1. 加载昇腾开发环境
ASCEND_ENV="/data/wja/ascend/ascend-toolkit/set_env.sh"
if [ -f "$ASCEND_ENV" ]; then
    source "$ASCEND_ENV"
    echo -e "${GREEN}✅ 已加载 Ascend 环境${NC}"
else
    echo -e "${RED}⚠️ 警告: 未找到 set_env.sh${NC}"
fi

# 2. 设置目录
TARGET_ROOT="./csrc"
[ ! -d "$TARGET_ROOT" ] && echo "找不到目录" && exit 1
ABS_ROOT=$(realpath "$TARGET_ROOT")

# --- 初始化统计变量 ---
declare -a FAILED_PROJECTS
SUCCESS_COUNT=0
TOTAL_COUNT=0

# 3. 递归查找并编译
# 注意：这里改用 process substitution 防止 while 循环在子 shell 中运行导致变量丢失
while read -r cmake_file; do
    project_dir=$(dirname "$cmake_file")
    [[ "$project_dir" == *"/build"* ]] && continue

    ((TOTAL_COUNT++))
    rel_path=${project_dir#$ABS_ROOT/}
    
    echo -e "\n${YELLOW}▶ 正在编译 [$TOTAL_COUNT]: $rel_path${NC}"

    pushd "$project_dir" > /dev/null || continue
    [ ! -d "build" ] && mkdir build
    pushd build > /dev/null || continue
    
    # 编译命令
    cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
    make clean > /dev/null 2>&1
    make -j 32 > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  OK${NC}"
        ((SUCCESS_COUNT++))
    else
        echo -e "${RED}  FAILED${NC}"
        # 记录失败的项目相对路径
        FAILED_PROJECTS+=("$rel_path")
    fi

    popd > /dev/null
    popd > /dev/null

done < <(find "$ABS_ROOT" -name "CMakeLists.txt")

# --- 4. 打印汇总报告 ---
echo -e "\n\n================================================"
echo -e "           编译任务汇总统计"
echo -e "================================================"
echo -e "总计项目数: $TOTAL_COUNT"
echo -e "成功数量:   ${GREEN}$SUCCESS_COUNT${NC}"
echo -e "失败数量:   ${RED}${#FAILED_PROJECTS[@]}${NC}"

if [ ${#FAILED_PROJECTS[@]} -ne 0 ]; then
    echo -e "------------------------------------------------"
    echo -e "${RED}以下项目构建失败:${NC}"
    for fail in "${FAILED_PROJECTS[@]}"; do
        echo -e "  ❌ $fail"
    done
    echo -e "------------------------------------------------"
else
    echo -e "------------------------------------------------"
    echo -e "${GREEN}恭喜！所有项目均构建成功。${NC}"
fi
echo -e "================================================\n"