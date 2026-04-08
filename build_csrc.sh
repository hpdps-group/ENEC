#!/bin/bash

# --- 0. Force Environment Cleanup ---
# Key to solving "Text file busy": Kill old running processes
echo "Cleaning up residual processes..."
pkill -9 -f "compress" || true
pkill -9 -f "decompress" || true
sleep 0.5 # Give the kernel time to release file handles

# Define Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Load Ascend Development Environment
ASCEND_ENV="/data/wja/ascend/ascend-toolkit/set_env.sh"
if [ -f "$ASCEND_ENV" ]; then
    source "$ASCEND_ENV"
    echo -e "${GREEN}✅ Ascend environment loaded${NC}"
else
    echo -e "${RED}⚠️ Warning: set_env.sh not found${NC}"
fi

# 2. Set Directories
TARGET_ROOT="./csrc"
if [ ! -d "$TARGET_ROOT" ]; then
    echo -e "${RED}Error: Directory $TARGET_ROOT not found${NC}"
    exit 1
fi
ABS_ROOT=$(realpath "$TARGET_ROOT")

# Define and create directory for executables
EXEC_DIR="$ABS_ROOT/exec"
mkdir -p "$EXEC_DIR"
echo -e "${GREEN}📂 Executables will be stored in: $EXEC_DIR${NC}"

# --- Initialize Statistics Variables ---
declare -a FAILED_PROJECTS
SUCCESS_COUNT=0
TOTAL_COUNT=0

# 3. Recursively Find and Compile
# Use -prune or filtering to skip existing build directories for efficiency
while read -r cmake_file; do
    project_dir=$(dirname "$cmake_file")
    
    # Skip existing build directories and their subdirectories
    [[ "$project_dir" == *"/build"* ]] && continue

    ((TOTAL_COUNT++))
    rel_path=${project_dir#$ABS_ROOT/}
    
    # Extract directory name as suffix (e.g., ENEC-BF16-121-6-3-16)
    suffix=$(basename "$project_dir")
    
    echo -e "\n${YELLOW}▶ Compiling [$TOTAL_COUNT]: $rel_path${NC}"

    pushd "$project_dir" > /dev/null || continue
    
    # Prepare build directory
    mkdir -p build
    pushd build > /dev/null || continue
    
    # Execute compilation logic (Silent mode)
    cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
    make clean > /dev/null 2>&1
    make -j 32 > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  Compilation successful${NC}"
        ((SUCCESS_COUNT++))

        # --- Core Correction: Resolve Overwrite Conflicts ---
        for exe_name in "compress" "decompress"; do
            if [ -f "$exe_name" ]; then
                target_exe="$EXEC_DIR/${exe_name}_${suffix}"
                # Critical: rm before cp to prevent "Text file busy"
                rm -f "$target_exe" 
                if cp "$exe_name" "$target_exe"; then
                    echo -e "  └─ Extracted: ${exe_name}_${suffix}"
                    chmod +x "$target_exe" # Ensure execution permissions
                else
                    echo -e "  ${RED}└─ Extraction failed: $target_exe (possibly still in use)${NC}"
                fi
            fi
        done
    else
        echo -e "${RED}  Compilation failed${NC}"
        FAILED_PROJECTS+=("$rel_path")
    fi

    popd > /dev/null # Exit build
    popd > /dev/null # Exit project directory

done < <(find "$ABS_ROOT" -name "CMakeLists.txt")

# --- 4. Print Summary Report ---
echo -e "\n\n================================================"
echo -e "         Build & Extraction Task Summary"
echo -e "================================================"
echo -e "Total Projects:   $TOTAL_COUNT"
echo -e "Success Count:    ${GREEN}$SUCCESS_COUNT${NC}"
echo -e "Failure Count:    ${RED}${#FAILED_PROJECTS[@]}${NC}"
echo -e "Output Directory: $EXEC_DIR"

if [ ${#FAILED_PROJECTS[@]} -ne 0 ]; then
    echo -e "------------------------------------------------"
    echo -e "${RED}The following projects failed to build:${NC}"
    for fail in "${FAILED_PROJECTS[@]}"; do
        echo -e "  ❌ $fail"
    done
    echo -e "------------------------------------------------"
else
    echo -e "------------------------------------------------"
    echo -e "${GREEN}Congratulations! All projects built and extracted successfully.${NC}"
fi
echo -e "================================================\n"