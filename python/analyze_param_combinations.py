import pandas as pd
import sys
import os

def analyze_csv(file_path):
    # 1. 基本文件检查
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        # 2. 读取数据
        df = pd.read_csv(file_path)
        
        # 3. 确定目标列
        # 你的截图中显示有 b, n, m，假设 L 也在表格中
        required_columns = ['b', 'n', 'm', 'L']
        
        # 容错：检查哪些列确实存在
        available_columns = [col for col in required_columns if col in df.columns]
        
        if not available_columns:
            print(f"Error: None of the parameters {required_columns} found in CSV.")
            print(f"Available columns: {df.columns.tolist()}")
            return

        # 4. 提取唯一组合并统计出现次数
        # groupby 会把相同的参数聚合在一起，size() 统计数量
        combinations = df.groupby(available_columns).size().reset_index(name='count')
        
        # 5. 格式化输出
        print("\n" + "="*50)
        print(f" 参数组合分析报告 ({os.path.basename(file_path)})")
        print("="*50)
        print(f"{'  '.join([col.center(5) for col in available_columns])} | {'Count'.center(8)}")
        print("-" * 50)
        
        for _, row in combinations.iterrows():
            params = "  ".join([str(int(row[col])).center(5) for col in available_columns])
            count = str(int(row['count'])).center(8)
            print(f"{params} | {count}")
            
        print("-" * 50)
        print(f"总计: 共有 {len(combinations)} 种不同的参数组合。")
        print("="*50 + "\n")

    except Exception as e:
        print(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    # 检查命令行输入
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <path_to_csv>")
        print("Example: python analyze.py ./hyperparams_results.csv")
    else:
        # sys.argv[1] 获取命令行传入的第一个参数（路径）
        analyze_csv(sys.argv[1])