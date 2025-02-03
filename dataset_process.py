import json
from datasets import Dataset

def process_dataset(input_file, output_file):
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)["data"]
    
    processed_data = []
    
    for item in raw_data:
        # 处理上下文（保留换行符）
        context = "".join([
            line.replace("\t", "　")  # 替换制表符为全角空格
            .replace("\n", "\\n")     # 保留换行标记
            for line in item["context"]
        ])
        
        # 处理输入（对话行）
        input_line = item["input"].strip().replace("\t", "　")
        
        # 构建完整prompt
        prompt = f"{context}{input_line}"
        
        # 处理输出（保留换行符和特殊符号）
        response = "".join([
            line.replace("\t", "　")
            .replace("\n", "\\n")
            for line in item["output"]
        ]).strip()
        
        processed_data.append({
            "prompt": prompt,
            "response": response
        })
    
    # 转换为Dataset并保存
    dataset = Dataset.from_list(processed_data)
    dataset.to_json(output_file)

if __name__ == "__main__":
    input_path = "data/dataset.txt"  # 原始数据路径
    output_path = "data/dataset.json"  # 输出路径
    process_dataset(input_path, output_path)