import os
import glob
import json
from tqdm import tqdm

dir_path = "F:\\game\\BaiduNetdiskDownload\\ATRI\\ATRI -My Dear Moments-\\output\\text\\parsed"
output_path = "data\\dataset.json"

def validate_entry(entry):
    """验证数据条目格式"""
    required_keys = ["lines_index", "line_index", "context", "input", "output"]
    return all(key in entry for key in required_keys)

def file_reader(dir_path):
    all_lines = []
    txt_files = glob.glob(os.path.join(dir_path, '*.txt'))
    
    print(f"发现 {len(txt_files)} 个文本文件")
    for file_path in tqdm(txt_files, desc="读取文件中"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                all_lines.append(lines)
        except UnicodeDecodeError:
            print(f"\n警告：文件 {os.path.basename(file_path)} 编码错误，已跳过")
    return all_lines

def file_writer(dataset):
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 构建标准JSON结构
    output_data = {
        "metadata": {
            "total_samples": len(dataset["data"]),
            "source_directory": dir_path
        },
        "data": dataset["data"]
    }
    
    # 带格式验证的写入
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n成功生成数据集，共 {len(dataset['data'])} 条数据")
        
        # 验证生成的文件
        with open(output_path, 'r', encoding='utf-8') as f:
            json.load(f)
        print("JSON格式验证通过")
    except Exception as e:
        print(f"\n写入文件时发生错误：{str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)

def data_process(all_lines):
    dataset = {"data": []}
    error_count = 0
    
    print("\n开始处理数据...")
    for lines_idx, lines in enumerate(tqdm(all_lines, desc="处理对话数据")):
        line_idx = 0
        while line_idx < len(lines):
            current_line = lines[line_idx]
            
            # 跳过空行
            if not current_line:
                line_idx += 1
                continue
                
            try:
                # 亚托莉对话块处理
                if current_line.startswith("亚托莉"):
                    # 合并连续亚托莉对话
                    end_idx = line_idx + 1
                    while end_idx < len(lines) and lines[end_idx].startswith("亚托莉"):
                        end_idx += 1
                    
                    # 上下文处理
                    context_start = max(0, line_idx - 51) if line_idx >= 51 else 0
                    context_end = max(0, line_idx - 1)
                    
                    entry = {
                        "lines_index": lines_idx,
                        "line_index": line_idx,
                        "context": lines[context_start:context_end],
                        "input": lines[context_end] if context_end >=0 else "",
                        "output": lines[line_idx:end_idx]
                    }
                    
                    if validate_entry(entry):
                        dataset["data"].append(entry)
                    else:
                        error_count += 1
                    
                    line_idx = end_idx
                else:
                    line_idx += 1
            except IndexError:
                error_count += 1
                line_idx += 1
    
    print(f"处理完成，有效数据：{len(dataset['data'])} 条，错误数据：{error_count} 条")
    return dataset

if __name__ == "__main__":
    all_lines = file_reader(dir_path)
    processed_data = data_process(all_lines)
    file_writer(processed_data)