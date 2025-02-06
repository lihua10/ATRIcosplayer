import os
import glob
import json
from tqdm import tqdm
import re

dir_path = "F:\\game\\BaiduNetdiskDownload\\ATRI\\ATRI -My Dear Moments-\\output\\text\\parsed"
output_path = "data\\dataset.json"

# 在文件顶部添加全局集合
role_tags = set()

def validate_entry(entry):
    """验证数据条目格式"""
    required_keys = ["lines_index", "line_index", "context", "input", "output"]
    return all(key in entry for key in required_keys)

def read_txt_files(folder_path):
    """读取原始文本文件，保留行首的\t"""
    all_files = glob.glob(os.path.join(folder_path, "**/*.txt"), recursive=True)
    all_lines = []
    
    for file_path in all_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 修改：只去除换行符，保留行首空白
            lines = [line.rstrip('\n') for line in f if line.rstrip('\n')]
            all_lines.append(lines)
    
    return all_lines

def preprocess_line(line):
    global role_tags
    """预处理文本行，处理角色标签和替换指定名称"""
    # 优先处理旁白行（保留原始空白）
    if line.startswith('\t'):
        content = line[1:].strip()  # 去除\t和两端空白
        line =  f'({content})' if content else ''
    
    # 处理其他内容（去除两端空白）
    line = line.strip()
    
    if line.startswith("亚托莉\t"):
        line =  f"亚托莉：{line[4:]}"
    
    if "\t" in line:
        role, content = line.split("\t", 1)
        role = role.replace("斑鸠夏生", "夏生").replace("夏生", "用户")
        content = content.replace("斑鸠夏生", "夏生").replace("夏生", "用户")
        line = f"{role}：{content}"
    else:
        line = line.replace("斑鸠夏生", "夏生").replace("夏生", "用户").replace("我们","用户和亚托莉").replace("我","用户")

    # 新增标签收集逻辑（处理所有<...>形式的标签）
    tags = re.findall(r'<([^>]+)>', line)
    for tag in tags:
        role_tags.add(f"<{tag}>")


    return line

def file_reader(dir_path):
    all_lines = []
    txt_files = glob.glob(os.path.join(dir_path, '*.txt'))
    
    print(f"发现 {len(txt_files)} 个文本文件")
    for file_path in tqdm(txt_files, desc="读取文件中"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [preprocess_line(line.rstrip('\n')) for line in f if line.rstrip('\n')]
                all_lines.append(lines)
        except UnicodeDecodeError:
            print(f"\n警告：文件 {os.path.basename(file_path)} 编码错误，已跳过")
    return all_lines

def file_writer(dataset):
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 修改输出结构为纯数组
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset["data"], f, ensure_ascii=False, indent=2)
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
    global role_tags
    dataset = {"data": []}
    error_count = 0
    
    print("\n开始处理数据...")
    for lines_idx, lines in enumerate(tqdm(all_lines, desc="处理对话数据")):
        line_idx = 0
        while line_idx < len(lines):
            current_line = lines[line_idx]
            
            # 检测亚托莉对话起始点
            if current_line.startswith("亚托莉"):
                # 记录起始位置
                start_idx = line_idx
                
                # 合并连续内容（亚托莉对话和旁白）
                while line_idx < len(lines):
                    current_line = lines[line_idx]
                    # 允许合并的类型：亚托莉对话或旁白
                    if not (current_line.startswith("亚托莉") or current_line.startswith("(")):
                        break
                    line_idx += 1
                
                line_idx_user = start_idx -1
                # 合并连续内容（用户提问和旁白）
                while line_idx_user > 0:
                    current_line = lines[line_idx_user]

                    if not (current_line.startswith("用户") or current_line.startswith("(")):
                        break
                    line_idx_user -= 1




                # 获取上下文范围（修正结束位置）
                context_start = max(0, line_idx_user - 7)
                context_end = max(0, line_idx_user)  # 排除用户输入行
                context = lines[context_start : context_end]
                
                
                # 构建条目（保留旁白的括号）
                entry = {
                    "lines_index": lines_idx,
                    "line_index": line_idx_user,
                    "context": context,
                    "input": [
                        line.replace("用户：", "") if line.startswith("用户")
                        else line  # 旁白行保持原样
                        for line in lines[line_idx_user+1:start_idx]
                    ],


                    "output": [
                        line.replace("亚托莉：", "") if line.startswith("亚托莉")
                        else line  # 旁白行保持原样
                        for line in lines[start_idx:line_idx]
                    ]
                }
                
                if len(entry["input"]) > 0:
                    dataset["data"].append(entry)
            else:
                line_idx += 1
    
    print(f"处理完成，有效数据：{len(dataset['data'])} 条，错误数据：{error_count} 条")
    
    # 处理完成后打印唯一标签
    print("\n发现的所有角色标签：")
    print('[' + ', '.join(f'"{tag}"' for tag in sorted(role_tags)) + ']')
    
    return dataset

if __name__ == "__main__":
    all_lines = file_reader(dir_path)
    processed_data = data_process(all_lines)
    file_writer(processed_data)