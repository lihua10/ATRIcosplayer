# -*- coding: utf-8 -*-
# train.py
import os
import shutil
# 设置 CUDA 内存分配相关环境变量，降低内存碎片风险
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import json
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm.auto import tqdm
import time
import traceback
import sys
import gc
import psutil
from torch.utils import cpp_extension

# ================ 配置参数 ==================
MODEL_PATH = r"C:\Users\LH\.cache\huggingface\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\14dd1130311655b43c3ce41dd505f70f6ca89845"
DATA_PATH = "data/dataset.json"  # 数据文件，格式为 JSON，包含 "data" 字段
OUTPUT_DIR = r"F:\output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================ 量化配置（4-bit） ==================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # 启用双重量化
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# ================ 加载模型和 Tokenizer ==================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# 修改前
# tokenizer.add_special_tokens(special_tokens_dict)

# 修改后
custom_tokens = [
    "乃音子", "亚托莉",
    "凛凛花", "凯瑟琳",
    "水菜萌", "洋子", "用户",
    "陌生人", "龙司"
]
num_added = tokenizer.add_tokens(custom_tokens)  # 正确方式

# 系统标记保持特殊标记
system_tokens = ["<|assistant|>", "<|user|>", "<|system|>"]
tokenizer.add_special_tokens({"additional_special_tokens": system_tokens})
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_tokenizer"))
# 调整嵌入层大小以匹配分词器（非常关键！）
model.resize_token_embeddings(len(tokenizer))

# 若模型未设置 pad_token，则使用 eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ================ 附加 LoRA Adapter ==================
# 根据需要调整 r, lora_alpha, lora_dropout 以及 task_type
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,  # 秩，可根据需要调整
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # 原始适配的注意力层
        "embed_tokens",  # 嵌入层
        "lm_head"       # 输出层
    ],
    modules_to_save=[],  # 确保不保存任何层的全量参数
    bias="none"
)
model = get_peft_model(model, lora_config)

# ================ 数据加载 ==================
def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON格式错误在 {e.lineno} 行 {e.colno} 列")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            start = max(0, e.lineno - 3)
            end = min(len(lines), e.lineno + 2)
            print("错误上下文：\n" + "".join(lines[start:end]))
        exit(1)
    # 数据中每项应包含 "prompt" 和 "response"
    processed = []
    for item in data:
        context = "".join(item["context"]).strip()
        prompt = "".join(item["input"]).strip()
        response = "".join(item["output"]).strip()
        processed.append({
            "context" : context,
            "prompt": prompt,
            "response": response
        })

    return Dataset.from_list(processed)

raw_dataset = load_data(DATA_PATH)

def tokenize_example(example):
    # 构造训练文本：包含 system、user、assistant 三个部分
    full_text = (
        "<|system|>\n你是一个名为亚托莉的少女，请用角色身份回答。请注意，回答时请不要输出特殊标记尖括号（<>）（但输出其中包裹的人名）,也不要输出<|system|>、<|user|>、<|assistant|>，这些特殊标记仅用于内部提示。\n以下是上下文背景：\n"+ example["context"]+"\n"
        "<|user|>\n" + example["prompt"].strip() + "\n"
        "<|assistant|>\n" + example["response"].strip()


    )
    # 对 <用户> 部分裁剪一半以节省显存并保留紧跟 Assistant 的上下文
    # full_text = shorten_user_section(full_text, keep_ratio=0.5)
    
    tokenized = tokenizer(full_text, truncation=False)
    decoded = tokenizer.decode(tokenized["input_ids"])
    # print("Decoded full text:\n", decoded)
    
    assistant_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    input_ids = tokenized["input_ids"]
    labels = input_ids.copy()
    if assistant_id in input_ids:
        assistant_index = input_ids.index(assistant_id)
        if assistant_index + 1 >= len(input_ids):
            tokenized["labels"] = input_ids.copy()
        else:
            for i in range(assistant_index + 1):
                labels[i] = -100
            tokenized["labels"] = labels
    else:
        tokenized["labels"] = input_ids.copy()
    
    # 调试：确保存在非-100的label
    valid_tokens = sum(1 for t in tokenized["labels"] if t != -100)
    if valid_tokens == 0:
        print("警告：该样例所有 token 均为 mask(-100)")
    
    return tokenized




# 对整个数据集应用 tokenize
tokenized_dataset = raw_dataset.map(tokenize_example, remove_columns=raw_dataset.column_names)

# 计算所有样本的长度
lengths = [len(x["input_ids"]) for x in tokenized_dataset]
sorted_lengths = sorted(lengths)

# 舍弃后 5% 超长数据，并使用剩余 95% 样本中的最大长度作为固定 padding 长度
cutoff_index = int(len(sorted_lengths) * 0.95)
# 注意：如果 n=100，cutoff_index 为95，对应索引 0~94，共 95 个样本；此时取最大值为 sorted_lengths[94]
global_max_length = sorted_lengths[cutoff_index - 1]
print(f"使用前95%数据中的最大token长度作为固定 padding 长度: {global_max_length}")

# 过滤掉长度超过 global_max_length 的样本（直接舍弃后 5% 的数据，之后永不参与计算）
def filter_extreme_samples(example):
    return len(example["input_ids"]) <= global_max_length

tokenized_dataset = tokenized_dataset.filter(filter_extreme_samples)

# 打印一个样本，检查 token 化结果是否合理
# print("示例 tokenized 数据：", tokenized_dataset[0])

# ================ 自定义 Collator ==================
def custom_data_collator(features):
    """
    自定义 collator：对 input_ids、attention_mask、labels 使用全局固定的 pad 长度进行 padding。
    """
    # 分别提取各个字段
    input_ids = [f['input_ids'] for f in features]
    attention_masks = [f['attention_mask'] for f in features]
    labels = [f['labels'] for f in features]

    # 使用预先计算好的全局固定 pad 长度，即 global_max_length
    pad_length = global_max_length

    # 对 input_ids 进行 padding：tokenizer.pad_token_id 填充；attention_mask 填充 0；labels 填充 -100
    padded_input_ids = [ids + [tokenizer.pad_token_id] * (pad_length - len(ids)) for ids in input_ids]
    padded_attention_masks = [mask + [0] * (pad_length - len(mask)) for mask in attention_masks]
    padded_labels = [lab + [-100] * (pad_length - len(lab)) for lab in labels]

    batch = {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long)
    }
    return batch

# ================ 新增配置管理类 ==================
class TrainingConfigManager:
    CONFIG_FILE = "training_config.json"
    
    def __init__(self, initial_config):
        self.config = initial_config
        self.load_config()
    
    def load_config(self):
        if os.path.exists(self.CONFIG_FILE):
            with open(self.CONFIG_FILE, 'r') as f:
                saved_config = json.load(f)
                self.config.update(saved_config)
    
    def save_config(self):
        with open(self.CONFIG_FILE, 'w') as f:
            json.dump(self.config, f)


# ================ 修改训练参数初始化 ==================
initial_config = {
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 4,
    'learning_rate': 1e-5,
    'max_retries': 20  # 最大重试次数
}

config_manager = TrainingConfigManager(initial_config)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=config_manager.config['per_device_train_batch_size'],
    gradient_accumulation_steps=config_manager.config['gradient_accumulation_steps'],
    learning_rate=config_manager.config['learning_rate'],
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="steps",
    save_steps=100,

    save_total_limit=3,  # 增加保存数量限制
    report_to="none",
    remove_unused_columns=False,
    load_best_model_at_end=False,
    optim="paged_adamw_8bit",  # 使用分页优化器
    dataloader_pin_memory=False,  # 减少内存占用

    bf16=True,   # 启用 BF16 模式
    fp16=False   # 确保不使用 FP16
)

# ================ 自定义日志回调 ==================
class TrainProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            current_lr = logs.get("learning_rate", 0)
            mem = f"{torch.cuda.memory_reserved()//1024//1024 if torch.cuda.is_available() else 0}MB"
            epoch_info = f"{state.epoch:.2f}"
            loss_info = f"{logs.get('loss', 0):.4f}"
            lr_info = f"{current_lr:.2e}" if isinstance(current_lr, float) else current_lr
            time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time))
            print(f"{epoch_info:<7}{mem:>10}{loss_info:>12}{lr_info:>12}{time_elapsed:>10}")

# ================ 在训练脚本中添加回调 ==================
class SaveTokenizerCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        # 强制整理内存碎片
        # torch.cuda.memory._record_memory_history()
        # torch.cuda.memory._dump_snapshot()
        gc.collect()
        
        output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(
            output_dir,
            safe_serialization=True,
            save_embedding_layers="unmerged"  # 确保不合并
        )

# ================ 创建内存监控回调 ==================
class MemoryMonitorCallback(TrainerCallback):
    def __init__(self, log_steps=10):
        self.log_steps = log_steps
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_steps == 0:
            print_memory_usage(f"Step {state.global_step}")
            print_gpu_memory_info()

# ================ 修改训练器初始化 ==================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=custom_data_collator,
    callbacks=[
        TrainProgressCallback(),
        SaveTokenizerCallback(),
        MemoryMonitorCallback(log_steps=5)  # 每5步记录一次
    ]
)

# 禁用梯度检查点（确保不要调用 enable）
model.gradient_checkpointing_disable()
model.config.use_cache = False  # 确保缓存关闭
assert model.training  # 必须处于训练模式

# 打印可训练参数
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"可训练参数: {name} | 形状: {param.shape}")

# 预期输出应包含类似以下内容（具体名称因模型而异）：
# "base_model.model.model.embed_tokens.lora_A.weight"
# "base_model.model.model.embed_tokens.lora_B.weight"
# "base_model.model.lm_head.lora_A.weight"
# "base_model.model.lm_head.lora_B.weight"

# 确保保存完整配置
model.save_pretrained(
    OUTPUT_DIR,
    safe_serialization=True,
    save_embedding_layers="unmerged"  # 改为不合并权重
)

# 初始化进程监控
process = psutil.Process()

def print_memory_usage(prefix=""):
    # 获取CPU内存 (RSS)
    cpu_mem = process.memory_info().rss // 1024 // 1024
    # 获取GPU显存
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() // 1024 // 1024
        gpu_cache = torch.cuda.memory_reserved() // 1024 // 1024
        print(f"{prefix} CPU内存: {cpu_mem}MB | GPU显存: {gpu_mem}MB | 缓存池: {gpu_cache}MB")
    else:
        print(f"{prefix} CPU内存: {cpu_mem}MB")

# 在训练循环中定期调用

def print_gpu_memory_info(device=0):
    # 总显存（单位字节）
    total_memory = torch.cuda.get_device_properties(device).total_memory

    # 当前PyTorch分配并保留的显存（单位字节）
    reserved_memory = torch.cuda.memory_reserved(device)
    # 当前真正使用中的显存（单位字节）
    allocated_memory = torch.cuda.memory_allocated(device)

    # 根据PyTorch缓存机制计算"空闲"的显存
    free_memory = total_memory - reserved_memory

    print(f"设备 {device} 总显存: {total_memory//1024//1024}MB")
    print(f"已保留显存: {reserved_memory//1024//1024}MB")
    print(f"已使用显存: {allocated_memory//1024//1024}MB")
    print(f"可分配空闲显存: {free_memory//1024//1024}MB")




if __name__ == "__main__":
    while True:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            trainer.train(resume_from_checkpoint=False)
            break
        except Exception as e:
            print(e)

    # 最终保存
    model.to("cpu")
    
    # 确保分词器 pad_token 设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 保存适配器和分词器
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_tokenizer"))
