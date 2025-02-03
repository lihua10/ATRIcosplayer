# -*- coding: utf-8 -*-
# train.py
import os
# 设置 CUDA 内存分配相关环境变量，降低内存碎片风险
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
import time

# ================ 配置参数 ==================
MODEL_PATH = r"C:\Users\LH\.cache\huggingface\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\14dd1130311655b43c3ce41dd505f70f6ca89845"
DATA_PATH = "data/dataset.json"  # 数据文件，格式为 JSON，包含 "data" 字段
OUTPUT_DIR = "lora_output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================ 量化配置（4-bit） ==================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
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
# 添加特殊标记，确保这些标记不被拆分
special_tokens_dict = {"additional_special_tokens": ["<???>", "<不知是谁的声音>", "<乃音子>", "<亚托莉>", "<亚托莉的声音>", "<凛凛花>", "<凛凛花的声音>", "<凯瑟琳>", "<凯瑟琳的声音>", "<初中部男生>", "<大家>", "<女性的声音>", "<妻>", "<孩子Ａ>", "<孩子Ｂ>", "<孩子Ｃ>", "<安田>", "<富田>", "<小龙>", "<少女>", "<居民 A>", "<居民 B>", "<广播>", "<废品店老板>", "<心>", "<快递小哥>", "<摊主>", "<早间广播>", "<机器人少女>", "<水菜萌>", "<水菜萌的声音>", "<水菜萌的妈妈>", "<没见过的男子>", "<洋子>", "<猫？>", "<用户>", "<用户・水菜萌>", "<用户的声音>", "<电视>", "<糖果店的大妈>", "<美代>", "<肉店老板娘>", "<融>", "<诗菜>", "<陌生人>", "<陌生人的声音>", "<高中部的男子>", "<龙司>", "<龙司的声音>", "<ＤＪ>","<context>","<|assistant|>", "<|user|>", "<|system|>"]}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
if num_added_tokens > 0:
    model.resize_token_embeddings(len(tokenizer))
# 若模型未设置 pad_token，则使用 eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ================ 附加 LoRA Adapter ==================
# 根据需要调整 r, lora_alpha, lora_dropout 以及 task_type
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
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
        prompt = "".join(item["<用户>"]).strip()
        response = "".join(item["<亚托莉>"]).strip()
        processed.append({
            "context" : context,
            "prompt": prompt,
            "response": response
        })

    return Dataset.from_list(processed)

raw_dataset = load_data(DATA_PATH)

# ================ 自定义 Tokenize 函数 ==================
def shorten_user_section(text: str, keep_ratio: float = 1) -> str:
    """
    将文本中所有的</user/>部分裁剪，只保留后 keep_ratio 比例的行。
    例如，keep_ratio=0.5 表示保留每个 </user/> 块中后 50% 的行，
    这样可以保留紧跟着亚托莉回答的对话上下文。


    参数:
        text: 完整的原始文本。
        keep_ratio: 保留的行比例，默认为 0.5（即一半）。

    返回:
        裁剪后的文本。
    """
    # 以 <|user|> 分割文本，第一部分通常不是 user 对话内容
    parts = text.split("<|user|>")
    if len(parts) == 1:
        return text  # 如果找不到 <|user|> 则直接返回原文本





    new_text = parts[0]
    # 对每个 <|user|> 部分处理：只保留每个部分的后 keep_ratio 部分
    for user_block in parts[1:]:
        lines = user_block.splitlines()
        # 至少保留一行
        num_keep = max(1, int(len(lines) * keep_ratio))
        # 保留后面的 num_keep 行
        shortened = "\n".join(lines[-num_keep:])
        new_text += "<|user|>" + shortened
    return new_text



def tokenize_example(example):
    # 构造训练文本：包含 system、user、assistant 三个部分
    full_text = (
        "<|system|>\n你是一个名为亚托莉的少女，请用角色身份回答。\n以下是上下文背景：\n"+ example["context"]+"\n"
        "<|user|>\n" + example["prompt"].strip() + "\n"
        "<|assistant|>\n" + example["response"].strip()


    )
    # 对 <用户> 部分裁剪一半以节省显存并保留紧跟 Assistant 的上下文
    # full_text = shorten_user_section(full_text, keep_ratio=0.5)
    
    tokenized = tokenizer(full_text, truncation=False)
    decoded = tokenizer.decode(tokenized["input_ids"])
    print("Decoded full text:\n", decoded)
    
    assistant_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    input_ids = tokenized["input_ids"]
    if assistant_id in input_ids:
        assistant_index = input_ids.index(assistant_id)


        # 保证 assistant 标记后至少有一个 token 用于计算 loss
        if assistant_index + 1 >= len(input_ids):
            print("Warning: assistant 标记后无有效 token，使用完整 input_ids")
            tokenized["labels"] = input_ids.copy()
        else:
            labels = input_ids.copy()
            # mask 掉 assistant 标记之前部分
            for i in range(assistant_index + 1):
                labels[i] = -100
            tokenized["labels"] = labels
    else:
        tokenized["labels"] = input_ids.copy()
    return tokenized




# 对整个数据集应用 tokenize
tokenized_dataset = raw_dataset.map(tokenize_example, remove_columns=raw_dataset.column_names)

# 打印一个样本，检查 token 化结果是否合理
print("示例 tokenized 数据：", tokenized_dataset[0])

# ================ 自定义 Collator ==================
def custom_data_collator(features):
    """
    自定义 collator：对 input_ids、attention_mask、labels 手动 padding 到 batch 内最长序列长度。
    labels 中用于 loss 的 -100 也会被补充到合适位置，保证训练数据形状一致。
    """
    # 分别提取各个字段
    input_ids = [f['input_ids'] for f in features]
    attention_masks = [f['attention_mask'] for f in features]
    labels = [f['labels'] for f in features]

    # 获取本 batch 中最长的长度
    max_length = max(len(ids) for ids in input_ids)

    # 对 input_ids 进行 padding，pad_token_id 填充；对 attention_mask 填充 0；对 labels 填充 -100
    padded_input_ids = [ids + [tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids]
    padded_attention_masks = [mask + [0] * (max_length - len(mask)) for mask in attention_masks]
    padded_labels = [lab + [-100] * (max_length - len(lab)) for lab in labels]

    batch = {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long)
    }
    return batch

# ================ 训练参数 ==================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    # 先使用一个更低的学习率，排查数值稳定性问题
    learning_rate=1e-5,
    logging_steps=1,
    logging_first_step=True,
    logging_strategy="steps",
    # 为排查问题暂时关闭混合精度及 8bit optimizer
    fp16=False,                       
    optim="adamw",                    
    save_strategy="steps",
    save_steps=200,
    report_to="none",
    disable_tqdm=False,
    max_grad_norm=1.0,
)

# 如果需要自定义日志回调，可保留或修改已有 TrainProgressCallback
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

# ================ 开始训练 ==================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=custom_data_collator,
    callbacks=[TrainProgressCallback()]
)

trainer.train()

# 保存 LoRA 适配器（或整个模型，根据需求）
model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))