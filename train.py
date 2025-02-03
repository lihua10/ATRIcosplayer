# train.py
import os
# 设置 CUDA 内存分配相关环境变量，降低内存碎片风险
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import json
from datasets import Dataset
import torch
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
DATA_PATH = "data/dataset.txt"  # 数据文件，需为 JSON 格式，包含 "data" 字段
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
    attn_implementation="eager",  # 使用传统 attention 实现方式
    quantization_config=bnb_config   # 使用4-bit量化配置降低显存占用
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# ================ 数据加载和预处理 ==================
def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)["data"]
    except json.JSONDecodeError as e:
        print(f"JSON格式错误在 {e.lineno} 行 {e.colno} 列")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            start = max(0, e.lineno-3)
            end = min(len(lines), e.lineno+2)
            print("错误上下文：\n" + "".join(lines[start:end]))
        exit(1)

    processed = []
    for item in data:
        context = "".join(item["context"])
        input_text = item["input"].strip()
        prompt = f"{context}{input_text}"
        response = "".join(item["output"]).strip()
        processed.append({
            "prompt": prompt,
            "response": response
        })
    return Dataset.from_list(processed)

dataset = load_data(DATA_PATH)

def format_func(sample):
    text = f"<|system|>\n你是一个名为亚托莉的少女，请用角色身份回答。\n<|user|>\n{sample['prompt']}\n<|assistant|>\n{sample['response']}"
    # 限制长度，确保不超过模型可接受的最大长度
    return {"text": text[:2000]}

formatted_dataset = dataset.map(format_func)

def tokenize_func(samples):
    return tokenizer(
        samples["text"],
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=True
    )

tokenized_dataset = formatted_dataset.map(
    tokenize_func,
    batched=True,
    batch_size=32,  # 内部批次大小，不影响 GPU 占用；最终加载到模型时会按 per_device_train_batch_size 组装
    remove_columns=["text"]
)

# ================ 配置 LoRA ==================
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 如果发现模型部分参数在 meta 设备，则转换到目标设备
if any(param.device.type == "meta" for param in model.parameters()):
    print("发现 meta 参数，正在将模型转换到", DEVICE)
    model = model.to_empty(device=DEVICE)

# 使输入 embeddings 可求梯度，便于 gradient checkpointing 捕获反向传播图
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

# 启用 gradient checkpointing（如果模型支持），进一步降低显存占用
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# ================ 定义训练回调 ==================
class TrainProgressCallback(TrainerCallback):
    def __init__(self):
        self.epoch_bar = None
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"{'Epoch':<7}{'GPU_mem':>10}{'Loss':>12}{'LearningRate':>12}{'Time':>10}")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_bar = tqdm(total=state.max_steps, desc=f'Epoch {state.epoch}', leave=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            current_lr = logs.get('learning_rate', 'N/A')
            mem = f'{torch.cuda.memory_reserved()/1E9 if torch.cuda.is_available() else 0:.1f}GB'
            epoch_info = f"{state.epoch}/{state.num_train_epochs}"
            loss_info = f"{logs.get('loss', 0):.4f}"
            lr_info = f"{current_lr:.2e}" if isinstance(current_lr, float) else current_lr
            time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time))
            print(f"\033[K{epoch_info:<7}{mem:>10}{loss_info:>12}{lr_info:>12}{time_elapsed:>10}", end='\r')

    def on_step_end(self, args, state, control, **kwargs):
        if self.epoch_bar:
            self.epoch_bar.update(1)
            # 防止 log_history 为空的情况
            if state.log_history:
                latest_log = state.log_history[-1]
                loss_value = latest_log.get('loss', 0)
                lr_value = latest_log.get('learning_rate', 0)
            else:
                loss_value = 0
                lr_value = 0
            self.epoch_bar.set_postfix({
                'loss': f"{loss_value:.4f}",
                'lr': f"{lr_value:.2e}"
            })

# ================ 训练参数 ==================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,  # 调低 batch size 降低显存占用
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    logging_first_step=True,  # 第一训练步就记录日志
    fp16=True,
    optim="paged_adamw_8bit",
    save_strategy="steps",
    save_steps=200,
    report_to="none",
    disable_tqdm=False,
)

def data_collator(features):
    return {
        "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
        "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in features]),
        "labels": torch.stack([torch.tensor(f["input_ids"]) for f in features])
    }

# ================ 开始训练 ==================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    callbacks=[TrainProgressCallback()]
)

trainer.train()

# 保存适配器
model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))