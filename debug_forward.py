import os
os.environ["FLASH_ATTN_DISABLE"] = "1"  # 禁用 flash attention

import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

# ================ 配置参数 ==================
model_path = r"C:\Users\LH\.cache\huggingface\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\14dd1130311655b43c3ce41dd505f70f6ca89845"
dataset_path = "data/dataset.json"  # 请根据实际情况修改数据集路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================ 加载 Tokenizer ==================
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    use_fast=True
)

# 添加特殊 token，根据需要设置（必须添加，否则可能出现解码错误）
tokenizer.add_special_tokens({
    'bos_token': '<|beginoftext|>',
    'eos_token': '<|endoftext|>',
    'pad_token': '<|pad|>'
})

# ================ 加载模型 ==================
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager"  # 使用传统的 attention 实现方式
)

# ================ 配置 LoRA ==================
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False  # 训练时必须让 LoRA 层可训练
)
model = get_peft_model(model, peft_config)
model = model.to(DEVICE)

# ================= 数据加载与预处理 ==================
def load_data(file_path):
    """
    加载数据集文件（JSON格式），并预处理
    数据格式应为一个列表，其中每个条目包含 "context"、"input" 和 "output" 字段
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    processed = []
    for item in data:
        # 构造完整的 prompt：将 context 与 input 拼接
        context = "".join(item["context"])
        input_text = item["input"].strip()
        prompt = f"{context}{input_text}"
        # 构造 response
        response = "".join(item["output"]).strip()
        processed.append({
            "prompt": prompt,
            "response": response
        })
    return Dataset.from_list(processed)

def format_function(sample):
    """
    根据 chat 模板格式化输入
    此处假定tokenizer有apply_chat_template方法，可将对话内容转换为最终的文本输入
    """
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": "你是一个名为亚托莉的少女，请用角色身份回答。"},
        {"role": "user", "content": sample["prompt"]},
        {"role": "assistant", "content": sample["response"]}
    ], tokenize=False)
    return {"text": text}

def tokenize_func(samples):
    """
    对格式化后的文本进行分词，保证输入长度不超过 512，必要时进行截断和填充
    """
    return tokenizer(
        samples["text"],
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

# 加载并预处理数据
dataset = load_data(dataset_path)
formatted_dataset = dataset.map(format_function, remove_columns=["prompt", "response"])
tokenized_dataset = formatted_dataset.map(
    tokenize_func,
    batched=True,
    batch_size=32,
    remove_columns=["text"]
)

# ================ 构造 Data Collator ==================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 因为是因果语言模型，不使用 Masked LM
)

# ================= 执行 Debug 前向传播 ==================
# 选取前两个样本进行调试
sample = tokenized_dataset[:2]
batch = data_collator(sample)
# 将 batch 内的张量移到 DEVICE 上
batch = {k: v.to(DEVICE) for k, v in batch.items()}

# 切换到评估模式，关闭 dropout 等
model.eval()
with torch.no_grad():
    outputs = model(**batch)

print("Debug forward loss:", outputs.loss)
