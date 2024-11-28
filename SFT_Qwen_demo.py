import torch
import json
from transformers import AutoModelForCausalLM, Trainer,AutoTokenizer,DataCollatorForSeq2Seq
from transformers import TrainingArguments

# 加载预训练的 tokenizer
tokenizer = AutoTokenizer.from_pretrained('/home/Data/.bin/.config/dz2/work/model/Qwen2.5-0.5B-Instruct', use_fast=False, trust_remote_code=True)
# 加载 Qwen2.5 模型
model = AutoModelForCausalLM.from_pretrained('/home/Data/.bin/.config/dz2/work/model/Qwen2.5-0.5B-Instruct', device_map="auto")

# 数据集路径
dataset_file = '/home/Data/.bin/.config/dz2/work/PJ/NLP_FInal_PJ/MATH/train/merged_dataset.json'

# 数据预处理函数
def process_func(example): 
    MAX_LENGTH = 512  # 最大长度
    input_ids, attention_mask, labels = [], [], []
    
    # 从 conversations 中提取 human 和 gpt 对话内容
    instruction = example['conversations'][0]['value']  # human 输入
    response = example['conversations'][1]['value']     # 模型 回复

    # 使用 Qwen2.5 所需的输入格式
    instruction_tokenized = tokenizer(f"<|im_start|>system\n现在请你完成数学题<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    response_tokenized = tokenizer(f"{response}", add_special_tokens=False)
    
    input_ids = instruction_tokenized["input_ids"] + response_tokenized["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction_tokenized["attention_mask"] + response_tokenized["attention_mask"] + [1]
    labels = [-100] * len(instruction_tokenized["input_ids"]) + response_tokenized["input_ids"] + [tokenizer.pad_token_id]
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 加载数据集并预处理
with open(dataset_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

tokenized_data = [process_func(example) for example in data]

'''
from peft import LoraConfig, TaskType
from peft import PeftModel,get_peft_model

# 定义 Lora 配置
model.to('cuda')
model.enable_input_require_grads()
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "gate_proj"
    ],
    inference_mode=False,  # 训练模式
    r=4,  # Lora 秩
    lora_alpha=8,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)
# 使用 Lora 配置加载模型
model = get_peft_model(model,config)

'''


# 自定义训练参数
args = TrainingArguments(
    output_dir="./output/Qwen2.5_instruct_lora",  # 输出目录
    per_device_train_batch_size=4,  # 每设备 batch size
    gradient_accumulation_steps=4,  # 梯度累加
    logging_steps=10,  # 每 10 步输出一次日志
    num_train_epochs=3,  # 训练轮数
    save_steps=100,  # 每 100 步保存一次
    learning_rate=1e-4,  # 学习率
    save_on_each_node=True,  # 每个节点保存模型
    gradient_checkpointing=True,  # 打开梯度检查点
)

# 使用训练数据和 DataCollator 创建 Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_data,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# 开始训练
trainer.train()
