import torch
import json
from transformers import AutoModelForCausalLM, Trainer,AutoTokenizer,DataCollatorForSeq2Seq
from transformers import TrainingArguments
import random

# 加载预训练的 tokenizer
tokenizer = AutoTokenizer.from_pretrained('/home/Data/.bin/.config/dz2/work/model/Qwen2.5-0.5B-Instruct', use_fast=False, trust_remote_code=True)
# 加载 Qwen2.5 模型
model = AutoModelForCausalLM.from_pretrained('/home/Data/.bin/.config/dz2/work/model/Qwen2.5-0.5B-Instruct', device_map="auto")

# 数据集路径
train_dataset_file = '/home/Data/.bin/.config/dz2/work/PJ/NLP_Final_PJ/MATH/data_for_server/train.json'
test_dataset_file = '/home/Data/.bin/.config/dz2/work/PJ/NLP_Final_PJ/MATH/data_for_server/test.json'  # 测试集路径

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

# 加载并处理训练数据
with open(train_dataset_file, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

tokenized_train_data = [process_func(example) for example in train_data]

# 加载并处理测试数据
with open(test_dataset_file, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

tokenized_test_data = [process_func(example) for example in test_data]

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
    save_steps=500,  # 每 100 步保存一次
    learning_rate=1e-4,  # 学习率
    save_on_each_node=False,  # 每个节点保存模型
    gradient_checkpointing=True,  # 打开梯度检查点
)

# 使用训练数据和 DataCollator 创建 Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_data,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# 开始训练
#trainer.train()

# 保存模型
trainer.save_model('./output/Qwen2.5_instruct_lora_final')  # 保存最终模型

# 保存tokenizer
tokenizer.save_pretrained('./output/Qwen2.5_instruct_lora_final')


def generate_response(input_text):
    inputs = tokenizer(f"<|im_start|>system\n现在请你完成数学题<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt")
    inputs = inputs.to(model.device)
    
    # 生成回复
    output = model.generate(inputs['input_ids'], max_length=512, num_beams=5, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response

# 用于生成整个测试集的预测结果并保存为 JSON 文件
def generate_all_responses(test_data):
    # 随机选择最多 500 个数据
    sample_data = random.sample(test_data, min(500, len(test_data)))
    
    results = []
    for example in sample_data:
        # 获取输入文本
        input_text = example['conversations'][0]['value']
        standard_answer = example['conversations'][1]['value']
        
        # 生成模型的预测回复
        predicted_response = generate_response(input_text)
        
        # 将输入和预测的回复保存到结果列表中
        results.append({
            "input": input_text,
            "standard_answer": standard_answer,
            "predicted_response": predicted_response
        })
    
    # 将结果保存为 JSON 文件
    output_file = './output/predicted_responses.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Predictions saved to {output_file}")
    return results


# 对整个测试集进行预测并保存结果
predictions = generate_all_responses(test_data)