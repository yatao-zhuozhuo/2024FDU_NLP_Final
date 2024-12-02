import torch
import json
from transformers import AutoModelForCausalLM, Trainer,AutoTokenizer,DataCollatorForSeq2Seq
from transformers import TrainingArguments
import random

model = AutoModelForCausalLM.from_pretrained('./output/Qwen2.5_instruct_lora_final')

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained('./output/Qwen2.5_instruct_lora_final')

test_dataset_file = '/home/Data/.bin/.config/dz2/work/PJ/NLP_Final_PJ/MATH/data_for_server/test.json'  # 测试集路径

# 加载并处理测试数据
with open(test_dataset_file, 'r', encoding='utf-8') as f:
    test_data = json.load(f)
    
import json

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