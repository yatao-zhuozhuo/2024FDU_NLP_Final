import os
import json
import random

# 根目录路径
root_folder = '/home/Data/.bin/.config/dz2/work/PJ/NLP_Final_PJ/MATH/test'

# 输出文件路径
output_file = '/home/Data/.bin/.config/dz2/work/PJ/NLP_Final_PJ/MATH/data_for_server/test.json'

final_dataset = []

# 遍历根目录下的所有子文件夹
for sub_dir in os.listdir(root_folder):
    sub_dir_path = os.path.join(root_folder, sub_dir)
    
    if os.path.isdir(sub_dir_path):
        json_files = [f for f in os.listdir(sub_dir_path) if f.endswith('.json')]
        
        # 从每个 JSON 文件中读取数据
        for json_file in json_files:
            file_path = os.path.join(sub_dir_path, json_file)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 如果问题超过 500 条，随机选取 500 条
            if len(data) > 500:
                data = random.sample(data, 500)

            # 创建对话数据
            conversation = {
                "conversations": [
                    {
                        "from": "human",
                        "value": data["problem"]  # "problem"作为human的提问
                    },
                    {
                        "from": "gpt",                            
                        "value": data["solution"]  # "solution"作为gpt的回答
                    }
                ],
                "system": "You are a helpful assistant skilled in solving mathematical problems. Provide step-by-step explanations for each solution and ensure clarity in reasoning.",
                "tools": {
                    "calculator": {
                        "description": "A basic calculator tool to perform arithmetic operations.",                            
                        "enabled": True,
                        "functions": ["add", "subtract", "multiply", "divide"]
                    },
                    "algebra_solver": {
                        "description": "An algebra solver tool to simplify equations and solve for unknowns.",
                        "enabled": True,
                        "functions": ["simplify", "solve"]
                    }
                }
            }
            final_dataset.append(conversation)


# 将处理后的数据保存到输出文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=4)

print(f"数据已成功合并并保存到 {output_file}")

