import os
import json
import random

# 根目录路径
root_folder = '/home/Data/.bin/.config/dz2/work/PJ/NLP_FInal_PJ/MATH/train'

# 输出文件路径
output_file = '/home/Data/.bin/.config/dz2/work/PJ/NLP_FInal_PJ/MATH/train/merged_dataset.json'

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
            
            # 如果问题超过 1000 条，随机选取 1000 条
            if len(data) > 1000:
                data = random.sample(data, 1000)

            conversation = {
                "conversations": [
                    {
                        "from": "human",
                        "value": data["problem"]  
                    },
                    {
                        "from": "gpt",
                        "value": data["solution"]  
                    }
                ]
            }
            final_dataset.append(conversation)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=4)

print(f"数据已成功合并并保存到 {output_file}")
