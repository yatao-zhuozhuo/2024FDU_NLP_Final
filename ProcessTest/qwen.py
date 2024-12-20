import os
from openai import OpenAI
def qwen(prompt):
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        # api_key=os.getenv("DASHSCOPE_API_KEY"), 
        api_key="",
        
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-plus", # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': 'You are a math problem verifier.'},
            {'role': 'user', 'content': prompt}],
        )
    
    
    return completion.choices[0].message.content
    
if __name__ == "__main__":
    print(qwen("What is 2+2?"))
    
    