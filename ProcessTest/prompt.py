def process_test_prompt(question, model_answer, gt_answer):
    prompt=f'''
    Here is the math problem:
    {question}
    
    I trained a model to solve this problem. The model's answer is:
    {model_answer}
    
    The correct answer is:
    {gt_answer}
    
    Please evaluate the model's inference process. You shouldn't only focus on the final answer, but also the process of solving the problem.
    Then you should provide a score for the model's inference process. The score should be between 0 and 100.
    100 means the model's inference process and result are perfect, 0 means the model's inference process and result are totally wrong. 
    The specific score should be based on the correctness rate of the inference process and the final result.
    
    Finally, provide a score in a bracket, like this: [90]
    
    '''