import json
import re
import time
from prompt import process_test_prompt
from qwen import qwen  

def test_pipeline(predict_json, output_json):
    # load a json file
    with open(predict_json, 'r', encoding="utf-8") as f:
        # content = json.load(f)
        content = [json.loads(line) for line in f]
    score_list = []
    timing_list = []

    total_time = 0  # Initialize total time
    count = 0

    for item in content:
        question = item['prompt']
        standard_answer = item['label']
        predicted_response = item["predict"]

        specific_prompt = process_test_prompt(question, predicted_response, standard_answer)
        
        start_time = time.time()  # Start timing
        response = qwen(specific_prompt)
        end_time = time.time()    # End timing
        
        execution_time = end_time - start_time
        timing_list.append(execution_time)
        total_time += execution_time  # Update total time
        
        
        print(f"Current total execution time: {total_time:.2f} seconds")  # Output running total time

        try:
            # extract the score from the response in brackets
            score = re.search(r'\[(.*?)\]', response).group(1)
            # turn the score into a float
            score = float(score)
            score_list.append(score)

            result = {
                "question": question,
                "score": score,
                "execution_time": execution_time
            }

            with open(output_json, 'a', encoding='utf-8') as out_file:
                out_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                count += 1
        except Exception as e:
            print(f"Error: {e}")
            print(f"Response: {response}")

        if count == 200:
            break

    return score_list, timing_list

if __name__ == "__main__":
    predict_json = "generated_predictions_oral_model_MATH.jsonl"
    output_json = "scores_oral_MATH.jsonl"
    score_list, timing_list = test_pipeline(predict_json, output_json)

    # calculate the average score
    if score_list:
        avg_score = sum(score_list) / len(score_list)
        print(f"Average score: {avg_score}")

    # print timing statistics
    if timing_list:
        total_time = sum(timing_list)
        avg_time = total_time / len(timing_list)
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Average execution time per response: {avg_time:.2f} seconds")

        
        
    