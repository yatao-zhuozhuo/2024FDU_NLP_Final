from prompt import process_test_prompt
from qwen import qwen   

def test_pipeline(predict_json):
    # load a json file
    with open(predict_json, 'r') as f:
        content = json.load(f)
    score_list = []
    for item in content:
        question = item['input']
        standard_answer = item['standard_answer']
        predicted_response = item["predicted_response"]
        
        specific_prompt = process_test_prompt(question, predicted_response, standard_answer)
        response = qwen(specific_prompt)
        try:
            # extract the score from the response in brackets
            score = re.search(r'\[(.*?)\]', response).group(1)
            # turn the score into a float
            score = float(score)
            score_list.append(score)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Response: {response}")
    return score_list

if __name__ == "__main__":
    predict_json = "predict.json"
    score_list = test_pipeline(predict_json)
    # calculate the average score
    avg_score = sum(score_list)/len(score_list)
    print(f"Average score: {avg_score}")

        
        
    