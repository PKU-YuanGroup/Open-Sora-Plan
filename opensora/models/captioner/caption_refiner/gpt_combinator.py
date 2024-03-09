import openai
import ast

def caption_qa(caption_list, api_key, api_base):
    openai.api_key = api_key
    openai.api_base = api_base

    question = "What is the color of a red apple"
    answer = "red"
    pred = "green"
    try:
        # Compute the correctness score
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # model="gpt-4",
            # model="gpt-4-vision-compatible",
            messages=[
                {
                    "role": "system",
                    "content":
                        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                        "------"
                        "##INSTRUCTIONS: "
                        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                        "- Consider synonyms or paraphrases as valid matches.\n"
                        "- Evaluate the correctness of the prediction compared to the answer."
                },
                {
                    "role": "user",
                    "content":
                        "Please evaluate the following video-based question-answer pair:\n\n"
                        f"Question: {question}\n"
                        f"Correct Answer: {answer}\n"
                        f"Predicted Answer: {pred}\n\n"
                        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                        "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                }
            ]
        )
        # Convert response to a Python dictionary.
        response_message = completion["choices"][0]["message"]["content"]
        response_dict = ast.literal_eval(response_message)
        print(response_dict)

    except Exception as e:
        print(f"Error processing file : {e}")


def caption_summary(long_caption, api_key, api_base):
    """
    apply GPT3-Turbo as the combination for original caption and the prompted captions for a video
    """
    openai.api_key = api_key
    openai.api_base = api_base

    try:
        # Compute the correctness score
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content":
                        "You are an intelligent chatbot designed for summarizing from a long sentence. "
                },
                {
                    "role": "user",
                    "content":
                        "Please summarize the following sentences. Make it shorter than 70 words."
                        f"the long sentence: {long_caption}\n"
                        "Provide your summarization with less than 70 words. "
                        "DO NOT PROVIDE ANY OTHER TEXT OR EXPLANATION. Only provide the summary sentence. "
                }
            ]
        )
        # "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
        # "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        # "For example, your response should look like this: {'summary': 'your summary sentence'}."

        # Convert response to a Python dictionary.
        response_message = completion["choices"][0]["message"]["content"]
        response_dict = ast.literal_eval(response_message)

    except Exception as e:
        print(f"Error processing file : {e}")
    
    return response_dict

if __name__ == "__main__":
    caption_summary()