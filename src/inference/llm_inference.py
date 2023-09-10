import logging
from typing import Dict, List, Union

import pandas as pd
from gpt4all import GPT4All


class DummyLLMModel:
    def generate(self, prompt: str, max_tokens: int) -> str:
        return '{"sentiment": "Positive", "response": "Thank you for your kind words!"}'


# Initialize your LLM models here.
LLM_MODELS = {
    "LLM1": {
        "model": DummyLLMModel(),
        "prompt_template": "Your LLM1 Prompt Template here...",
    },
    "LLM2": {
        "model": DummyLLMModel(),
        "prompt_template": "Your LLM2 Prompt Template here...",
    }
    # Add more models if needed
}


def generate_prompt(review: str, template: str) -> str:
    return template.format(review=review)


def post_process_output(output: str) -> Dict[str, Union[str, str]]:
    processed_output = eval(output)
    return {
        "sentiment": processed_output.get("sentiment", "Unknown"),
        "response": processed_output.get("response", "Unknown"),
    }


def predict(
    model_name: str, input_reviews: List[str]
) -> List[Dict[str, Union[str, str]]]:
    model_info = LLM_MODELS.get(model_name)
    if model_info is None:
        return [{"error": f"Model {model_name} not found"}]

    model = model_info["model"]
    prompt_template = model_info["prompt_template"]

    results = []

    for review in input_reviews:
        prompt = generate_prompt(review, prompt_template)
        output = model.generate(prompt, max_tokens=150).strip()
        processed_output = post_process_output(output)
        results.append(processed_output)

    return results
