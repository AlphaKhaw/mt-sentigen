import os
import sys
from typing import List, Tuple

import torch
from transformers import AutoTokenizer

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")

from model.multitask_model import MultitaskModelWithT5

SENTIMENT_MAPPING = {0: "negative", 1: "neutral", 2: "positive"}


def predict_one(
    model: MultitaskModelWithT5, tokenizer: AutoTokenizer, input_text: str
) -> Tuple[str]:
    """
    Given a pretrained model and tokenizer, predict the sentiment class and
    generate text for a single input text.

    Args:
        model (MultitaskModelWithT5): The pretrained model for sentiment
            classification and response generation.
        tokenizer (AutoTokenizer): The tokenizer for text processing.
        input_text (str): The input text for which sentiment class and
            generated text is required.

    Returns:
        Tuple[str]: A tuple containing the predicted sentiment class and the
            generated text.
    """
    # Prepare input
    input_encoding = tokenizer(input_text, return_tensors="pt")
    input_ids = input_encoding.input_ids
    attention_mask = input_encoding.attention_mask
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]])

    # Get model output
    encoder_output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
    )

    # Get sentiment class
    _, predicted_class = torch.max(encoder_output.logits, dim=1)
    print(predicted_class.argmax().item())
    sentiment_class = SENTIMENT_MAPPING[predicted_class.argmax().item()]

    # Generate text
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=50,
            repetition_penalty=2.0,
            no_repeat_ngram_size=4,
            num_beams=2,
            temperature=0.5,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )

    generated_text = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True
    )

    return sentiment_class, generated_text


def predict(
    model: MultitaskModelWithT5,
    tokenizer: AutoTokenizer,
    input_texts: List[str],
) -> Tuple[list, list]:
    """
    Given a pretrained model and tokenizer, predict the sentiment class and
    generate text for multiple input texts.

    Args:
        model (MultitaskModelWithT5): The pretrained model for sentiment
            classification and text generation.
        tokenizer (AutoTokenizer): The tokenizer for text processing.
        input_texts (List[str]): A list of input texts for which sentiment
            classes and generated texts are required.

    Returns:
        Tuple[list, list]]: A list of tuples, each containing the predicted
            sentiment class and generated text for each input text.
    """
    sentiment_classes, generated_texts = [], []

    # Iterate through each input text in the list
    for input_text in input_texts:
        # Prepare input
        input_encoding = tokenizer(input_text, return_tensors="pt")
        input_ids = input_encoding.input_ids
        attention_mask = input_encoding.attention_mask
        decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]])

        # Get model output
        encoder_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )

        # Get sentiment class
        _, predicted_class = torch.max(encoder_output.logits, dim=1)
        sentiment_class = SENTIMENT_MAPPING[predicted_class.argmax().item()]

        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                max_length=50,
                repetition_penalty=2.0,
                no_repeat_ngram_size=4,
                num_beams=2,
                temperature=0.5,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,
            )

        generated_text = tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )

        sentiment_classes.append(sentiment_class)
        generated_texts.append(generated_text)

    return sentiment_classes, generated_texts
