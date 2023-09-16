import logging
from typing import Dict, List

from colorama import Fore, Style, init
from llama_cpp import Llama

init(autoreset=True)

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


def custom_log(msg: str, color: str) -> None:
    """
    Log a custom message with a specified color.

    Args:
        msg (str): The message to be logged.
        color (str): The color code to be used for the message.

    Returns:
        None
    """
    logging.info(color + msg + Style.RESET_ALL)


def predict(
    model: Llama,
    input_reviews: List[str],
    config: dict,
) -> Dict[str, dict]:
    """
    Generate predictions for a list of reviews using a specified model and
    configuration settings.

    Parameters:
        model (Llama): The Llama model object to be used for generating
            predictions.
        input_reviews (List[str]): A list of review strings for which
            predictions will be generated.
        config (dict): A dictionary containing the configuration settings,
            including 'generate_parameters' and 'prompts'.

    Returns:
        Dict[str, dict]: A dictionary where keys are the review strings and the
            values are the generated predictions.

    Note:
        The function attempts to generate a prediction multiple times
        (up to 'max_attempts') for each review if it fails to evaluate the
        model output.
    """
    max_attempts = config["generate_parameters"]["max_attempts"]
    results = {}
    for review in input_reviews:
        prompt = config["prompts"]["Llama-2-7B Chat"]["instruction"].replace(
            "{review}", review
        )
        for _ in range(max_attempts):
            output = model(
                prompt, max_tokens=config["generate_parameters"]["max_tokens"]
            )
            try:
                evaluated_output = eval(output["choices"][0]["text"])
                results[review] = evaluated_output
                break
            except Exception as e:
                custom_log(f"{e} - Re-attempting LLM Generation", Fore.RED)

    return results
