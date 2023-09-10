import logging
from typing import Dict, List, Union

from gpt4all import GPT4All

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


def predict(
    model: GPT4All,
    model_name: str,
    input_reviews: Union[List[str], str],
    config: dict,
    is_single_input: bool,
) -> Union[List[Dict[str, Union[str, str]]], Dict[str, Union[str, str]]]:
    if is_single_input:
        prompt = config["prompts"][model_name]["instruction"].replace(
            "{review}", input_reviews
        )
        output = model.generate(
            prompt, max_tokens=config["generate_parameters"]["max_tokens"]
        ).strip()
        processed_output = post_process_output(output)
        return {input_reviews: processed_output}
    else:
        results = {}
        for review in input_reviews:
            prompt = config["prompts"][model_name]["instruction"].replace(
                "{review}", review
            )
            output = model.generate(
                prompt, max_tokens=config["generate_parameters"]["max_tokens"]
            ).strip()
            processed_output = post_process_output(output)
            results[review] = processed_output
            # logging.info(f"Review: {Fore.GREEN}{review}{Style.RESET_ALL}")

        return results


def post_process_output(output: str) -> Dict[str, Union[str, str]]:
    if not isinstance(output, str) or not output:
        raise ValueError("Invalid output")

    if not output.endswith("}"):
        output += "}"

    try:
        processed_output = eval(output)
    except Exception as e:
        logging.error(f"Failed to evaluate output: {e}")
        raise

    processed_output = eval(output)
    return {
        "sentiment": processed_output.get("sentiment", "Unknown"),
        "response": processed_output.get("response", "Unknown"),
    }
