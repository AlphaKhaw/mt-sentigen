import logging
import os
import sys
from typing import Union

import torch
import uvicorn
import yaml
from omegaconf import DictConfig
from pydantic import BaseModel
from transformers import AutoTokenizer, T5ForConditionalGeneration

from fastapi import FastAPI, HTTPException

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")

from training.inference import predict, predict_one

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


# Initialize Hydra
def config() -> DictConfig:
    """
    Load the configuration from a YAML file.

    Returns:
        DictConfig: The configuration as an OmegaConf DictConfig object.
    """
    with open("conf/base/pipelines.yaml", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
            return cfg
        except Exception as e:
            logging.info(
                f"An error occurred while loading the configuration: {e}"
            )
            raise e


# Initialize FastAPI
app = FastAPI()

# Check for CUDA availability
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class TextItems(BaseModel):
    """
    Pydantic BaseModel for validating input text items.

    Attributes:
        texts (Union[str, list[str]]): The input texts, can be a single string
            or a list of strings.
    """

    texts: Union[str, list[str]]


@app.on_event("startup")
def startup_event():
    """
    Initializes model and tokenizer upon application startup.
    """
    try:
        app.cfg = config()
        model_weights_path = app.cfg["inference"]["model_weights_path"]
        model = app.cfg["dataloader"]["encoder_decoder_model"]
        app.model = T5ForConditionalGeneration.from_pretrained(
            model_weights_path
        ).to(device)
        app.tokenizer = AutoTokenizer.from_pretrained(model)
        logging.info("Start server complete")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def health():
    """
    Root API endpoint to check the health of the service.

    Returns:
        dict: A dictionary containing a welcome message.
    """
    return {"messages": "Hello from FastAPI!"}


@app.post("/predict_one/")
async def api_predict_one(text_items: TextItems):
    """
    API endpoint to make a prediction on a single text input.

    Args:
        text_items (TextItems): The input texts for which predictions are to be
            made.

    Returns:
        dict: The predicted class and generated text.
    """
    if not text_items.texts:
        raise HTTPException(
            status_code=400, detail="Input text cannot be empty."
        )
    input_text = (
        text_items.texts
        if isinstance(text_items.texts, str)
        else text_items.texts[0]
    )
    try:
        predicted_class, generated_text = predict_one(
            app.model, app.tokenizer, input_text
        )
        return {
            "sentiment_class": predicted_class,
            "generated_text": generated_text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/")
async def api_predict_many(text_items: TextItems):
    """
    API endpoint to make predictions on multiple text inputs.

    Args:
        text_items (TextItems): The input texts for which predictions are to be
            made.

    Returns:
        dict: Lists of predicted classes and generated texts.
    """

    if not text_items.texts:
        raise HTTPException(
            status_code=400, detail="Input text cannot be empty."
        )
    try:
        sentiment_classes, generated_texts = predict(
            app.model, app.tokenizer, text_items.texts
        )
        return {
            "sentiment_classes": sentiment_classes,
            "generated_texts": generated_texts,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "src.fastapi.main:app",
        host="127.0.0.1",
        port=8080,
        reload=False,
        log_level="info",
    )
