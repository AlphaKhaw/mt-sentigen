import logging
import os
import sys
from typing import Any, List

import pandas as pd
import uvicorn
import yaml
from colorama import Fore, Style, init
from llama_cpp import Llama
from pydantic import BaseModel

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")

from inference.llm_inference import predict

init(autoreset=True)

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


def custom_log(msg: str, color: str):
    """
    Custom logging function with Colorama.

    Args:
        msg (str): Input logging message.
        color (str): Input color of text.
    """
    logging.info(color + msg + Style.RESET_ALL)


def config() -> dict:
    """
    Load the configuration from a YAML file.

    Returns:
        DictConfig: The configuration as an OmegaConf DictConfig object.
    """
    with open("conf/base/pipelines.yaml", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
            return cfg["inference"]
        except Exception as e:
            logging.info(
                f"An error occurred while loading the configuration: {e}"
            )
            raise e


app = FastAPI()


class PredictOneRequest(BaseModel):
    text: str


class PredictMultipleRequest(BaseModel):
    texts: List[str]


@app.on_event("startup")
async def load_model() -> None:
    """
    Load all the models into memory during application startup.
    """
    cfg = config()
    logging.info(cfg["llm_model_weights_path"]["Llama-2-7B Chat"])
    global MODEL
    MODEL = Llama(
        cfg["llm_model_weights_path"]["Llama-2-7B Chat"],
        verbose=cfg["model_parameters"]["verbose"],
        n_ctx=cfg["model_parameters"]["n_ctx"],
    )
    if MODEL is None:
        raise HTTPException(status_code=404, detail="Model weights not found")


@app.get("/")
async def welcome() -> dict:
    """
    Root API endpoint to check the health of the service.

    Returns:
        dict: A dictionary containing a welcome message.
    """
    return {"messages": "Welcome to MT-SentiGen!"}


@app.post("/predict_one/")
async def predict_one(request: PredictOneRequest) -> Any:
    """
    Make a prediction based on a single text input.

    Parameters:
        request (PredictOneRequest): The input JSON payload.

    Returns:
        Any: The predicted output, encoded as a JSON-compatible object.
    """
    text = request.text

    if not text:
        raise HTTPException(status_code=400, detail="Empty text input")

    cfg = config()
    prediction = predict(MODEL, [text], cfg)
    for review in prediction:
        custom_log(f"\nReview: {review}", Fore.BLUE)
        custom_log(f"Sentiment: {prediction[review]['sentiment']}", Fore.GREEN)
        custom_log(
            f"Generated Response: {prediction[review]['response']}", Fore.GREEN
        )
        custom_log(
            f"Improvements: {prediction[review]['improvements']}",
            Fore.GREEN,
        )
        custom_log(
            f"Criticisms: {prediction[review]['criticisms']}\n",
            Fore.GREEN,
        )
    return jsonable_encoder(prediction)


@app.post("/predict_multiple/")
async def predict_multiple(request: PredictMultipleRequest) -> Any:
    """
    Make predictions based on multiple text inputs.

    Parameters:
        request (PredictMultipleRequest): The input JSON payload.

    Returns:
        Any: The predicted outputs, encoded as a JSON-compatible object.
    """
    texts = request.texts

    if not texts or any(not t for t in texts):
        raise HTTPException(
            status_code=400, detail="Empty or invalid text inputs"
        )

    cfg = config()
    predictions = predict(MODEL, texts, cfg)
    for review in predictions:
        custom_log(f"\nReview: {review}", Fore.BLUE)
        custom_log(f"Sentiment: {predictions[review]['sentiment']}", Fore.GREEN)
        custom_log(
            f"Generated Response: {predictions[review]['response']}", Fore.GREEN
        )
        custom_log(
            f"Improvements: {predictions[review]['improvements']}",
            Fore.GREEN,
        )
        custom_log(
            f"Criticisms: {predictions[review]['criticisms']}\n",
            Fore.GREEN,
        )
    return jsonable_encoder(predictions)


@app.post("/predict_csv/")
async def predict_csv(file: UploadFile = File(...)) -> FileResponse:
    """
    Make predictions based on a CSV file containing multiple text inputs.

    Parameters:
        file (UploadFile): The uploaded CSV file.
        model_name (str): The name of the model to be used for prediction.

    Returns:
        FileResponse: A CSV file containing the predictions.
    """
    df = pd.read_csv(file.file)
    if df["text"].isnull().any():
        raise HTTPException(
            status_code=400, detail="CSV contains empty text fields"
        )

    cfg = config()
    predictions_dict = predict(MODEL, df["text"].tolist(), cfg)
    for review in predictions_dict:
        custom_log(f"\nReview: {review}", Fore.BLUE)
        custom_log(
            f"Sentiment: {predictions_dict[review]['sentiment']}", Fore.GREEN
        )
        custom_log(
            f"Generated Response: {predictions_dict[review]['response']}",
            Fore.GREEN,
        )
        custom_log(
            f"Improvements: {predictions_dict[review]['improvements']}",
            Fore.GREEN,
        )
        custom_log(
            f"Criticisms: {predictions_dict[review]['criticisms']}\n",
            Fore.GREEN,
        )
    return jsonable_encoder(predictions_dict)


if __name__ == "__main__":
    uvicorn.run(
        "src.fastapi.llm_api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
