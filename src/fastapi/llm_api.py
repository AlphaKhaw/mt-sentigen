import logging
import os
import sys
from typing import Any, List

import pandas as pd
import uvicorn
import yaml
from gpt4all import GPT4All
from pydantic import BaseModel

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")

from inference.llm_inference import predict

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


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
    model_name: str


class PredictMultipleRequest(BaseModel):
    texts: List[str]
    model_name: str


@app.on_event("startup")
async def load_model() -> None:
    """
    Load all the models into memory during application startup.
    """
    cfg = config()
    global MODELS
    MODELS = {
        "Hermes": GPT4All(
            model_name="nous-hermes-13b.ggmlv3.q4_0.bin",
            model_path=cfg["llm_model_weights_path"]["Hermes"],
            allow_download=False,
        ),
        "Llama-2-7B Chat": GPT4All(
            model_name="llama-2-7b-chat.ggmlv3.q4_0.bin",
            model_path=cfg["llm_model_weights_path"]["Llama-2-7B Chat"],
            allow_download=False,
        ),
    }


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
    model_name = request.model_name

    if not text:
        raise HTTPException(status_code=400, detail="Empty text input")

    model = MODELS.get(model_name)
    if model is None:
        raise HTTPException(
            status_code=404, detail=f"Model {model_name} not found"
        )

    cfg = config()
    logging.info("Initialized config")
    prediction = predict(model, model_name, text, cfg, is_single_input=True)
    logging.info(prediction)
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
    model_name = request.model_name

    if not texts or any(not t for t in texts):
        raise HTTPException(
            status_code=400, detail="Empty or invalid text inputs"
        )

    model = MODELS.get(model_name)
    if model is None:
        raise HTTPException(
            status_code=404, detail=f"Model {model_name} not found"
        )

    cfg = config()
    predictions = predict(model, model_name, texts, cfg, is_single_input=False)
    return jsonable_encoder(predictions)


@app.post("/predict_csv/")
async def predict_csv(
    file: UploadFile = File(...), model_name: str = Query(...)
) -> FileResponse:
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

    model = MODELS.get(model_name)
    if model is None:
        raise HTTPException(
            status_code=404, detail=f"Model {model_name} not found"
        )

    cfg = config()
    predictions_dict = predict(
        model, model_name, df["text"].tolist(), cfg, is_single_input=False
    )
    df["predictions"] = df["text"].map(predictions_dict)
    df.to_csv("predicted.csv", index=False)
    return FileResponse("predicted.csv")


if __name__ == "__main__":
    uvicorn.run(
        "src.fastapi.llm_api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
