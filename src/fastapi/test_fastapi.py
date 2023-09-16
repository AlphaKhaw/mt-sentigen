import json
import logging
from io import StringIO

import pandas as pd
import requests

# Initialize logging
logging.basicConfig(level=logging.INFO)

# BASE_URL = "http://127.0.0.1:8000"
# BASE_URL = "http://0.0.0.0:8000"
BASE_URL = "http://3.1.6.147:8000"


def test_welcome():
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert response.json() == {"messages": "Welcome to MT-SentiGen!"}
    logging.info("test_welcome() was successful.")


def test_predict_one():
    payload = {"text": "Your service was amazing, keep up the good work!"}
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        f"{BASE_URL}/predict_one/", data=json.dumps(payload), headers=headers
    )
    print(response.json())

    assert response.status_code == 200
    logging.info("test_predict_one() was successful.")


def test_predict_multiple():
    payload = {
        "texts": [
            "Your service was great!",
            "I am not happy with your service.",
            "It was okay.",
        ]
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        f"{BASE_URL}/predict_multiple/",
        data=json.dumps(payload),
        headers=headers,
    )
    print(response.json())

    assert response.status_code == 200


def test_predict_csv():
    # Create a test DataFrame
    df = pd.DataFrame(
        {"text": ["Amazing service!", "Could be better.", "Meh."]}
    )
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    files = {"file": ("test.csv", csv_buffer, "text/csv")}
    response = requests.post(f"{BASE_URL}/predict_csv/", files=files)
    print(response.json())

    assert response.status_code == 200


if __name__ == "__main__":
    test_welcome()
    test_predict_one()
    # test_predict_multiple()
    # test_predict_csv()
