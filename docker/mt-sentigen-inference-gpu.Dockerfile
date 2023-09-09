# Use the latest Ubuntu base image
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=noninteractive

# Install some basic utilities and Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    python3.10 \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY ./requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy only the necessary files
COPY ./models /app/models
COPY ./src/fastapi/main.py /app/src/fastapi/main.py
COPY ./src/model /app/src/model
COPY ./src/inference/inference.py /app/src/inference/inference.py
COPY ./conf/base/pipelines.yaml /app/conf/base/pipelines.yaml

# Expose the port your app runs on
EXPOSE 8000

# Use CMD to run the FastAPI application
CMD ["uvicorn", "src.fastapi.main:app", "--host", "0.0.0.0", "--port", "8000"]
