# FROM nvcr.io/nvidia/cuda:11.3.0-cudnn8-devel-ubuntu18.04
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Setup shell
SHELL ["/bin/bash", "-c"]

# Arguments and Environment Variables
ARG CONDA_ENV_FILE="inference-conda-env.yaml"
ARG HOME_DIR="/app"

# Install system packages and clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR $HOME_DIR

# Install Miniconda
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -u -b -p /miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Conda settings
ENV PATH /miniconda3/bin:$PATH

# Conda environment setup
COPY $CONDA_ENV_FILE .
RUN conda env create -f $CONDA_ENV_FILE

# Copy only the necessary files
COPY ./models /app/models
COPY ./src/fastapi/main.py /app/src/fastapi/main.py
COPY ./src/model /app/src/model

# Expose the port
EXPOSE 8000

# Set the entry point
SHELL ["conda", "run", "-n", "mt-sentigen", "/bin/bash", "-c"]
CMD ["uvicorn", "src.fastapi.main:app", "--host", "0.0.0.0", "--port", "8000"]
