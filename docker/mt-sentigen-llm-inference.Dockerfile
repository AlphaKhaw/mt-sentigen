FROM python:3-slim-bullseye

# Set TERM environment variable
ENV TERM xterm-256color

# Install system dependencies
RUN apt-get update && apt-get install -y \
	build-essential \
	libopenblas-dev \
	ninja-build

# RUN python -m pip install --no-cache-dir --upgrade \
# 	cmake \
# 	fastapi \
# 	sse-starlette \
# 	uvicorn \
#     colorama \
#     pyYAML \
#     pydantic \
#     pandas \
#     python-multipart

# Set working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY llm-inference-requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install llama_cpp_python separately with specific flags
RUN CMAKE_ARGS="-DLLAMA_CLBLAST=on" \
	FORCE_CMAKE=1 \
	LLAMA_CLBLAST=1 \
	pip install --no-cache-dir llama_cpp_python --verbose

# Copy only the necessary files
COPY ./models/llama-2-7b-chat.Q3_K_M.gguf /app/models/llama-2-7b-chat.Q3_K_M.gguf
# COPY ./models/llama-2-7b-chat.Q4_K_M.gguf /app/models/llama-2-7b-chat.Q4_K_M.gguf
COPY ./src/fastapi/llm_api.py /app/src/fastapi/llm_api.py
COPY ./src/inference/llm_inference.py /app/src/inference/llm_inference.py
COPY ./conf/base/pipelines.yaml /app/conf/base/pipelines.yaml

# Expose the port
EXPOSE 8000

# Use CMD to run the FastAPI application
CMD ["uvicorn", "src.fastapi.llm_api:app", "--host", "0.0.0.0", "--port", "8000"]
