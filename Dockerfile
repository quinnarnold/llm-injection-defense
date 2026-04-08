FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install uv

WORKDIR /app
COPY pyproject.toml .
RUN uv pip install --system .

COPY src/ src/

EXPOSE 8080

CMD ["uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8080"]
