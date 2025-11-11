#!/bin/bash

docker rm ollama webui --force

docker run -d \
  --network host \
  -v ${HOME}/ollama-models:/root/.ollama \
  --name ollama \
  ollama/ollama:latest


docker run -d \
  --network host \
  -e OLLAMA_BASE_URL=http://localhost:11434 \
  -e WEBUI_SECRET_KEY="YOUR_SECURE_SECRET_KEY" \
  --name webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main