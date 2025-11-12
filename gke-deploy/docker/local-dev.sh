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


# open http://localhost:8080/

# curl http://localhost:11434/api/generate -d '{
#   "model": "gemma3n:e2b",
#   "prompt": "Why is the sky blue? reply in 50 words",
#   "stream": false
# }' | jq -r '.response'



## create nginx

CERT_DIR="${HOME}/cert"
NGINX_CONF_PATH="/tmp/nginx.conf"

echo "Ensuring certificate directory exists: ${CERT_DIR}"
mkdir -p "${CERT_DIR}"

echo "Generating self-signed SSL certificate (valid for 365 days)..."
# The CN (Common Name) should ideally match the hostname, 
# but for local testing 'localhost' is used.
# NOTE: Browsers will show a warning because this is a self-signed certificate.
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout "${CERT_DIR}/nginx.key" \
    -out "${CERT_DIR}/nginx.crt" \
    -subj "/C=US/ST=State/L=City/O=Organization/OU=IT/CN=localhost" 2>/dev/null
echo "Certificate generated in ${CERT_DIR}"


# 3. Create NGINX Configuration File (nginx.conf)
echo "Creating nginx.conf in ${NGINX_CONF_PATH}..."
# Using a 'here document' to write the configuration file directly
# We use 'EOF' to prevent shell variable expansion inside the NGINX config
cat << 'EOF' > "${NGINX_CONF_PATH}"
# Server block for HTTP (Port 80)
# This handles the forwarding/redirecting of all HTTP requests to HTTPS
server {
    listen 80;
    server_name localhost;

    # Forward all HTTP traffic to HTTPS
    return 301 https://$host$request_uri;
}

# Server block for HTTPS (Port 443)
server {
    listen 443 ssl;
    server_name localhost;

    # SSL Configuration
    ssl_certificate /etc/nginx/certs/nginx.crt;
    ssl_certificate_key /etc/nginx/certs/nginx.key;

    # Recommended SSL settings (modern and secure)
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;

    # Reverse Proxy to the Open WebUI container
    # When using --network host, containers access each other via localhost and their ports.
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (crucial for LLM streaming and chat apps)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF
echo "nginx.conf created."



docker rm nginx-proxy --force

echo "Starting NGINX Proxy container on host network on ports 80 and 443..."
# PWD is the current directory where nginx.conf must be located
docker run -d \
  --network host \
  -v "${CERT_DIR}":/etc/nginx/certs:ro \
  -v "${NGINX_CONF_PATH}":/etc/nginx/conf.d/default.conf:ro \
  --name nginx-proxy \
  --restart always \
  nginx:latest

