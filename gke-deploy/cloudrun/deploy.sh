export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1" # Must be the same region as the VPC/Subnet
export DEFAULT_SA_EMAIL="${PROJECT_ID}-compute@developer.gserviceaccount.com"
export OLLAMA_SERVICE_NAME="ollama-server-l4"
export WEBUI_SERVICE_NAME="open-webui-external"
export VPC_NETWORK="default"  # Replace with your VPC network name
export VPC_SUBNET="default"   # Replace with your VPC subnet name
export WEBUI_SECRET_KEY="YOUR_SECURE_SECRET_KEY" # CHANGE THIS!

# Deploy the Ollama service using the default service account
gcloud run deploy $OLLAMA_SERVICE_NAME \
  --image ollama/ollama:latest \
  --region $REGION \
  --platform managed \
  --no-allow-unauthenticated \
  --ingress internal \
  --network $VPC_NETWORK \
  --subnet $VPC_SUBNET \
  --vpc-egress all-traffic \
  --memory 16Gi \
  --cpu 4 \
  --min-instances 0 \
  --max-instances 1 \
  --command /bin/ollama \
  --args "serve" \
  --cpu-boost \
  --gpu-type nvidia-l4  \
  --gpu 1\
  --no-gpu-zonal-redundancy \
  --set-env-vars OLLAMA_HOST=0.0.0.0:11434 \
  --port 11434

# Get the private URL
export OLLAMA_URL=$(gcloud run services describe $OLLAMA_SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --format 'value(uri)')

# # Grant IAM permission
# gcloud run services add-iam-policy-binding $OLLAMA_SERVICE_NAME \
#   --region $REGION \
#   --member "serviceAccount:$DEFAULT_SA_EMAIL" \
#   --role "roles/run.invoker"


# Deploy the WebUI service using the default service account
# zicongmei/open-webui:main is a mirror of ghcr.io/open-webui/open-webui:main
gcloud run deploy $WEBUI_SERVICE_NAME \
  --image zicongmei/open-webui:main  \
  --region $REGION \
  --platform managed \
  --no-allow-unauthenticated \
  --ingress internal \
  --network $VPC_NETWORK \
  --subnet $VPC_SUBNET \
  --vpc-egress all-traffic \
  --memory 1Gi \
  --cpu 1 \
  --port 8080
  --set-env-vars OLLAMA_BASE_URL=$OLLAMA_URL,WEBUI_SECRET_KEY=$WEBUI_SECRET_KEY