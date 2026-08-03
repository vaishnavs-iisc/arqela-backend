# Google Cloud Run Deployment Guide — Arqela Backend

This guide outlines how to deploy `arqela-backend` to Google Cloud Run with **zero cold starts**, low latency, and automatic scaling.

---

## Prerequisites

1. Install Google Cloud SDK (`gcloud`):
   ```bash
   brew install --cask google-cloud-sdk
   gcloud init
   ```
2. Set your default GCP Project ID:
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

---

## 1-Step Terminal Command to Deploy

Run this command inside the `arqela-backend` project root directory:

```bash
gcloud run deploy arqela-backend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --min-instances=1 \
  --max-instances=10 \
  --cpu=2 \
  --memory=2Gi \
  --set-env-vars="SUPABASE_URL=https://your-project.supabase.co,DB_DSN=postgresql://user:password@host:6543/postgres?sslmode=require,GROQ_API_KEY=gsk_...,GEMINI_API_KEY=...,COHERE_API_KEY=..."
```

---

## Configuration Breakdown

* `--min-instances=1`: Keeps **1 warm container running 24/7**, eliminating cold start delays completely (<100ms response time).
* `--cpu=2 --memory=2Gi`: Ensures adequate CPU and RAM for fast LiteLLM streaming, embeddings, and multi-agent graph execution.
* `--allow-unauthenticated`: Permits your web frontend (`arqela-web`) to call the API endpoints.

---

## Health Check Verification

Cloud Run will probe `GET /health`. Once deployed, verify your service status:

```bash
curl https://YOUR-CLOUD-RUN-URL.a.run.app/health
# Output: {"status":"ok","service":"arqela-backend"}
```

---

## Connecting the Frontend (`arqela-web`)

Set the Cloud Run URL in your web app environment (`.env.local` or Vercel environment variables):

```env
NEXT_PUBLIC_API_URL=https://YOUR-CLOUD-RUN-URL.a.run.app
```
