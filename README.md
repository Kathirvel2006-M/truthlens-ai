# TruthLens AI — Full-stack starter

Responsive website with FastAPI backend and explicit media-analysis error handling.

## Current capabilities
- Image: calls a configurable Hugging Face inference model.
- Video: samples up to 8 frames and analyzes them using the configured image model. This is only frame-based screening, not a dedicated temporal video detector.
- Audio: requires `AUDIO_ENDPOINT_URL` to point to a deployed compatible audio deepfake classifier. If it is not configured, the API clearly says audio analysis is unavailable instead of fabricating a result.
- `/api/health` reports configuration status.

## Run locally
1. Python 3.11+
2. `pip install -r requirements.txt`
3. Set `HF_TOKEN` and optionally `IMAGE_MODEL` environment variables.
4. `uvicorn app.main:app --reload`
5. Open `http://127.0.0.1:8000`

## Deploy on Render
1. Upload all project files to a GitHub repository (not the ZIP itself).
2. In Render, create a Web Service from that repository; settings are included in `render.yaml`.
3. Add `HF_TOKEN` in Render Environment. Token must have inference permission. Never put it in frontend code or commit it.
4. Deploy and check `/api/health`.

## Model/accuracy limitations
The configured model may not be available from the selected inference provider. If so, set `IMAGE_MODEL` to a compatible, provider-served image AI detector or deploy a dedicated inference endpoint. Verify label names before interpreting scores. Model confidence is not the same as measured accuracy. Evaluate on a held-out dataset containing real, AI-generated, compressed, edited and out-of-domain files. Do not use detector output as sole proof.
