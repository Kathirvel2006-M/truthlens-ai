import os, io, time, statistics
from typing import Any
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
import cv2
import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = APP_DIR
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_API_BASE = os.getenv("HF_API_BASE", "https://router.huggingface.co/hf-inference/models").rstrip("/")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "umm-maybe/AI-image-detector").strip()
AUDIO_MODEL = os.getenv("AUDIO_MODEL", "Vansh180/deepfake-audio-wav2vec2").strip()
AUDIO_ENDPOINT_URL = os.getenv("AUDIO_ENDPOINT_URL", "").strip()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "40"))
MAX_VIDEO_FRAMES = int(os.getenv("MAX_VIDEO_FRAMES", "8"))
ALLOWED = {
    "image": {"image/jpeg", "image/png", "image/webp"},
    "video": {"video/mp4", "video/webm", "video/quicktime"},
    "audio": {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp4", "audio/aac", "audio/ogg", "audio/webm", "audio/flac"}
}

app = FastAPI(title="TruthLens AI", version="1.0.0")

@app.get("/")
def home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/api/health")
def health():
    return {"status":"ok", "model_configured":bool(HF_TOKEN), "image_model":IMAGE_MODEL,
            "audio_model":AUDIO_MODEL, "audio_endpoint_configured":bool(AUDIO_ENDPOINT_URL),
            "max_upload_mb":MAX_UPLOAD_MB}

def hf_post(model: str, data: bytes, content_type: str, timeout: int = 90):
    if not HF_TOKEN:
        raise HTTPException(status_code=503, detail="AI service is not configured yet. The site owner must add HF_TOKEN in hosting environment variables.")
    url = f"{HF_API_BASE}/{model}"
    try:
        resp = requests.post(url, headers={"Authorization":f"Bearer {HF_TOKEN}","Content-Type":content_type},
                             data=data, timeout=timeout)
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="The AI model took too long to respond. Please try again with a smaller file.")
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="Could not reach the AI model service. Please try again later.")
    if resp.status_code in (401,403):
        raise HTTPException(status_code=502, detail="The AI API token is invalid or does not have inference permission.")
    if resp.status_code == 404:
        raise HTTPException(status_code=502, detail=f"Model '{model}' was not found by the inference service. The site owner must configure a supported model.")
    if resp.status_code == 503:
        raise HTTPException(status_code=503, detail="The selected AI model is loading or is not available from this inference provider. Try later or configure a supported endpoint.")
    if not resp.ok:
        msg = "The AI model rejected this file."
        try:
            msg = resp.json().get("error", msg)
        except Exception:
            pass
        raise HTTPException(status_code=502, detail=str(msg)[:240])
    try:
        result = resp.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="The AI model returned an unexpected response.")
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=502, detail=str(result["error"])[:240])
    return result

def parse_scores(result: Any):
    # Typical image/audio classification output: [{label, score}, ...]
    if isinstance(result, dict) and "labels" in result and "scores" in result:
        return [{"label":str(l), "score":float(s)} for l,s in zip(result["labels"],result["scores"])]
    if isinstance(result, list):
        out=[]
        for item in result:
            if isinstance(item, dict) and "label" in item and "score" in item:
                try: out.append({"label":str(item["label"]), "score":float(item["score"])})
                except (TypeError,ValueError): pass
        return out
    return []

def classify_image_bytes(data: bytes):
    result = hf_post(IMAGE_MODEL, data, "image/jpeg", timeout=90)
    scores = parse_scores(result)
    if not scores:
        raise HTTPException(status_code=502, detail="The image model returned an unsupported response format. Configure a compatible image-classification model.")
    return scores

def score_interpretation(scores):
    def normalized(label):
        return label.lower().replace("_"," ").replace("-"," ").strip()
    # Do not assume every model uses the same labels. Refuse to fabricate a fake/real verdict.
    fake_terms = ("artificial","ai generated","ai generated image","fake","synthetic","generated","deepfake","spoof")
    real_terms = ("real","human","authentic","bonafide","bona fide","original","natural")
    fake = [x for x in scores if any(t in normalized(x["label"]) for t in fake_terms)]
    real = [x for x in scores if any(t in normalized(x["label"]) for t in real_terms)]
    if fake and real:
        fake_score=max(x["score"] for x in fake)
        real_score=max(x["score"] for x in real)
        confidence=max(fake_score,real_score)
        if confidence < .60:
            verdict="Inconclusive"
        else:
            verdict="Likely AI-generated / manipulated" if fake_score > real_score else "Likely authentic"
        return {"verdict":verdict,"confidence":round(confidence*100,1),
                "ai_score":round(fake_score*100,1),"real_score":round(real_score*100,1)}
    if fake:
        s=max(fake,key=lambda x:x["score"])
        return {"verdict":"Possible AI-generated / manipulated" if s["score"] >= .60 else "Inconclusive",
                "confidence":round(s["score"]*100,1),"ai_score":round(s["score"]*100,1),"real_score":None}
    if real:
        s=max(real,key=lambda x:x["score"])
        return {"verdict":"Possible authentic media" if s["score"] >= .60 else "Inconclusive",
                "confidence":round(s["score"]*100,1),"ai_score":None,"real_score":round(s["score"]*100,1)}
    return {"verdict":"Model output needs review","confidence":None,"ai_score":None,"real_score":None}

def make_response(media_type, scores, started, method, extra=None):
    sorted_scores=sorted(scores,key=lambda x:x["score"],reverse=True)[:5]
    interpretation=score_interpretation(sorted_scores)
    return {"ok":True,"media_type":media_type,**interpretation,
            "top_scores":[{"label":s["label"],"score":round(s["score"]*100,2)} for s in sorted_scores],
            "method":method,"processing_ms":round((time.time()-started)*1000),
            "warning":"AI detector estimates can be wrong, especially on compressed, edited, unfamiliar, or out-of-domain media. Do not use this result as sole proof.",
            **(extra or {})}

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    started=time.time()
    data=await file.read()
    if not data:
        raise HTTPException(status_code=400,detail="The uploaded file is empty.")
    if len(data)>MAX_UPLOAD_MB*1024*1024:
        raise HTTPException(status_code=413,detail=f"File is too large. Maximum size is {MAX_UPLOAD_MB} MB.")
    ctype=(file.content_type or "").lower()
    name=(file.filename or "upload").lower()
    if ctype in ALLOWED["image"] or name.endswith((".jpg",".jpeg",".png",".webp")):
        try:
            image=Image.open(io.BytesIO(data))
            image.verify()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=400,detail="This image file is invalid or unsupported.")
        scores=classify_image_bytes(data)
        return make_response("image",scores,started,f"Image classification model: {IMAGE_MODEL}")
    if ctype in ALLOWED["video"] or name.endswith((".mp4",".webm",".mov")):
        # Extract a small number of frames. Temporary file is removed immediately.
        import tempfile
        suffix=".mp4" if name.endswith(".mp4") else (".webm" if name.endswith(".webm") else ".mov")
        path=None
        try:
            with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
                tmp.write(data); path=tmp.name
            cap=cv2.VideoCapture(path)
            if not cap.isOpened():
                raise HTTPException(status_code=400,detail="Could not read this video. Try a standard MP4 file.")
            total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total <= 0:
                raise HTTPException(status_code=400,detail="This video has no readable frames.")
            indices=np.linspace(0,max(0,total-1),num=min(MAX_VIDEO_FRAMES,total),dtype=int)
            frame_scores=[]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx))
                ok,frame=cap.read()
                if not ok: continue
                ok,buf=cv2.imencode(".jpg",frame)
                if ok: frame_scores.append(classify_image_bytes(buf.tobytes()))
            cap.release()
            if not frame_scores:
                raise HTTPException(status_code=400,detail="Could not extract frames from this video.")
            # Average per-label scores across sampled frames; this is a frame-based estimate, not a temporal deepfake detector.
            agg={}
            for fs in frame_scores:
                for item in fs: agg.setdefault(item["label"],[]).append(item["score"])
            scores=[{"label":k,"score":statistics.mean(v)} for k,v in agg.items()]
            return make_response("video",scores,started,f"Frame sampling with image model: {IMAGE_MODEL}",
                                 {"frames_analyzed":len(frame_scores),"frames_sampled":len(indices),
                                  "video_method_note":"Frame-based screening only. This is not a dedicated temporal video deepfake model."})
        finally:
            if path and os.path.exists(path):
                try: os.remove(path)
                except OSError: pass
    if ctype in ALLOWED["audio"] or name.endswith((".mp3",".wav",".m4a",".aac",".ogg",".flac",".webm")):
        if AUDIO_ENDPOINT_URL:
            if not HF_TOKEN:
                raise HTTPException(status_code=503,detail="Audio endpoint is configured but HF_TOKEN is missing.")
            try:
                resp=requests.post(AUDIO_ENDPOINT_URL,headers={"Authorization":f"Bearer {HF_TOKEN}","Content-Type":ctype or "application/octet-stream"},
                                   data=data,timeout=120)
            except requests.Timeout:
                raise HTTPException(status_code=504,detail="Audio analysis timed out. Try a shorter audio clip.")
            except requests.RequestException:
                raise HTTPException(status_code=502,detail="Could not reach the audio detection endpoint.")
            if not resp.ok:
                raise HTTPException(status_code=502,detail=f"Audio detector endpoint returned HTTP {resp.status_code}. Check endpoint configuration.")
            try: scores=parse_scores(resp.json())
            except Exception: scores=[]
            if not scores:
                raise HTTPException(status_code=502,detail="Audio endpoint response was not a compatible list of label/score values.")
            return make_response("audio",scores,started,"Configured audio classification endpoint")
        # Inference provider support differs by model; do not pretend a regular image API can classify audio.
        raise HTTPException(status_code=503,detail="Audio detection is not active yet. Configure AUDIO_ENDPOINT_URL to a deployed audio deepfake-classification endpoint. No result was fabricated.")
    raise HTTPException(status_code=415,detail="Unsupported file type. Upload JPG, PNG, WEBP, MP4, WEBM, MOV, MP3, WAV, M4A, AAC, OGG, or FLAC.")
