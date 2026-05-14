import os
import json
import time
import torch
import numpy as np
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from streamer import StreamingChatterbox

# Setup logging
LOG_FILE = "streaming_server.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileWithMetadata(LOG_FILE) if hasattr(logging, "FileWithMetadata") else logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ChatterboxAPI")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "lora_nepali_epoch_4")

model = None
streamer = None

def load_model():
    global model, streamer
    if model is None:
        logger.info("📥 Loading Chatterbox model...")
        model = ChatterboxMultilingualTTS.from_pretrained("cuda")
        
        # Load LoRA
        if os.path.exists(CHECKPOINT_PATH):
            from peft import PeftModel
            logger.info(f"Applying latest LoRA: {CHECKPOINT_PATH}")
            model.t3.tfmr = PeftModel.from_pretrained(model.t3.tfmr, CHECKPOINT_PATH)
            model.t3.tfmr = model.t3.tfmr.to("cuda")
            
            # Text Embeddings
            emb_path = os.path.join(CHECKPOINT_PATH, "text_emb.pt")
            if os.path.exists(emb_path):
                logger.info(f"Loading LoRA text embeddings...")
                state = torch.load(emb_path, map_location="cpu", weights_only=True)
                state_vocab_size = state["weight"].shape[0] if "weight" in state else state.shape[0]
                model_vocab_size = model.t3.hp.text_tokens_dict_size
                
                if state_vocab_size != model_vocab_size:
                    model.t3.resize_text_embeddings(state_vocab_size)
                
                if "weight" in state:
                    model.t3.text_emb.load_state_dict(state)
                else:
                    model.t3.text_emb.weight.data = state.to(model.t3.text_emb.weight.device)
        
        # Default conditionals (Prensa)
        ref_path = os.path.join(PROJECT_ROOT, "samples", "prensa_fixed.mp3")
        if os.path.exists(ref_path):
            model.prepare_conditionals(wav_fpath=ref_path)
            
        streamer = StreamingChatterbox(model)
        logger.info("✅ Model and Streamer ready.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_model()
    yield
    # Shutdown
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("👋 Shutting down Chatterbox API")

app = FastAPI(
    title="Chatterbox Streaming API",
    description="Ultra-low latency Nepali Text-to-Speech streaming engine.",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSRequest(BaseModel):
    text: str
    language: str = "ne"

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_index():
    """Returns the streaming playground UI."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r") as f:
        return f.read()

@app.post("/synthesize")
async def synthesize(request: TTSRequest):
    """
    HTTP endpoint for TTS synthesis. 
    Returns a streaming response of raw float32 PCM audio.
    """
    if not streamer:
        raise HTTPException(status_code=503, detail="Model not initialized")

    def audio_generator():
        for chunk in streamer.stream_inference(
            text=request.text,
            language_id=request.language,
            chunk_size_tokens=35,
            context_window=40,
        ):
            yield chunk['audio'].astype(np.float32).tobytes()

    return StreamingResponse(audio_generator(), media_type="audio/l16;rate=24000")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming with performance metadata.
    """
    await websocket.accept()
    logger.info("WebSocket connected")
    try:
        while True:
            data = await websocket.receive_text()
            req = json.loads(data)
            text = req.get("text", "")
            language = req.get("language", "ne")
            
            if not text: continue

            logger.info(f"WS TTS: '{text[:50]}...' [{language}]")
            await websocket.send_json({"status": "starting", "sample_rate": streamer.sr})

            for chunk in streamer.stream_inference(
                text=text,
                language_id=language,
                chunk_size_tokens=35,
                context_window=40,
            ):
                await websocket.send_bytes(chunk['audio'].astype(np.float32).tobytes())
                
                meta = {
                    "status": "streaming",
                    "delta": chunk['delta'],
                    "is_final": chunk['is_final']
                }
                if chunk.get('ttfb') is not None:
                    meta['ttfb'] = chunk['ttfb']
                
                await websocket.send_json(meta)

            await websocket.send_json({"status": "complete"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
