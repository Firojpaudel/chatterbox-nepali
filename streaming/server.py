import os
import json
import time
import torch
import numpy as np
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from streamer import StreamingChatterbox

# Setup logging to file
LOG_FILE = "streaming_server.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileWithMetadata(LOG_FILE) if hasattr(logging, "FileWithMetadata") else logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StreamingServer")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "lora_nepali_epoch_4")

model = None
streamer = None

def get_model():
    global model, streamer
    if model is None:
        logger.info("📥 Loading Chatterbox model...")
        model = ChatterboxMultilingualTTS.from_pretrained("cuda")
        
        # Load the latest LoRA correctly using PEFT
        if os.path.exists(CHECKPOINT_PATH):
            from peft import PeftModel
            logger.info(f"Applying latest LoRA: {CHECKPOINT_PATH}")
            model.t3.tfmr = PeftModel.from_pretrained(model.t3.tfmr, CHECKPOINT_PATH)
            model.t3.tfmr = model.t3.tfmr.to("cuda")
            
            # CRITICAL: Load fine-tuned text embeddings for Nepali tonality/accent
            emb_path = os.path.join(CHECKPOINT_PATH, "text_emb.pt")
            if os.path.exists(emb_path):
                logger.info(f"Loading LoRA text embeddings: {emb_path}")
                state = torch.load(emb_path, map_location="cpu", weights_only=True)
                state_vocab_size = state["weight"].shape[0] if "weight" in state else state.shape[0]
                model_vocab_size = model.t3.hp.text_tokens_dict_size
                
                if state_vocab_size != model_vocab_size:
                    logger.info(f"Resizing T3 vocabulary from {model_vocab_size} to {state_vocab_size}")
                    model.t3.resize_text_embeddings(state_vocab_size)
                
                if "weight" in state:
                    model.t3.text_emb.load_state_dict(state)
                else:
                    model.t3.text_emb.weight.data = state.to(model.t3.text_emb.weight.device)
                logger.info("✅ LoRA Text Embeddings loaded.")
        
        # Default conditionals (Prakash)
        ref_path = os.path.join(PROJECT_ROOT, "samples", "Prakash.mp3")
        if os.path.exists(ref_path):
            model.prepare_conditionals(wav_fpath=ref_path)
            
        streamer = StreamingChatterbox(model)
        logger.info("✅ Model and Streamer ready.")
    return streamer

@app.on_event("startup")
async def startup_event():
    get_model()

@app.get("/")
async def get():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")
    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            text = request.get("text", "")
            language = request.get("language", "ne")
            
            if not text:
                continue

            logger.info(f"TTS Request: '{text[:50]}...' [{language}]")
            
            await websocket.send_json({
                "status": "starting",
                "sample_rate": streamer.sr
            })

            # Multilingual-optimized: 50 tokens = ~2s audio, amortizes vocoding overhead
            # High Quality Parallel: 35 tokens (~1.4s), 40 tokens context window
            for chunk in streamer.stream_inference(
                text=text,
                language_id=language,
                chunk_size_tokens=35,
                context_window=40,
            ):
                # Send audio chunk as binary
                audio_data = chunk['audio'].astype(np.float32).tobytes()
                await websocket.send_bytes(audio_data)
                
                # Send performance metadata
                meta = {
                    "status": "streaming",
                    "delta": chunk['delta'],
                    "is_final": chunk['is_final']
                }
                if chunk.get('ttfb') is not None:
                    meta['ttfb'] = chunk['ttfb']
                    logger.info(f"⏱️ TTFB: {chunk['ttfb']:.0f}ms")
                
                # Log inter-chunk delta (latency between segments)
                logger.info(f"📦 Chunk Delta: {chunk['delta']:.0f}ms")
                
                await websocket.send_json(meta)

            await websocket.send_json({"status": "complete"})
            logger.info("TTS streaming complete")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
