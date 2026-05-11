import pandas as pd
import os
import io
import urllib.request
import tarfile
import gc
from tqdm import tqdm

def download_and_format_english():
    print("🚀 Initializing English Data Augmentation (LJSpeech)...")
    
    output_dir = "data/chatterbox-multilingual-data/extra_english"
    os.makedirs(output_dir, exist_ok=True)
    
    tar_path = "LJSpeech-1.1.tar.bz2"
    if not os.path.exists(tar_path):
        print("📥 Downloading LJSpeech dataset (2.6 GB)...")
        urllib.request.urlretrieve("https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", tar_path)
    
    print("📦 Extracting tarball...")
    if not os.path.exists("LJSpeech-1.1"):
        with tarfile.open(tar_path, "r:bz2") as tar:
            tar.extractall()
    
    print("📋 Processing metadata...")
    metadata_path = "LJSpeech-1.1/metadata.csv"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    records = []
    chunk_size = 3000
    
    print(f"✅ Formatting {len(lines)} samples to match Chatterbox Parquet format...")
    
    # Needs soundfile and librosa to calculate duration
    import soundfile as sf
    
    for i, line in enumerate(tqdm(lines)):
        parts = line.strip().split('|')
        if len(parts) < 3:
            continue
            
        file_id = parts[0]
        text = parts[2] # normalized text
        
        wav_path = f"LJSpeech-1.1/wavs/{file_id}.wav"
        if not os.path.exists(wav_path):
            continue
            
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
            
        # Get duration
        info = sf.info(wav_path)
        duration = info.duration
        
        records.append({
            "audio": {"bytes": audio_bytes, "path": f"ljspeech_{file_id}.wav"},
            "text": text,
            "language": "en",
            "duration": duration
        })
        
        if len(records) >= chunk_size:
            df = pd.DataFrame(records)
            chunk_id = i // chunk_size
            out_path = os.path.join(output_dir, f"extra_en_chunk_{chunk_id}.parquet")
            df.to_parquet(out_path)
            print(f"\n💾 Saved {out_path} ({len(records)} samples)")
            records = []
            del df
            gc.collect()
            
    if records:
        df = pd.DataFrame(records)
        chunk_id = (i // chunk_size) + 1
        out_path = os.path.join(output_dir, f"extra_en_chunk_{chunk_id}.parquet")
        df.to_parquet(out_path)
        print(f"\n💾 Saved {out_path} ({len(records)} samples)")
        
    # Clean up
    print("🧹 Cleaning up extracted files...")
    import shutil
    shutil.rmtree("LJSpeech-1.1")
    os.remove(tar_path)
        
    print("\n🎉 Augmentation Complete! Extra English data is ready for V5.")

if __name__ == "__main__":
    download_and_format_english()
