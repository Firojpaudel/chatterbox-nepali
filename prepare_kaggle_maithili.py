import os
import json
import argparse
import soundfile as sf
from datasets import load_dataset, Audio

def main():
    parser = argparse.ArgumentParser(description="Download and segregate Maithili data for Kaggle SFT.")
    parser.add_argument("--repo_id", type=str, default="Firoj112/chatterbox-multilingual-data")
    parser.add_argument("--output_dir", type=str, default="data/maithili_kaggle")
    args = parser.parse_args()

    print(f"📥 Downloading dataset from {args.repo_id}...")
    ds = load_dataset(args.repo_id, split="train")

    print("🔍 Filtering exclusively for Maithili ('mai') language tags...")
    def is_maithili(example):
        lang = example.get('language') or example.get('lang') or ""
        return lang.strip().lower() == "mai"

    mai_ds = ds.filter(is_maithili)
    mai_ds = mai_ds.cast_column("audio", Audio(decode=True))
    
    count = len(mai_ds)
    print(f"✅ Found {count} Maithili samples!")
    if count == 0:
        return

    wav_dir = os.path.join(args.output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "maithili_manifest.jsonl")
    
    print(f"💾 Saving wav files and generating manifest...")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(mai_ds):
            audio_data = item["audio"]
            wav_filename = f"mai_{i}.wav"
            wav_path = os.path.join(wav_dir, wav_filename)
            
            # Save audio to disk
            sf.write(wav_path, audio_data["array"], audio_data["sampling_rate"])
            
            # Write JSONL manifest entry
            manifest_item = {
                "audio_path": f"wavs/{wav_filename}",
                "text": item["text"],
                "duration": len(audio_data["array"]) / audio_data["sampling_rate"]
            }
            f.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")
            
            if (i+1) % 1000 == 0:
                print(f"Processed {i+1}/{count} files...")

    print("\n🎉 Segregation Complete!")
    print(f"Upload {args.output_dir}/ to Kaggle and run:")
    print(f"python src/chatterbox/train_maithili.py --manifest /kaggle/input/maithili_kaggle/maithili_manifest.jsonl")

if __name__ == "__main__":
    main()
