"""Quick script to analyze language distribution in the multilingual dataset."""
import os
from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset, Audio

print("Loading dataset in streaming mode...")
ds = load_dataset("Firoj112/chatterbox-multilingual-data", split="train", streaming=True)
# Disable audio decoding to avoid the torchcodec error
ds = ds.cast_column("audio", Audio(decode=False))
ds = ds.remove_columns(["audio"])

lang_counts = {}
lang_duration = {}
total_samples = 0

print("Scanning dataset (this will take a few minutes)...")
for item in ds:
    lang = item.get('language', 'unknown')
    duration = item.get('duration', 0)
    
    lang_counts[lang] = lang_counts.get(lang, 0) + 1
    lang_duration[lang] = lang_duration.get(lang, 0) + duration
    total_samples += 1
    
    if total_samples % 1000 == 0:
        print(f"  Scanned {total_samples} samples so far...")
        for l in sorted(lang_counts.keys()):
            hrs = lang_duration[l] / 3600
            print(f"    {l}: {lang_counts[l]} samples, {hrs:.1f} hours")

print(f"\n{'='*60}")
print(f"FINAL DATASET ANALYSIS")
print(f"{'='*60}")
print(f"Total samples: {total_samples}")
total_hours = sum(lang_duration.values()) / 3600
print(f"Total duration: {total_hours:.1f} hours\n")

print(f"{'Language':<10} {'Samples':>10} {'Hours':>10} {'% Samples':>12} {'% Hours':>10}")
print(f"{'-'*52}")
for lang in sorted(lang_counts.keys(), key=lambda l: lang_duration[l], reverse=True):
    count = lang_counts[lang]
    hours = lang_duration[lang] / 3600
    pct_samples = count / total_samples * 100
    pct_hours = hours / total_hours * 100
    print(f"{lang:<10} {count:>10} {hours:>10.1f} {pct_samples:>11.1f}% {pct_hours:>9.1f}%")
