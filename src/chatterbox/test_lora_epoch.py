import argparse
import os
import torch
import soundfile as sf
from pathlib import Path
from peft import PeftModel
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def test_lora(ckpt_dir, text, lang, output_path, reference_audio):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Loading base model to {device}...")
    
    # Load base model
    model_wrapper = ChatterboxMultilingualTTS.from_pretrained(device)
    t3 = model_wrapper.t3
    
    # Resize embeddings to match tokenizer (crucial for custom tokens like [ne], [mai])
    vocab_size = len(model_wrapper.tokenizer.tokenizer.get_vocab())
    print(f"📏 Resizing T3 vocabulary to match tokenizer: {vocab_size} tokens")
    t3.resize_text_embeddings(vocab_size)
    
    print(f"🔄 Injecting LoRA adapter from {ckpt_dir}...")
    if ckpt_dir is not None:
        print(f"🎨 Applying LoRA adapter from: {ckpt_dir}")
        t3.tfmr = PeftModel.from_pretrained(t3.tfmr, ckpt_dir)
        t3.tfmr.to(device).eval()
    else:
        print("💡 No checkpoint provided. Using pure BASE model.")
    
    # Load text embeddings from the adapter folder
    if ckpt_dir is not None:
        emb_path = os.path.join(ckpt_dir, "text_emb.pt")
        if os.path.exists(emb_path):
            print("📥 Loading updated text embeddings...")
            t3.text_emb.load_state_dict(torch.load(emb_path, map_location=device))
    
    t3.eval()
    
    print(f"📊 Generating audio for [{lang}]: {text[:100]}...")
    with torch.no_grad():
        wav = model_wrapper.generate(
            text, 
            language_id=lang, 
            audio_prompt_path=reference_audio,
            temperature=0.8
        )
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, wav.squeeze(0).cpu().numpy(), model_wrapper.sr)
    print(f"✅ Audio saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, help="Path to LoRA checkpoint directory")
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--lang", type=str, default="ne", help="Language code (ne, mai, en)")
    parser.add_argument("--output", type=str, help="Output wav path")
    parser.add_argument("--ref", type=str, default="samples/Prakash.mp3", help="Reference audio path")
    args = parser.parse_args()
    
    # Default long texts if not provided
    samples = {
        "ne": "नेपाल दक्षिण एसियामा अवस्थित एक भूपरिवेष्टित देश हो। यसको भौगोलिक विविधता र सांस्कृतिक सम्पदा विश्वमै अद्वितीय छ। सगरमाथाको देश भनेर चिनिने नेपालमा विभिन्न भाषा र जातजातिका मानिसहरू बसोबास गर्छन्।",
        "mai": "मिथिलाक सांस्कृतिक विरासत बहुत समृद्ध अछि। जानकी मंदिरक सुन्दरता आ ओहि ठामक भाषाक मधुरता सभकेँ मोहित कए दैत अछि। ई क्षेत्र अपन कला आ साहित्यक लेल विश्व विख्यात अछि।",
        "en": "The transition to a multilingual model allows for a more natural and expressive synthesis of diverse languages. This fine-tuning process specifically targets the tonal nuances of Nepali and Maithili speech patterns."
    }
    
    if args.text:
        test_lora(args.ckpt if args.ckpt else None, args.text, args.lang, args.output or f"samples/test_results/test_{args.lang}.wav", args.ref)
    else:
        # Run all three
        for lang, text in samples.items():
            out = f"samples/test_results/epoch0_{lang}.wav"
            test_lora(args.ckpt, text, lang, out, args.ref)
