import json
import os
import sys

# Ensure src is in path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from chatterbox.utils.sanitizer import sanitize_text

RAW_FILE = os.path.join(CURRENT_DIR, "raw_banking_data.jsonl")
OUTPUT_FILE = os.path.join(CURRENT_DIR, "banking_test_cases.jsonl")

def process():
    if not os.path.exists(RAW_FILE):
        print(f"Error: {RAW_FILE} not found")
        return

    processed_count = 0
    with open(RAW_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
                raw_text = item.get("text", "")
                lang = item.get("language", "english").lower()
                
                # Determine sanitization language
                # "codemix" and "nepali" -> "ne"
                # "english" -> "en"
                san_lang = "en" if lang == "english" else "ne"
                
                sanitized_text = sanitize_text(raw_text, lang=san_lang)
                
                # Create processed item
                processed_item = {
                    "original_text": raw_text,
                    "sanitized_text": sanitized_text,
                    "category": item.get("category", "General"),
                    "language": lang,
                    "sanitization_lang": san_lang
                }
                
                f_out.write(json.dumps(processed_item, ensure_ascii=False) + "\n")
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing line: {line[:50]}... Error: {e}")

    print(f"Successfully processed {processed_count} test cases.")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process()