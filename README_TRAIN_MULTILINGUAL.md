# Training Guide: Adding Multilingual Support (Maithili, Nepali, etc.) to Chatterbox TTS

This guide explains how to fine-tune the Chatterbox Multilingual model for multiple new languages (e.g., Maithili `mai`, Nepali `ne`) and how inference handles these custom language tokens.

## 1. Project Directory Structure

Organize your data like this for efficient multilingual training:

```text
chatterbox/
├── data/
│   ├── nepali/
│   │   ├── wavs/
│   │   └── manifest.jsonl
│   ├── maithili/
│   │   ├── wavs/
│   │   └── manifest.jsonl
├── src/
│   └── chatterbox/
│       └── train_multilingual.py   <-- The multilingual training script
└── ...
```

## 2. Creating the Manifest Files

The training script requires `.jsonl` (JSON Lines) files where each line maps audio to text and includes the target language ID.

**File path**: `data/maithili/manifest.jsonl`
**Content example**:
```json
{"audio_path": "data/maithili/wavs/train_001.wav", "text": "अहाँक नाम की छियै?", "language": "mai"}
{"audio_path": "data/maithili/wavs/train_002.wav", "text": "हमर नाम राम छै।", "language": "mai"}
```

## 3. How Tokenization Works for New Languages (e.g., Maithili)

You might wonder: *Do we have a dedicated `[mai]` token? How does the model know which language to speak?*

Yes, we dynamically generate and support dedicated language tokens like `[mai]` and `[ne]`. Here is exactly how it is solved in the architecture:

1. **Dynamic Token Registration**: In `src/chatterbox/models/tokenizers/tokenizer.py` inside the `MTLTokenizer` class, the code automatically registers any new tokens that aren't natively in the base model's vocabulary:
   ```python
   new_tokens = ["[ne]", "[mai]"]
   self.tokenizer.add_tokens([t for t in new_tokens if t not in self.tokenizer.get_vocab()])
   ```
2. **Text Preprocessing**: When text is passed into the tokenizer for inferencing or training with `language_id="mai"`, the tokenizer intercepts it and prepends `[mai]` to the text:
   ```python
   lang_tag = f"[{language_id.lower()}]" # e.g., "[mai]"
   if not txt.startswith(lang_tag):
       txt = f"{lang_tag}{txt}"
   ```
3. **Embedding Resizing**: When loading a checkpoint in `mtl_tts.py`, the code compares the vocabulary size of the base model with the new checkpoint. If it detects that new tokens like `[mai]` were added, it dynamically resizes the T3 transformer text embeddings so that it perfectly maps the custom `[mai]` token to its learned embedding.

## 4. Inferencing with Multiple Languages

If you want to introduce more than one language during inferencing, you need to follow these steps:

### A. Updating `SUPPORTED_LANGUAGES`
Ensure your target language code (e.g., `mai`) is defined in the `SUPPORTED_LANGUAGES` mapping inside `src/chatterbox/mtl_tts.py`:
```python
SUPPORTED_LANGUAGES = {
    ...
    "ne": "Nepali",
    "mai": "Maithili",
    ...
}
```
*(Note: We have already added `"mai": "Maithili"` to `gradio_app.py` for you!)*

### B. Calling the Generate Function
When generating speech, simply pass the target `language_id` (e.g., `"mai"`, `"ne"`, `"en"`). The `ChatterboxMultilingualTTS` class will route it correctly, injecting the `[mai]` token seamlessly.

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Load model
model = ChatterboxMultilingualTTS.from_pretrained("cuda")

# Generate Maithili Audio
maithili_wav = model.generate(
    text="अहाँक नाम की छियै?",
    language_id="mai"  # <-- THIS triggers the [mai] token logic internally
)

# Generate Nepali Audio
nepali_wav = model.generate(
    text="तपाईंको नाम के हो?",
    language_id="ne"   # <-- THIS triggers the [ne] token logic internally
)
```

## 5. UI Integration

For the `gradio_app.py`, the dropdown menu for languages is populated directly from the `SUPPORTED_LANGUAGES` dictionary. Because `"mai": "Maithili"` was added, you can simply select "Maithili" from the Gradio interface, type Maithili text, and hit generate. The Gradio app passes `language_id="mai"` right to the backend inference module.
