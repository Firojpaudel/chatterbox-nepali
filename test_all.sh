#!/bin/bash

# Usage: ./test_all.sh [checkpoint_dir]
# Example: ./test_all.sh lora_nepali_epoch_1

CKPT=$1

if [ -z "$CKPT" ]; then
    echo "Usage: ./test_all.sh [checkpoint_dir]"
    exit 1
fi

if [ ! -d "$CKPT" ]; then
    echo "Error: Directory $CKPT not found."
    exit 1
fi

echo "🚀 Testing Checkpoint: $CKPT"
mkdir -p samples/test_results

export PYTHONPATH=src

# Load env variables (for HF token if needed)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "--- Generating Nepali ---"
.venv/bin/python src/chatterbox/test_lora_epoch.py \
    --ckpt "$CKPT" \
    --lang ne \
    --text " पश्चिम एसियाली युद्धपछि बढेको मूल्यकै कारण दोलखाका अनिल न्यौपानेको इन्धनमा हुने खर्च ३५ प्रतिशत हाराहारी वृद्धि भएको छ । आफू बस्ने ठाउँ काठमाडौंको शंखमूलदेखि भक्तपुरको राधेराधेस्थित काम गर्ने ड्राइभिङ सेन्टरसम्म जाँदा–आउँदा साताको पाँच लिटर पेट्रोल लाग्ने उनी बताउँछन् । ‘पहिला साताको ७–८ सय रुपैयाँ भए पुग्थ्यो, अहिले हजार–एघार सय रुपैयाँसम्म लाग्छ,’ उनी भन्छन्, ‘अलि–अलि जोगिने पैसा पनि पेट्रोलमै सकिन्छ ।’ अहिले मासिक ४ हजार ५ सय रुपैयाँसम्म पेट्रोलमै खर्चिनुपरेको उनी सुनाउँछन् । " \
    --output "samples/test_results/${CKPT}_ne.wav"

echo "--- Generating Maithili ---"
.venv/bin/python src/chatterbox/test_lora_epoch.py \
    --ckpt "$CKPT" \
    --lang mai \
    --text "मिथिलाक संस्कृति बहुत पुरान आर समृद्ध अछि, एतय कऽ लोक अपन परम्परा आर कलाक लेल जानल जाइत छथि। हम अहाँक बहुत आभारी छी।" \
    --output "samples/test_results/${CKPT}_mai.wav"

echo "--- Generating English (Using BASE for stability) ---"
.venv/bin/python src/chatterbox/test_lora_epoch.py \
    --lang en \
    --text "The multilingual text to speech system is now learning Nepali and Maithili through LoRA fine-tuning. We have successfully completed five epochs and are moving towards more stable generations." \
    --output "samples/test_results/${CKPT}_en_base.wav"

echo "✅ All tests completed. Samples saved in: samples/test_results/"
