import json

with open('Chatterbox_NP_run.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

for cell in data['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if '"transformers<5.2.0"' in line:
                source[i] = line.replace(' "transformers<5.2.0"', '')
            # Just in case they had single quotes
            elif "'transformers<5.2.0'" in line:
                source[i] = line.replace(" 'transformers<5.2.0'", '')

with open('Chatterbox_NP_run.ipynb', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)
