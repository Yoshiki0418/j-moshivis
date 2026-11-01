"""Find unique Fuji-mentioned images in the checkpoint's verified train split."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'eval_results/fuji_training_probe_20261010'
OUT.mkdir(parents=True, exist_ok=True)
run = json.loads((ROOT/'checkpoints/llmjp_v3_id0w1_run1/run.json').read_text())
record = run['metadata']['datasets']['train']
train = Path(record['path'])
digest = hashlib.sha256(train.read_bytes()).hexdigest()
assert digest == record['sha256'], 'Training manifest changed since training'
found = {}
matched_rows = 0
for line_number, line in enumerate(train.open(), 1):
    row = json.loads(line)
    if '富士山' not in row.get('text', ''):
        continue
    matched_rows += 1
    image = Path(row['image'])
    if not image.is_file():
        continue
    key = hashlib.sha256(image.read_bytes()).hexdigest()
    if key not in found:
        found[key] = dict(row, train_line=line_number, image_sha256=key)
candidates = list(found.values())
(OUT/'candidates.json').write_text(json.dumps(candidates, ensure_ascii=False, indent=2))
(OUT/'search_summary.json').write_text(json.dumps({
    'train_manifest': str(train), 'verified_train_sha256': digest,
    'matched_rows': matched_rows, 'unique_available_images': len(candidates),
    'selection_note': 'Text mentions Fuji; image verification is required. Training membership does not prove a sampled exposure.'
}, ensure_ascii=False, indent=2))
sheet = Image.new('RGB', (1600, 1200), 'white')
draw = ImageDraw.Draw(sheet)
for i, item in enumerate(candidates[:20]):
    im = Image.open(item['image']).convert('RGB')
    im.thumbnail((390, 205))
    x, y = (i%4)*400, (i//4)*240
    sheet.paste(im, (x+(400-im.width)//2, y))
    draw.text((x+10, y+210), f'{i+1}: {item["uid"][:16]}', fill='black')
sheet.save(OUT/'candidates_contact.jpg')
print(json.dumps({'matched_rows': matched_rows, 'unique_images': len(candidates)}, ensure_ascii=False))
for i, item in enumerate(candidates[:20]):
    text = item['text']
    pos = text.index('富士山')
    print(i+1, item['uid'][:16], text[max(0,pos-65):pos+100])
