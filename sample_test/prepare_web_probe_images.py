"""Register only explicitly selected previous diagnostic images, not the workspace."""
import json
from pathlib import Path
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'j-moshivis/web/samples'
SAMPLES=[('fuji_cows','富士山と牛','fdda4ed4-2a7b-4589-a425-6fad66319128.jpg'),
    ('probe_webp','検証画像：20230713','20230713_145512_23239672_w1920.webp'),
    ('probe_960','検証画像：960.png','960.png'),
    ('probe_avif','検証画像：追加AVIF','c0ac1b57078410974e0c8929cfa999e527fe1a514c08dcc79f6a66aed5cda49e.avif'),
    ('test3','日本の町並み：test3','test3.jpg')]
OUT.mkdir(parents=True,exist_ok=True)
catalog=[]
for name,label,filename in SAMPLES:
    with Image.open(ROOT/filename) as source:
        image=ImageOps.exif_transpose(source).convert('RGB')
        image.save(OUT/f'{name}.png')
    catalog.append({'id':name,'label':label,'url':f'/samples/{name}.png','source_filename':filename})
(OUT/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2))
print(f'Registered {len(catalog)} previous diagnostic images')
