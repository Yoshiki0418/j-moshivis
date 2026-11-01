"""Local HTTP/WebSocket smoke check with an existing user-only TTS recording."""
import asyncio
import json
import time
from pathlib import Path
import aiohttp
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
BASE='http://127.0.0.1:8998'

async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        for attempt in range(180):
            try:
                async with session.get(BASE+'/api/health') as response:
                    response.raise_for_status();health=await response.json()
                break
            except aiohttp.ClientConnectionError:
                await asyncio.sleep(1)
        else:raise RuntimeError('Server did not become ready')
        async with session.get(BASE+'/samples/catalog.json') as response:
            response.raise_for_status();samples=await response.json()
        assert len(samples)==5
        for sample in samples:
            async with session.get(BASE+sample['url']) as response:
                response.raise_for_status();assert response.content_type=='image/png'
                assert len(await response.read())>100
        manifest=ROOT/'eval_results/image_probe_test3_greedy_20261012/manifest_audio.jsonl'
        row=json.loads(manifest.read_text().splitlines()[0])
        # The manifest is authoritative for the user-only question recording.
        audio_path=Path(row['audio_path'])
        if not audio_path.is_absolute():audio_path=manifest.parent/audio_path
        audio,sr=sf.read(audio_path,dtype='float32');assert sr==24000 and audio.ndim==1
        frames=audio[:len(audio)//1920*1920].reshape(-1,1920)
        report={'health':health,'sample_images':len(samples),'generated_audio_packets':0,'text':'','errors':[]}
        started=time.monotonic()
        async with session.ws_connect(BASE+'/api/chat?text_temperature=0&audio_temperature=0.5',timeout=120) as ws:
            await ws.send_bytes(b'\x08'+(ROOT/'test3.jpg').read_bytes())
            while True:
                message=await asyncio.wait_for(ws.receive(),120)
                if message.type!=aiohttp.WSMsgType.BINARY:raise RuntimeError(str(message))
                if message.data[0]==4:raise RuntimeError(message.data[1:].decode())
                if message.data[0]==0:break
            for frame in frames:await ws.send_bytes(b'\x01'+frame.astype('<f4').tobytes())
            await ws.send_bytes(b'\x02')
            while True:
                message=await asyncio.wait_for(ws.receive(),120)
                if message.type!=aiohttp.WSMsgType.BINARY:raise RuntimeError(str(message))
                kind=message.data[0]
                if kind==1:
                    values=np.frombuffer(message.data[1:],dtype='<f4')
                    assert values.size>0 and np.isfinite(values).all()
                    report['generated_audio_packets']+=1
                elif kind==7:report['text']+=message.data[2:].decode()
                elif kind==4:raise RuntimeError(message.data[1:].decode())
                elif kind==3 and '応答生成を続けました' in message.data[1:].decode():break
        assert report['generated_audio_packets']>10
        report['elapsed_sec']=time.monotonic()-started
        (ROOT/'logs/realtime_web_20261015_smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':asyncio.run(main())
