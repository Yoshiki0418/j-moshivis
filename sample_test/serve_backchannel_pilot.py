"""Read-only pilot review server with HTTP Range support and an asset allowlist."""
import argparse
import json
from pathlib import Path

from aiohttp import web


def create_app(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    allowed = {'': root / 'index.html', 'index.html': root / 'index.html'}
    for sample in manifest['samples']:
        for key in ['image_url', 'original_url', 'augmented_url',
                    'snippet_original_url', 'snippet_augmented_url']:
            rel = Path(sample[key])
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Non-local asset in manifest')
            allowed[rel.as_posix()] = root / rel

    async def serve(request):
        path = allowed.get(request.match_info['asset'])
        if path is None or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={'Cache-Control': 'no-cache',
                                              'X-Content-Type-Options': 'nosniff'})

    app = web.Application()
    app.router.add_get('/{asset:.*}', serve)
    return app


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--port', type=int, default=8766)
    args = p.parse_args()
    web.run_app(create_app(args.root), host='127.0.0.1', port=args.port)
