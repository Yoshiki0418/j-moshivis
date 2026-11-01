"""Read-only Firefox layout check against a running dataset viewer.

The browser may enter edit mode but never saves or approves dataset records.
PNG screenshot data is emitted as base64 on stdout; diagnostics go to stderr.
"""
import argparse
import json
import socket
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--profile', required=True)
parser.add_argument('--width', type=int, default=1600)
parser.add_argument('--height', type=int, default=1100)
parser.add_argument('--edit', action='store_true')
parser.add_argument('--scroll', type=int, default=0)
parser.add_argument('--context', action='store_true')
parser.add_argument('--hide-list', action='store_true')
parser.add_argument('--media', action='store_true')
parser.add_argument('--verify', action='store_true')
parser.add_argument('--url', default='http://127.0.0.1:8765/#annotate')
parser.add_argument('--firefox', default='/snap/firefox/current/usr/lib/firefox/firefox')
args = parser.parse_args()

try:
    probe = socket.create_connection(('127.0.0.1', 2828), timeout=.3)
except OSError:
    pass
else:
    probe.close()
    raise RuntimeError('Port 2828 is already occupied; do not use another browser session')
process = subprocess.Popen([args.firefox, '--headless', '--no-remote', '--profile', args.profile, '--marionette'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
connection = None
buffer = b''
sequence = 0
def receive():
    global buffer
    while b':' not in buffer:
        chunk = connection.recv(65536)
        if not chunk:
            raise ConnectionError('Browser connection closed')
        buffer += chunk
    prefix, rest = buffer.split(b':', 1)
    size = int(prefix)
    while len(rest) < size:
        chunk = connection.recv(65536)
        if not chunk:
            raise ConnectionError('Browser connection closed')
        rest += chunk
    buffer = rest[size:]
    return json.loads(rest[:size])

def command(name, parameters):
    global sequence
    sequence += 1
    packet = json.dumps([0, sequence, name, parameters]).encode()
    connection.sendall(str(len(packet)).encode() + b':' + packet)
    response = receive()
    if response[2]:
        raise RuntimeError(response[2])
    return response[3]

def js(script):
    return command('WebDriver:ExecuteScript', {'script': script, 'args': [], 'newSandbox': False, 'sandbox': 'default'})['value']

try:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            connection = socket.create_connection(('127.0.0.1', 2828), timeout=3)
            break
        except OSError:
            time.sleep(.1)
    if connection is None:
        raise RuntimeError('Firefox Marionette did not start')
    connection.settimeout(25)
    receive()
    command('WebDriver:NewSession', {'capabilities': {'alwaysMatch': {'acceptInsecureCerts': True}, 'firstMatch': [{}]}})
    command('WebDriver:SetWindowRect', {'width': args.width, 'height': args.height})
    command('WebDriver:Navigate', {'url': args.url})
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        if js("return !document.getElementById('detail-content').hidden && document.getElementById('detail-dialogue').children.length > 0;"):
            break
        time.sleep(.2)
    else:
        raise RuntimeError('Dataset detail did not load')
    if args.hide_list:
        js("document.getElementById('annotation-list-toggle').click();return true;")
    if args.context:
        js("document.getElementById('review-context-details').open=true;return true;")
    before_edit = js("const d=document.getElementById('detail-dialogue').getBoundingClientRect();const i=document.getElementById('image-viewer').getBoundingClientRect();return {dialogueY:d.y+scrollY,imageY:i.y+scrollY,height:d.height};")
    if args.edit:
        js("document.getElementById('review-edit').click();return true;")
    if args.media:
        js("document.getElementById('audio-section').open=true;document.getElementById('alignment-section').open=true;return true;")
    # Only observe/edit the browser draft. Never save or approve production data.
    js("document.documentElement.style.scrollBehavior='auto';window.scrollTo(0," + str(args.scroll) + ");return true;")
    time.sleep(.4)
    geometry = js("""
      const ids=['detail-panel','detail-content','detail-dialogue','image-viewer','detail-image','review-section','review-actions','dialogue-heading-actions','audio-section','alignment-section'];
      const result={viewport:[innerWidth,innerHeight],page:[document.documentElement.scrollWidth,document.documentElement.scrollHeight],scrollY};
      for(const id of ids){const el=document.getElementById(id);const r=el.getBoundingClientRect();result[id]={x:r.x,y:r.y,w:r.width,h:r.height,scrollHeight:el.scrollHeight,hidden:el.hidden};}
      result.turns=document.querySelectorAll('#detail-dialogue .message').length;
      result.inputs=[...document.querySelectorAll('#detail-dialogue textarea')].map(el=>({height:el.clientHeight,scroll:el.scrollHeight}));
      return result;
    """)
    print(json.dumps(geometry), file=sys.stderr)
    if args.verify:
        dialogue = geometry['detail-dialogue']
        image = geometry['image-viewer']
        viewport_width = geometry['viewport'][0]
        assert geometry['page'][0] <= viewport_width + 1, 'Horizontal page overflow'
        expected_height = 420 if viewport_width <= 800 else 600
        assert abs(dialogue['h'] - expected_height) < 1, 'Transcript height changed'
        assert abs(image['h'] - dialogue['h']) < 1, 'Image/transcript height mismatch'
        if viewport_width > 800:
            assert abs(image['y'] - dialogue['y']) < 1, 'Image/transcript top mismatch'
            assert image['x'] + image['w'] <= dialogue['x'], 'Work panels overlap'
        else:
            assert image['y'] + image['h'] <= dialogue['y'], 'Stacked panels overlap'
        if args.edit:
            assert abs(before_edit['dialogueY'] - dialogue['y'] - geometry['scrollY']) < 1, 'Editing moved the transcript'
            assert abs(before_edit['imageY'] - image['y'] - geometry['scrollY']) < 1, 'Editing moved the image'
            assert geometry['inputs'], 'No edit inputs'
            assert all(i['scroll'] <= i['height'] + 2 for i in geometry['inputs']), 'Text clipped inside edit input'
        controls = geometry['dialogue-heading-actions' if args.edit else 'review-actions']
        toolbar = geometry['review-section']
        assert controls['w'] > 0 and controls['h'] > 0, 'Actions are not visible'
        assert controls['x'] >= toolbar['x'] and controls['x'] + controls['w'] <= toolbar['x'] + toolbar['w'] + 1, 'Actions outside toolbar'
        assert controls['y'] >= toolbar['y'] and controls['y'] + controls['h'] <= toolbar['y'] + toolbar['h'] + 1, 'Actions overlap work area'
        print('Browser geometry checks: PASS', file=sys.stderr)
    screenshot = command('WebDriver:TakeScreenshot', {'id': None, 'highlights': [], 'full': False, 'scroll': False})
    print(screenshot['value'])
finally:
    if connection:
        connection.close()
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
