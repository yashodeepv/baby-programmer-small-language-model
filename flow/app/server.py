"""server.py - A local HTTP server for Spec Studio. Standard library only.

    GET  /               the app (static/index.html)
    GET  /api/prompt     the LLM prompt, as text
    GET  /api/examples   the example specs
    POST /api/build      {"spec": "..."} -> everything the UI shows (pipeline.build)

Binds to 127.0.0.1 only: it runs code generation and simulation on
whatever is pasted in, and is meant for the person at this machine.
"""

import glob
import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from flow.app.pipeline import build
from flow.app.prompt import build_prompt, example_spec

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'spec', 'examples')
MAX_BODY = 1_000_000


def examples():
    out = [{'name': 'leave_requests (all exact)', 'text': example_spec()}]
    for path in sorted(glob.glob(os.path.join(EXAMPLES, '*.spec.md'))):
        with open(path) as f:
            out.append({'name': os.path.basename(path)[:-len('.spec.md')], 'text': f.read()})
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = 'SpecStudio/0.1'

    def log_message(self, fmt, *args):
        pass                                    # keep the terminal quiet

    def send(self, code, body, ctype):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def json(self, obj, code=200):
        self.send(code, json.dumps(obj), 'application/json')

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            with open(os.path.join(STATIC, 'index.html'), 'rb') as f:
                self.send(200, f.read(), 'text/html; charset=utf-8')
        elif self.path == '/api/prompt':
            self.send(200, build_prompt(), 'text/plain; charset=utf-8')
        elif self.path == '/api/examples':
            self.json(examples())
        else:
            self.send(404, 'not found', 'text/plain')

    def do_POST(self):
        if self.path != '/api/build':
            return self.send(404, 'not found', 'text/plain')
        n = int(self.headers.get('Content-Length', 0))
        if n > MAX_BODY:
            return self.json({'fatal': 'the spec is too large (limit 1 MB)'}, 413)
        try:
            spec = json.loads(self.rfile.read(n) or b'{}').get('spec', '')
            self.json(build(spec))
        except Exception as e:                  # a bug here should show in the UI, not kill the server
            traceback.print_exc()
            self.json({'fatal': f'{type(e).__name__}: {e}'}, 500)


def serve(port=8765, open_browser=True):
    httpd = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    url = f'http://127.0.0.1:{port}/'
    print(f'Spec Studio running at {url}  (Ctrl+C to stop)')
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
