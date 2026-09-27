"""python3.11 -m flow.app [--port 8765] [--no-browser]

Starts Spec Studio locally and opens it in the browser.
"""

import argparse

from flow.app.server import serve

if __name__ == '__main__':
    ap = argparse.ArgumentParser(prog='python -m flow.app', description=__doc__)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--no-browser', action='store_true')
    a = ap.parse_args()
    serve(a.port, not a.no_browser)
