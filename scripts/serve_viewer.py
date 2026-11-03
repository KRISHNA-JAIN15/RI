"""Serve the 3-D viewer and run data on localhost (no caching, so live training shows up).

Usage: python scripts/serve_viewer.py [--port 8000] [--run runs/ddpg_paper]
"""

from __future__ import annotations

import argparse
import functools
import os
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        if self.path == "/favicon.ico":  # browsers request it even though the page defines an inline icon
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()

    def log_message(self, fmt, *args):  # quieter: hide the 4-second manifest polling
        if "manifest.json" not in (fmt % args if args else fmt):
            super().log_message(fmt, *args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--run", default="runs/ddpg_paper")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    os.chdir(ROOT)
    handler = functools.partial(NoCacheHandler, directory=str(ROOT))
    url = f"http://localhost:{args.port}/viewer/?run={args.run}"
    print(f"serving {ROOT}\nopen {url}")
    if not args.no_browser:
        webbrowser.open(url)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
