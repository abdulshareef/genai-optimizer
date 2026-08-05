"""A tiny HTTP service, so PHP, Node, Java, .NET or n8n can also use this.

    genai-optimize serve --port 8088

Endpoints
    GET  /health
    GET  /models
    POST /optimize          {"prompt": "...", "level": "balanced"}
    POST /optimize/output   {"text": "..."}

Standard library only, no FastAPI needed. Good for a sidecar container or a
LAN service. Put it behind your own auth before exposing it outside.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .engine import OptimizationEngine
from .pricing import PRICES_UPDATED, list_models
from .security import SecurityBlocked

MAX_BODY = 4 * 1024 * 1024   # 4 MB


class Handler(BaseHTTPRequestHandler):
    server_version = "genai-optimizer/" + __version__

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/health"):
            self._send(200, {"status": "ok", "version": __version__})
        elif self.path.startswith("/models"):
            self._send(200, {"prices_updated": PRICES_UPDATED,
                             "models": [m.to_dict() for m in list_models()]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        try:
            data = self._body()
        except Exception as exc:
            return self._send(400, {"error": "bad json: %s" % exc})

        try:
            if self.path.startswith("/optimize/output"):
                engine = OptimizationEngine(output_level=data.get("level", "balanced"))
                result = engine.optimize_output(data.get("text", ""))
                return self._send(200, result.to_dict())

            if self.path.startswith("/optimize"):
                engine = OptimizationEngine(
                    level=data.get("level", "balanced"),
                    security_mode=data.get("security", "warn"),
                    priority=data.get("priority", "balanced"),
                    allowed_providers=data.get("providers"),
                    select_model=data.get("select_model", True),
                )
                result = engine.optimize_prompt(data.get("prompt", ""),
                                                system=data.get("system"))
                return self._send(200, result.to_dict())
        except SecurityBlocked as exc:
            return self._send(422, {"error": "blocked", "detail": str(exc)})
        except Exception as exc:  # pragma: no cover
            return self._send(500, {"error": str(exc)})

        self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):  # keep the console quiet
        return


def serve(host: str = "127.0.0.1", port: int = 8088) -> None:  # pragma: no cover
    httpd = ThreadingHTTPServer((host, port), Handler)
    print("genai-optimizer service running on http://%s:%d" % (host, port))
    print("Try: curl -s localhost:%d/health" % port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        httpd.server_close()
