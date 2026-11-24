"""Serve the provider inventory over a private local Unix socket."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
from socketserver import UnixStreamServer
import stat
from tempfile import TemporaryDirectory


@contextmanager
def _socket_path(path: Path | None):
    if path is None:
        with TemporaryDirectory(prefix="hc-") as root:
            yield Path(root) / "i.sock"
        return
    path = path.expanduser().absolute()
    parent = path.parent.stat()
    if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or parent.st_mode & 0o077:
        raise ValueError("Provider status socket requires a private, user-owned parent directory")
    yield path


def serve_provider_status(read_status: Callable[[], dict], socket_path: Path | None = None) -> None:
    class ProviderHandler(BaseHTTPRequestHandler):
        timeout = 5

        def log_message(self, format, *args):
            pass

        def do_GET(self):
            if self.path != "/v1/inference/providers":
                self.send_error(404)
                return
            body = json.dumps(read_status()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with _socket_path(socket_path) as path:
        if len(os.fsencode(path)) >= 104:
            raise ValueError("Provider status socket path is too long. Use --socket with a shorter path.")
        # A private parent directory protects the socket before chmod as well.
        with UnixStreamServer(str(path), ProviderHandler) as server:
            try:
                path.chmod(0o600)
                print(json.dumps({"socket": str(path)}), flush=True)
                server.serve_forever()
            finally:
                path.unlink(missing_ok=True)
