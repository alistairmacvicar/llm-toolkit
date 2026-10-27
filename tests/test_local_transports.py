"""Exercise installed HTTP/WebSocket clients against local, credential-free servers."""

import asyncio
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from llm_toolkit.agents.realtime import RealtimeSession
from llm_toolkit.media.file_source import load_link, save_link


class DownloadHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        if self.path == "/chunked":
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for chunk in (b"abc", b"defg", b"hij"):
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
        elif self.path == "/truncated":
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b"partial")
        elif self.path == "/stall":
            self.send_header("Content-Length", "4")
            self.end_headers()
            self.wfile.flush()
            time.sleep(0.2)
        else:
            self.send_header("Content-Length", "10")
            self.end_headers()
            self.wfile.write(b"abcdefghij")


@pytest.fixture
def download_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), DownloadHandler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("path", ["/known", "/chunked"])
def test_real_http_downloads_enforce_limit_and_decode_transfer_encoding(download_server, path):
    assert base64.b64decode(load_link(download_server + path, max_bytes=10)) == b"abcdefghij"
    with pytest.raises(ValueError, match="max_bytes"):
        load_link(download_server + path, max_bytes=9)


@pytest.mark.parametrize("path,error", [("/truncated", OSError), ("/stall", TimeoutError)])
def test_failed_http_transfer_preserves_existing_file(download_server, tmp_path, path, error):
    destination = tmp_path / "media"
    destination.write_bytes(b"original")
    with pytest.raises(error):
        save_link(download_server + path, str(destination), timeout=0.05)
    assert destination.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [destination]


def test_real_websocket_events_headers_remote_close_and_explicit_reconnect():
    async def scenario():
        requests, received = [], []

        async def peer(socket):
            requests.append(socket.request)
            received.append(json.loads(await socket.recv()))
            await socket.send(json.dumps({"type": "session.updated"}))
            await socket.close()

        async with serve(peer, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            session = RealtimeSession(f"ws://127.0.0.1:{port}/realtime?keep=yes", "local-model", api_key="local-test-key")
            try:
                for _ in range(2):
                    await session.connect()
                    await session.update_session(instructions="Test", output_modalities=["text"])
                    assert (await session.receive_event())["type"] == "session.updated"
                    with pytest.raises(ConnectionClosed):
                        await session.receive_event()
                    assert not session.connected
            finally:
                await session.close()
        assert len(requests) == 2
        for request in requests:
            assert request.headers["Authorization"] == "Bearer local-test-key"
            assert parse_qs(urlsplit(request.path).query) == {"keep": ["yes"], "model": ["local-model"]}
        assert all(event["type"] == "session.update" for event in received)

    asyncio.run(asyncio.wait_for(scenario(), 5))
