"""Tests for the napari command bridge server.

The socket and execution machinery is tested without a GUI: the tests
drive ``drain()`` themselves, the way the Qt timer does in the real
script.
"""

import importlib.util
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

import pytest

SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "skills"
    / "opening-napari-session"
    / "scripts"
    / "napari_server.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("napari_server", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bridge = _load_module()


@pytest.fixture
def sock_dir():
    """A short-pathed directory.

    ``tmp_path`` is too long for AF_UNIX, whose paths are capped near 104
    bytes; that limit is the subject of its own test below.
    """
    path = pathlib.Path(tempfile.mkdtemp(dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def server(sock_dir):
    """A running server whose queue is drained by a background thread."""
    srv = bridge.CommandServer({"answer": 42}, sock_dir / "napari.sock")
    srv.start()
    stop = threading.Event()

    def pump():
        while not stop.is_set():
            srv.drain()
            time.sleep(0.005)

    pump_thread = threading.Thread(target=pump, daemon=True)
    pump_thread.start()
    yield srv
    stop.set()
    pump_thread.join(timeout=2)
    srv.stop()


def send(srv, code: str) -> dict:
    conn = socket.socket(socket.AF_UNIX)
    conn.connect(str(srv.socket_path))
    conn.sendall(code.encode() + b"\x00")
    buf = b""
    while not buf.endswith(b"\x00"):
        chunk = conn.recv(1 << 16)
        if not chunk:
            break
        buf += chunk
    conn.close()
    return json.loads(buf[:-1])


def test_expression_returns_its_repr(server):
    reply = send(server, "answer + 1")
    assert reply["ok"] is True
    assert reply["result"] == "43"


def test_statement_returns_ok_and_persists_in_namespace(server):
    assert send(server, "greeting = 'hi'")["ok"] is True
    assert send(server, "greeting")["result"] == "'hi'"


def test_multiline_code_is_executed(server):
    reply = send(server, "total = 0\nfor i in range(4):\n    total += i\n")
    assert reply["ok"] is True
    assert send(server, "total")["result"] == "6"


def test_stdout_is_captured_and_returned(server):
    """Callers write print(); without this the output vanishes server-side."""
    reply = send(server, "print('visible')")
    assert reply["ok"] is True
    assert "visible" in reply["output"]


def test_stdout_is_captured_for_statements_too(server):
    reply = send(server, "for i in range(2):\n    print('line', i)\n")
    assert reply["output"].splitlines() == ["line 0", "line 1"]


def test_stdout_capture_does_not_replace_the_result(server):
    reply = send(server, "print('noise') or answer")
    assert reply["result"] == "42"
    assert "noise" in reply["output"]


def test_error_returns_traceback_and_server_survives(server):
    reply = send(server, "1 / 0")
    assert reply["ok"] is False
    assert "ZeroDivisionError" in reply["error"]
    assert send(server, "answer")["result"] == "42"  # still serving


def test_socket_file_is_private_to_the_user(server):
    mode = os.stat(server.socket_path).st_mode & 0o777
    assert mode == 0o600


def test_socket_directory_the_server_creates_is_private(sock_dir):
    """A directory the server makes must not be traversable by others."""
    srv = bridge.CommandServer({}, sock_dir / "made-by-server" / "napari.sock")
    srv.start()
    try:
        assert os.stat(srv.socket_path.parent).st_mode & 0o777 == 0o700
    finally:
        srv.stop()


def test_socket_is_private_even_in_a_shared_directory(sock_dir):
    """mkdir(mode=) does nothing to a directory that already exists, and
    a path like /tmp/wapari always exists already, so the socket's own
    mode is the only thing protecting it there."""
    shared = sock_dir / "world-readable"
    shared.mkdir(mode=0o755)
    srv = bridge.CommandServer({}, shared / "napari.sock")
    srv.start()
    try:
        assert os.stat(srv.socket_path).st_mode & 0o777 == 0o600
    finally:
        srv.stop()


def test_socket_is_a_unix_socket_not_a_tcp_port(server):
    assert server.socket_path.is_socket()


def test_a_live_server_is_not_silently_displaced(server):
    """Unlinking a live server's socket leaves two viewers running with
    only one reachable, and the user cannot tell which is which."""
    second = bridge.CommandServer({}, server.socket_path)
    with pytest.raises(RuntimeError, match="already serving"):
        second.start()
    assert send(server, "answer")["result"] == "42"  # first one untouched


def test_only_one_of_many_simultaneous_starts_wins(sock_dir):
    """Probing for a listener cannot settle this: between bind() and
    listen() a socket exists that refuses connections, so two servers
    starting together can both believe the path is free."""
    path = sock_dir / "contested.sock"
    servers = [bridge.CommandServer({}, path) for _ in range(8)]
    started, refused = [], 0
    barrier = threading.Barrier(len(servers))

    def race(srv):
        nonlocal refused
        barrier.wait()
        try:
            srv.start()
            started.append(srv)
        except RuntimeError:
            refused += 1

    threads = [threading.Thread(target=race, args=(s,)) for s in servers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    try:
        assert len(started) == 1
        assert refused == len(servers) - 1
    finally:
        for srv in started:
            srv.stop()


def test_a_refused_start_does_not_remove_the_winners_socket(sock_dir):
    path = sock_dir / "contested.sock"
    winner = bridge.CommandServer({}, path)
    winner.start()
    loser = bridge.CommandServer({}, path)
    try:
        with pytest.raises(RuntimeError):
            loser.start()
        loser.stop()
        assert path.is_socket()
    finally:
        winner.stop()


def test_main_refuses_an_occupied_socket_before_loading_napari(server):
    """A window that flashes open and dies is worse than a refusal, and
    importing napari is what takes the 30-60 seconds."""
    started = time.monotonic()
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--socket", str(server.socket_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    elapsed = time.monotonic() - started

    assert done.returncode == 2
    assert "already serving" in done.stderr
    assert "Traceback" not in done.stderr  # a refusal, not a crash
    # Importing napari takes seconds; refusing must not pay that cost.
    assert elapsed < 5


def test_availability_check_passes_on_a_free_path(sock_dir):
    bridge.ensure_available(sock_dir / "free.sock")  # must not raise


def test_a_displaced_server_does_not_delete_the_live_socket(server):
    second = bridge.CommandServer({}, server.socket_path)
    with pytest.raises(RuntimeError):
        second.start()
    second.stop()
    assert server.socket_path.is_socket()


def test_stale_socket_file_is_replaced(sock_dir):
    path = sock_dir / "napari.sock"
    path.write_text("leftover from a crashed run")
    srv = bridge.CommandServer({}, path)
    srv.start()
    try:
        assert path.is_socket()
    finally:
        srv.stop()


def test_stop_removes_the_socket_file(sock_dir):
    srv = bridge.CommandServer({}, sock_dir / "napari.sock")
    srv.start()
    srv.stop()
    assert not (sock_dir / "napari.sock").exists()


def test_a_dead_servers_socket_is_replaced(sock_dir):
    """A crashed run leaves a socket file that nothing is listening on.

    Process death closes every descriptor, releasing the lock along with
    the socket, so the simulation has to drop both.
    """
    first = bridge.CommandServer({}, sock_dir / "napari.sock")
    first.start()
    first._sock.close()
    os.close(first._lock_fd)
    first._lock_fd = None
    second = bridge.CommandServer({}, sock_dir / "napari.sock")
    second.start()
    try:
        assert second.socket_path.is_socket()
    finally:
        second.stop()


def test_syntax_error_reports_one_traceback(server):
    reply = send(server, "def oops(:")
    assert reply["ok"] is False
    assert reply["error"].count("SyntaxError") == 1
    assert "During handling of the above exception" not in reply["error"]


def test_overlong_socket_path_is_rejected_with_a_clear_message(tmp_path):
    """AF_UNIX caps the path near 104 bytes; the OSError is unhelpful."""
    deep = tmp_path / ("d" * 80) / ("e" * 80)
    srv = bridge.CommandServer({}, deep / "napari.sock")
    with pytest.raises(ValueError, match="socket path is too long"):
        srv.start()


def test_default_socket_path_is_in_a_user_private_directory():
    path = bridge.default_socket_path()
    assert path.name.endswith(".sock")
    assert str(path).startswith(str(pathlib.Path.home())) or str(path).startswith(
        os.environ.get("XDG_RUNTIME_DIR", "/run/user")
    )
