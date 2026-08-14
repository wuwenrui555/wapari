"""Tests for the bridge client, nsend.py.

The client is what an agent actually runs, so its contract is a shell
contract: code in, result on stdout, exit status that tells success from
failure.
"""

import importlib.util
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

import pytest

SCRIPTS = (
    pathlib.Path(__file__).resolve().parents[1]
    / "skills"
    / "opening-napari-session"
    / "scripts"
)
CLIENT = SCRIPTS / "nsend.py"


def _load_server_module():
    spec = importlib.util.spec_from_file_location(
        "napari_server", SCRIPTS / "napari_server.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bridge = _load_server_module()


@pytest.fixture
def short_dir():
    """AF_UNIX paths are capped near 104 bytes; tmp_path is too long."""
    path = pathlib.Path(tempfile.mkdtemp(dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def server():
    """A pumped server on a short socket path."""
    directory = pathlib.Path(tempfile.mkdtemp(dir="/tmp"))
    srv = bridge.CommandServer({"answer": 42}, directory / "napari.sock")
    srv.start()
    stop = threading.Event()

    def pump():
        while not stop.is_set():
            srv.drain()
            time.sleep(0.005)

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    yield srv
    stop.set()
    thread.join(timeout=2)
    srv.stop()
    shutil.rmtree(directory, ignore_errors=True)


def run_client(server, code: str, from_stdin: bool = False):
    args = [sys.executable, str(CLIENT), "--socket", str(server.socket_path)]
    if from_stdin:
        return subprocess.run(args, input=code, capture_output=True, text=True)
    return subprocess.run([*args, code], capture_output=True, text=True)


def test_prints_the_result_of_an_expression(server):
    done = run_client(server, "answer + 1")
    assert done.returncode == 0
    assert done.stdout.strip() == "43"


def test_reads_code_from_stdin(server):
    done = run_client(server, "answer * 2", from_stdin=True)
    assert done.returncode == 0
    assert done.stdout.strip() == "84"


def test_printed_output_reaches_the_caller(server):
    done = run_client(server, "print('from inside napari')")
    assert done.returncode == 0
    assert "from inside napari" in done.stdout


def test_printed_output_appears_before_the_result(server):
    done = run_client(server, "print('side effect') or answer")
    assert done.stdout.index("side effect") < done.stdout.index("42")


def test_failure_exits_nonzero_and_reports_the_traceback(server):
    done = run_client(server, "1 / 0")
    assert done.returncode != 0
    assert "ZeroDivisionError" in done.stdout + done.stderr


def test_traceback_goes_to_stderr_not_stdout(server):
    done = run_client(server, "1 / 0")
    assert "ZeroDivisionError" in done.stderr
    assert "ZeroDivisionError" not in done.stdout


def test_code_raising_exits_1_and_a_missing_server_exits_2(server, short_dir):
    """The two failures need different reactions: fix the code, or start
    a viewer. A single non-zero status cannot tell them apart."""
    assert run_client(server, "1 / 0").returncode == 1
    absent = subprocess.run(
        [sys.executable, str(CLIENT), "--socket", str(short_dir / "absent.sock"), "1"],
        capture_output=True,
        text=True,
    )
    assert absent.returncode == 2


def test_null_byte_in_code_is_rejected_before_sending(server):
    """NUL terminates the wire format, so code containing one would be
    truncated at a chunk boundary rather than failing predictably. It
    arrives on stdin, since argv cannot carry a NUL at all."""
    done = run_client(server, "answer  # \x00 trailing", from_stdin=True)
    assert done.returncode == 2
    assert "null byte" in done.stderr.lower()


def test_overlong_socket_path_is_not_reported_as_a_missing_server(tmp_path):
    deep = tmp_path / ("d" * 80) / ("e" * 80) / "napari.sock"
    done = subprocess.run(
        [sys.executable, str(CLIENT), "--socket", str(deep), "1"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 2
    assert "too long" in done.stderr.lower()
    assert "start one with" not in done.stderr.lower()  # wrong remedy


def test_an_incomplete_call_is_not_reported_as_raised_code(short_dir):
    """Exit 1 means the code ran and raised. A reply that carries neither
    a result nor an error means it never finished, which is exit 2."""
    srv = bridge.CommandServer({}, short_dir / "empty.sock")
    srv.start()
    stop = threading.Event()

    def pump_empty_replies():
        # Mimic the server's own timeout path: the request is dropped and
        # an empty reply goes back.
        while not stop.is_set():
            while not srv._queue.empty():
                _, reply, done = srv._queue.get()
                done.set()  # reply left empty on purpose
            time.sleep(0.005)

    thread = threading.Thread(target=pump_empty_replies, daemon=True)
    thread.start()
    try:
        done = subprocess.run(
            [sys.executable, str(CLIENT), "--socket", str(srv.socket_path), "1 + 1"],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 2
        assert "did not complete" in done.stderr.lower()
    finally:
        stop.set()
        thread.join(timeout=2)
        srv.stop()


def canned_reply_server(short_dir, payload: bytes):
    """A socket that answers every request with a fixed payload."""
    path = short_dir / "canned.sock"
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(path))
    listener.listen(4)

    def serve():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            conn.recv(1 << 16)
            conn.sendall(payload)
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    return path, listener


def test_a_reply_that_is_not_json_exits_2_not_1(short_dir):
    """Exit 1 tells the agent to go read a traceback from its own code.
    A malformed reply is not that, and --socket can point anywhere."""
    path, listener = canned_reply_server(short_dir, b"not json at all\x00")
    try:
        done = subprocess.run(
            [sys.executable, str(CLIENT), "--socket", str(path), "1"],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 2
        assert "Traceback" not in done.stderr
    finally:
        listener.close()


def test_a_reply_missing_its_result_exits_2_not_1(short_dir):
    path, listener = canned_reply_server(short_dir, b'{"ok": true}\x00')
    try:
        done = subprocess.run(
            [sys.executable, str(CLIENT), "--socket", str(path), "1"],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 2
        assert "Traceback" not in done.stderr
    finally:
        listener.close()


def test_a_wedged_viewer_times_out_instead_of_hanging(short_dir):
    """Code runs on the GUI thread, so a blocking call stops every later
    one. The caller must give up rather than wait indefinitely."""
    srv = bridge.CommandServer({}, short_dir / "wedged.sock")
    srv.start()  # started but never drained: requests are accepted, never run
    try:
        done = subprocess.run(
            [
                sys.executable,
                str(CLIENT),
                "--socket",
                str(srv.socket_path),
                "--timeout",
                "1",
                "1 + 1",
            ],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 2
        assert "did not answer" in done.stderr.lower()
    finally:
        srv.stop()


def test_missing_server_exits_nonzero_with_a_readable_message(short_dir):
    done = subprocess.run(
        [sys.executable, str(CLIENT), "--socket", str(short_dir / "absent.sock"), "1"],
        capture_output=True,
        text=True,
    )
    assert done.returncode != 0
    assert "no napari server" in done.stderr.lower()
