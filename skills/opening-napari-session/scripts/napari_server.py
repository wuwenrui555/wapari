"""Run a napari viewer that executes Python sent over a Unix socket.

Qt owns the main thread, so the socket thread only queues code; a QTimer
on the main thread drains the queue and executes it. Requests block until
their result comes back, which keeps a caller's commands ordered.

The socket executes what it receives, so it is a Unix domain socket with
0600 permissions rather than a TCP port: there is no port for another
local process to reach, and access is enforced by the filesystem.

Run it:

    python napari_server.py [--socket PATH]

Then send code with ``nsend.py``.
"""

import argparse
import contextlib
import fcntl
import io
import json
import os
import pathlib
import queue
import socket
import sys
import threading
import time
import traceback

TERMINATOR = b"\x00"
SOURCE = "<nsend>"
RECV_TIMEOUT_SECONDS = 60.0

# AF_UNIX paths are capped by the OS (104 bytes on macOS, 108 on Linux).
# Binding past it raises a bare "AF_UNIX path too long".
MAX_SOCKET_PATH_BYTES = 100


def default_socket_path() -> pathlib.Path:
    """Return the socket path used when none is given."""
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    base = pathlib.Path(runtime_dir) if runtime_dir else pathlib.Path.home() / ".cache"
    return base / "wapari" / "napari.sock"


def is_serving(socket_path: str | pathlib.Path) -> bool:
    """Return whether something is accepting connections on this path."""
    probe = socket.socket(socket.AF_UNIX)
    probe.settimeout(1)
    try:
        probe.connect(str(socket_path))
    except OSError:
        return False
    else:
        return True
    finally:
        probe.close()


def ensure_available(socket_path: str | pathlib.Path) -> pathlib.Path:
    """Raise unless a server could be started on this path.

    Called before napari is imported, so an occupied socket costs a clear
    message rather than a window that opens and dies.
    """
    socket_path = pathlib.Path(socket_path)
    if len(str(socket_path).encode()) > MAX_SOCKET_PATH_BYTES:
        raise ValueError(
            f"socket path is too long ({socket_path}). The OS caps AF_UNIX "
            f"paths near {MAX_SOCKET_PATH_BYTES} bytes; pass a shorter "
            f"--socket, for example {default_socket_path()}."
        )
    if socket_path.exists() and is_serving(socket_path):
        raise RuntimeError(
            f"a napari server is already serving on {socket_path}. "
            "Use it, or stop it before starting another."
        )
    return socket_path


def _recv_until_terminator(conn: socket.socket) -> bytes | None:
    buf = b""
    while not buf.endswith(TERMINATOR):
        chunk = conn.recv(1 << 16)
        if not chunk:
            return None
        buf += chunk
    return buf[: -len(TERMINATOR)]


class CommandServer:
    """Accept code over a Unix socket and run it in ``namespace``.

    Execution happens in :meth:`drain`, which the caller runs on the
    thread that owns the GUI. Nothing is executed on the socket thread.
    """

    def __init__(
        self, namespace: dict, socket_path: str | pathlib.Path | None = None
    ) -> None:
        self.namespace = namespace
        self.socket_path = pathlib.Path(socket_path or default_socket_path())
        self._queue: queue.Queue = queue.Queue()
        self._sock: socket.socket | None = None
        self._stopping = threading.Event()
        self._owns_socket = False
        self._lock_fd: int | None = None

    def start(self) -> None:
        """Bind the socket and start accepting connections.

        Refuses to displace a server that already holds this path: two
        viewers sharing one reachable socket is worse than a failure.
        """
        ensure_available(self.socket_path)
        self.socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # The lock, not the probe, is what settles ownership. Between
        # bind() and listen() a socket exists that refuses connections,
        # so two servers starting together can both find the path free;
        # only one can hold the lock.
        self._take_lock()
        if self.socket_path.exists():
            self.socket_path.unlink()  # left behind by a crashed run
        self._sock = socket.socket(socket.AF_UNIX)
        # bind() applies the umask, so a shared parent directory would
        # otherwise expose the socket until the chmod below lands.
        previous_umask = os.umask(0o177)
        try:
            self._sock.bind(str(self.socket_path))
        finally:
            os.umask(previous_umask)
        self.socket_path.chmod(0o600)
        self._owns_socket = True
        self._sock.listen(8)
        threading.Thread(
            target=self._accept_loop, args=(self._sock,), daemon=True
        ).start()

    def _take_lock(self) -> None:
        """Claim exclusive ownership of this socket path, or refuse."""
        lock_path = self.socket_path.with_name(self.socket_path.name + ".lock")
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise RuntimeError(
                f"a napari server is already serving on {self.socket_path}. "
                "Use it, or stop it before starting another."
            ) from None
        self._lock_fd = fd

    def _accept_loop(self, sock: socket.socket) -> None:
        # Hold the socket locally: stop() clears the attribute, and closing
        # it is what breaks us out of accept().
        while not self._stopping.is_set():
            try:
                conn, _ = sock.accept()
            except OSError:
                if self._stopping.is_set() or sock.fileno() < 0:
                    return  # closed by stop()
                # Transient, e.g. too many open files. Pause before
                # retrying so a persistent one cannot spin a whole core.
                time.sleep(0.1)
                continue
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            # A connection that opens and never sends would otherwise
            # park this thread for the life of the viewer.
            conn.settimeout(RECV_TIMEOUT_SECONDS)
            try:
                code = _recv_until_terminator(conn)
            except TimeoutError:
                return
            if code is None:
                return
            reply: dict = {}
            done = threading.Event()
            self._queue.put((code.decode(), reply, done))
            done.wait(timeout=3600)
            conn.sendall(json.dumps(reply).encode() + TERMINATOR)
        finally:
            conn.close()

    def drain(self) -> None:
        """Execute every queued request. Call this from the GUI thread."""
        while not self._queue.empty():
            code, reply, done = self._queue.get()
            # Callers write print(); without capturing it here the output
            # would go to this process's terminal and never reach them.
            printed = io.StringIO()
            try:
                # Syntax is checked once, in "exec" mode, which accepts
                # both forms; a genuine syntax error therefore surfaces as
                # one traceback rather than two chained ones. Only then is
                # the code classified, and that attempt swallows its own
                # failure because a statement is not an error.
                compiled = compile(code, SOURCE, "exec")
                is_expression = False
                try:
                    compiled, is_expression = compile(code, SOURCE, "eval"), True
                except SyntaxError:
                    pass
                with contextlib.redirect_stdout(printed):
                    if is_expression:
                        # An expression reports its value; anything else
                        # runs for its side effects and reports "ok".
                        reply["result"] = repr(eval(compiled, self.namespace))
                    else:
                        exec(compiled, self.namespace)
                        reply["result"] = "ok"
                reply["ok"] = True
            except Exception:
                reply["ok"] = False
                reply["error"] = traceback.format_exc()
            reply["output"] = printed.getvalue()
            done.set()

    def stop(self) -> None:
        """Close the socket and remove the file, if this server made it."""
        self._stopping.set()
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._owns_socket:
            self.socket_path.unlink(missing_ok=True)
            self._owns_socket = False
        if self._lock_fd is not None:
            os.close(self._lock_fd)  # releases the flock
            self._lock_fd = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=None, help="socket path to listen on")
    parser.add_argument("--title", default="wapari agent viewer")
    args = parser.parse_args()

    # Fail before napari loads: an occupied socket should not cost the
    # user a window that opens and immediately dies.
    try:
        socket_path = ensure_available(args.socket or default_socket_path())
    except (RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    import napari
    from qtpy.QtCore import QTimer

    viewer = napari.Viewer(title=args.title)
    server = CommandServer({"viewer": viewer, "napari": napari}, socket_path)
    server.start()
    print(f"[napari-server] listening on {server.socket_path}", flush=True)

    timer = QTimer()
    timer.timeout.connect(server.drain)
    timer.start(50)
    try:
        napari.run()
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
