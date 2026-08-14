"""Send Python to a running napari bridge and print what it returns.

    python nsend.py "viewer.layers"           # code as an argument
    python nsend.py < script.py               # or on stdin

Whatever the code prints is written first, then its result. Exit status
separates the two failures that need different reactions: **1** means the
code ran and raised, so read the traceback and fix the code; **2** means
the call did not complete — no server, an unusable socket path, a viewer
too busy to answer, or a connection lost part way — so check the session
rather than the code.
"""

import argparse
import functools
import json
import pathlib
import socket
import sys

TERMINATOR = b"\x00"


@functools.cache
def _server_module():
    """Load napari_server.py once, for its path helpers.

    ``skills/`` is not an importable package on purpose, and the server
    keeps its heavy imports inside ``main()``, so this stays cheap.
    """
    import importlib.util

    server = pathlib.Path(__file__).with_name("napari_server.py")
    spec = importlib.util.spec_from_file_location("napari_server", server)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code", nargs="?", help="code to run; omit to read stdin")
    parser.add_argument("--socket", default=None, help="socket the server listens on")
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="seconds to wait for a reply before giving up (default 600)",
    )
    args = parser.parse_args()

    server = _server_module()
    code = args.code if args.code is not None else sys.stdin.read()
    socket_path = (
        pathlib.Path(args.socket) if args.socket else server.default_socket_path()
    )
    max_path_bytes = server.MAX_SOCKET_PATH_BYTES

    # NUL ends a message on the wire, so it can never travel inside one.
    # Python source cannot contain it either, so this is always a mistake.
    if "\x00" in code:
        print("the code contains a null byte, which cannot be sent", file=sys.stderr)
        return 2
    if len(str(socket_path).encode()) > max_path_bytes:
        print(
            f"the socket path is too long ({socket_path}). The OS caps AF_UNIX "
            f"paths near {max_path_bytes} bytes, so no server can listen "
            "there; use a shorter --socket.",
            file=sys.stderr,
        )
        return 2

    try:
        conn = socket.socket(socket.AF_UNIX)
        conn.settimeout(args.timeout)
        conn.connect(str(socket_path))
    except OSError as error:
        print(
            f"no napari server at {socket_path} ({error}). Start one with "
            "napari_server.py, or pass --socket.",
            file=sys.stderr,
        )
        return 2

    with conn:
        conn.sendall(code.encode() + TERMINATOR)
        buf = b""
        while not buf.endswith(TERMINATOR):
            try:
                chunk = conn.recv(1 << 16)
            except TimeoutError:
                print(
                    f"the viewer did not answer within {args.timeout:g}s. Code "
                    "runs on the GUI thread, so an earlier blocking call may "
                    "still be holding it.",
                    file=sys.stderr,
                )
                return 2
            if not chunk:
                print("the napari server closed the connection", file=sys.stderr)
                return 2
            buf += chunk

    reply = json.loads(buf[: -len(TERMINATOR)])
    # Whatever the code printed comes first, so it reads like a session.
    printed = reply.get("output", "")
    if printed:
        sys.stdout.write(printed if printed.endswith("\n") else printed + "\n")
    if reply.get("ok"):
        print(reply["result"])
        return 0
    if "error" not in reply:
        # Neither a result nor a traceback: the viewer accepted the code
        # and never finished it, which is not the same as code that ran
        # and raised.
        print(
            "the viewer accepted the code but did not complete it",
            file=sys.stderr,
        )
        return 2
    print(reply["error"], file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
