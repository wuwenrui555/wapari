---
name: opening-napari-session
description: Use when the user wants a napari window the agent can drive by conversation, or when any other skill needs a live viewer to add layers to. Triggers on "open napari", "打开napari", "给我开个窗口", "show me this image", or any request to look at imaging data interactively.
---

# Opening a napari session

Every interactive skill in this repository needs the same thing first: a napari viewer this agent can act on. This skill starts one and hands the other skills a way to reach it.

## One path: start your own viewer

**Always create the viewer with the bridge, and only ever drive one you created.** Do not try to reach a napari window the user opened by hand — that is the case that needs an in-process plugin, and refusing it keeps this skill to a single code path that works in any agent with a shell.

The consequence is worth stating to the user once: a window they opened themselves is theirs, and this skill will start a separate one it owns. They can still click, pan and draw in the one it started; both of you are looking at the same viewer.

If the session happens to expose `napari-mcp` tools, they are a fine way to drive that same viewer, and their typed arguments read better than a code channel. They are an alternative, not a prerequisite: MCP servers load only when a session starts, so their absence is normal and never a reason to stop.

## Starting the bridge

Sessions usually run in the user's data directory, not in this repository, so address both the scripts and the environment absolutely. `$SKILL` is this skill's own directory and `$REPO` its repository root:

```bash
SKILL=<this skill's directory>
REPO="${SKILL%/skills/*}"
run() { uv run --project "$REPO" python "$@"; }   # a function, not a
# variable: zsh does not word-split RUN="uv run …" the way bash does
```

The server holds the process for the life of the window, so **start it in the background with its output kept**, then poll for the socket. A cold start takes 30-60 seconds while napari builds its caches, which is not a failure:

```bash
SOCKET="${XDG_RUNTIME_DIR:-$HOME/.cache}/wapari/napari.sock"   # the client's default
LOG=/tmp/wapari-server.log

run "$SKILL/scripts/napari_server.py" --title "wapari viewer" > "$LOG" 2>&1 &
run - "$SOCKET" <<'PY'
import socket, sys, time
path = sys.argv[1]
for _ in range(90):
    probe = socket.socket(socket.AF_UNIX)
    try:
        probe.connect(path)
        print("ready")
        break
    except OSError:
        time.sleep(1)
    finally:
        probe.close()
else:
    print("napari did not come up")
PY
```

Read `$LOG` whenever the poll fails. An immediate "ready" means a viewer from an earlier turn is still running, which is the good case — reuse it rather than starting another. Either way, confirm what answered before adding layers to it:

```bash
run "$SKILL/scripts/nsend.py" "viewer.title"
```

Leave the socket path at its default (`$XDG_RUNTIME_DIR/wapari/napari.sock`, or `~/.cache/wapari/napari.sock`) so `nsend.py` finds it without a flag, and because that directory belongs to the user. `/tmp` works but is world-writable. Pass `--socket` only when the default would exceed the operating system's path limit, which both scripts report clearly.

Give each viewer a name with `--title`; it is what lets the user tell two windows apart.

A second server on a path that is already owned exits 2 with "already serving" instead of displacing the first, and does so before loading napari, so nothing flashes on screen. Reuse the running viewer instead.

## Sending code

```bash
run "$SKILL/scripts/nsend.py" "viewer.layers"
run "$SKILL/scripts/nsend.py" <<'PY'
import numpy as np
viewer.add_image(np.random.random((512, 512)), name="demo", colormap="magma")
PY
```

The namespace holds `viewer` and `napari`, and nothing else — import what you need. Anything assigned persists between calls, so state builds up across turns.

An expression returns its `repr`; anything else returns `ok`. Whatever the code prints comes back too, ahead of the result. On failure the traceback goes to stderr, and the exit status separates the two cases that need different reactions: **1** means the code ran and raised, so read the traceback; **2** means the call did not complete, so check the session — no server, an unusable path, or a viewer that never answered.

Code runs on the GUI thread, so anything slow or blocking freezes the window and every later call until it finishes. Keep each message short, and load large data lazily rather than reading it in one call. A reply that does not arrive within `--timeout` seconds (600 by default) gives up with exit 2 rather than waiting; raise it for a call that is genuinely long, and read it as "the viewer is wedged" otherwise.

## Confirming what the user sees

The user is looking at their own screen, so a claim that something is displayed should rest on a screenshot rather than on the absence of an error. The bridge returns text, so write the image to a file inside the viewer and read that file:

```bash
run "$SKILL/scripts/nsend.py" <<'PY'
import imageio.v3 as iio
from qtpy.QtWidgets import QApplication
for _ in range(12):
    QApplication.processEvents()
iio.imwrite("/tmp/wapari-shot.png", viewer.screenshot(canvas_only=True))
print("written")
PY
```

Then read `/tmp/wapari-shot.png`. An all-black screenshot usually means the canvas never drew, not that the data is empty; see `references/gotchas.md`, which every skill in this repository shares.

## Handing off

Other skills assume a session exists and add layers to it. When one asks for a viewer, confirm the current one still answers (`viewer.title` round-trips) before adding to it, and start a new session only if it does not.

To stop a viewer, close its window, or terminate the process; the socket file is removed on a clean exit and a leftover file from a crash is replaced automatically on the next start.
