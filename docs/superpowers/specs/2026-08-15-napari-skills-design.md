# napari agent skills for multiplexed imaging — design

Date: 2026-08-15
Status: implemented

## Problem

Viewing CODEX / multiplexed slides is a daily task with no good agent support. The producing side is covered — segmentation, clustering and annotation pipelines exist (`SpatialOmicsAgentSkill`) — but nothing helps with the consuming side: open a 50 GB slide, decide which of 45 markers to put on screen, overlay results, draw annotations. A survey of the field found napari agent tooling (`napari-mcp`, Omega) and spatial-omics viewers (`napari-spatialdata`), but nothing that combines multiplexed imaging, napari and agent skills.

## Layering

Two layers, with a rule that decides which one a new file belongs to.

- `src/wapari/` — the library: readers, converters, panel knowledge.
- `skills/<skill-name>/SKILL.md` — the skills: session handling, dialogue, decisions.

The rule, stated in `README.md` and `CLAUDE.md`:

1. Would this still exist if there were no agent? Yes → package. No → skill.
2. Does it return a value, or maintain a session? Value → package. Session → skill.

Consequence for the pieces that exist today: `convert.py`, `tiff.py`, `color.py` and the marker table are package; the napari command bridge is a skill script, because it exists only to let an agent drive a GUI and because a library other people install should not ship a socket server that executes what it receives.

## Skills

Three skills with distinct roles. They are not siblings: skill 1 is infrastructure the other two depend on, skill 2 is the interactive flagship, skill 3 is a batch utility that needs no viewer.

### 1. `opening-napari-session`

Guarantees an agent-controllable napari session, by **always starting its own viewer through the bridge**.

`scripts/napari_server.py` plus `scripts/nsend.py` is a napari process with a socket server whose Qt timer drains queued code on the main thread, driven by any agent that can run a shell command. That includes agents with no MCP support at all — Codex reads `AGENTS.md` and runs CLIs, the same vendor-neutral stance `SpatialOmicsAgentSkill` takes.

Attaching to a napari window the user opened by hand is **deliberately out of scope**. It is the one thing the bridge cannot do and an in-process plugin can, and dropping it is what keeps this skill to a single code path with no mechanism to choose between. The user simply gets a second window, owned by the agent, which they can still interact with directly.

`napari-mcp` remains registered and is a fine way to drive that same viewer when a session exposes it, with better-typed arguments than a code channel. It is an alternative, never a prerequisite.

The bridge executes code it receives, so it listens on a **Unix domain socket with 0600 permissions** in a user-private runtime directory, not on a TCP port. There is no port for another local process to connect to, and access is enforced by the filesystem.

Its cost is deliberately small: it targets napari's most stable public surface (`Viewer`, `napari.run()`), qtpy's `QTimer` and the standard library, and touches no napari internals, so version churn should not reach it.

Four decisions came out of building it, each answering a way it failed in review:

- **A lock file, not a liveness probe, settles ownership.** Between `bind()` and `listen()` a socket exists that refuses connections, so two servers starting together both find the path free; measured, 11 of 60 concurrent pairs both bound. An exclusive `flock` decides it, and process death releases it, which is also what makes a crashed run's socket replaceable.
- **Refusal happens before napari is imported.** Otherwise an occupied path costs the user a window that opens and dies 30 seconds later.
- **Exit status distinguishes two failures that need different reactions.** 1 means the code ran and raised, so read the traceback; 2 means the call never completed, so look at the session. Anything that returns neither a result nor an error is the second kind.
- **Whatever the code prints is captured and returned.** Callers write `print()` by reflex; without capture the output went to the server's terminal and vanished, which is the silent-failure shape this repository exists to document.

### 2. `viewing-multiplex-image`

The interactive skill. Purpose first, markers second.

1. Open the image lazily (pyramid levels as a multiscale list; never read a full plane).
2. Show one channel only — DAPI, or channel 0 — so the user sees tissue immediately.
3. Report the panel grouped by **purpose**, not by biology alone:
   - **Segmentation**: DAPI plus membrane / boundary markers.
   - **Annotation**: lineage-specific markers that identify cell types.
   - **Other**: functional and state markers.
4. Ask what the user is doing — segmentation, annotation, or just looking — and add the markers for that purpose.

Classification is **table plus judgement**, implemented in `wapari.markers`:

- Known markers come from a curated table in the package, so the same panel classifies identically every time. A marker carries a set of roles rather than one category, because PanCK both draws epithelial boundaries and identifies epithelium, and a single category would have to discard one of those.
- Unknown markers are classified by the agent against the three criteria above, **proposed to the user, and confirmed before use**.
- Confirmed answers are offered back to the table, which therefore improves with use.

The seed table covers the 45-marker panel of the slide this was built against with nothing left unclassified.

The segmentation group maps directly onto the nuclear + boundary marker combo that `CODEXSegmentationSkill` expects, so the two skill sets interoperate without a new format.

### 3. `converting-image-to-ome-zarr`

Batch utility, no viewer. The implementation (`wapari.convert.qptiff_to_ome_zarr`) is complete: streaming copy of every pyramid level, bounded memory, channel names, pixel size, raw per-page XML and auxiliary series preserved, output verified pixel-by-pixel.

`SKILL.md` covers only the judgement:

- Convert when the data will be read repeatedly and randomly, shared, or cropped to disk.
- Do not convert to look at a slide once; open the qptiff lazily instead.
- Always verify; silent corruption is the known failure mode here.

## Shared knowledge

Traps are documented once, in `skills/opening-napari-session/references/gotchas.md` — the foundation skill, which the others already depend on — and referenced by the rest. Claude Code discovers only top-level `SKILL.md` files, so a `references/` subdirectory is not mistaken for a skill. Contents so far:

- Offscreen screenshots are black on macOS; the window must be shown and events processed.
- Shapes and Points layers have no undo; export annotations often.
- `S` scales a whole shape, `D` edits one vertex.
- A synthetic qptiff fixture needs `software="PerkinElmer-QPI"` and `metadata=None`, and a real qptiff pyramid is flat sequential pages.
- A pyramid level's zarr store is a Group when multi-level and an Array when single-level.

## Out of scope for now

- Overlaying segmentation masks and cell types (the next skills).
- Drawing annotations and exporting QuPath-compatible geojson — the planned differentiator, but it depends on the crop / mask work below.
- `crop_by_mask`: crop by polygon with a fill value, or by bounding box. Designed and approved, deferred behind this work.
- CosMx and OME-TIFF conversion inputs; the skill name already allows them.

## Later: GUI widgets

napari lacks conveniences QuPath users take for granted — hide or show every layer at once, turn the current shape into a mask, crop to it. Raised 2026-08-15, to be designed separately.

The boundary rule already answers where they go. A "hide all layers" button is useful to someone clicking it by hand, so it is package, not skill. Since wapari already depends on napari and PyQt5, declaring npe2 entry points costs nothing extra, and `pip install wapari` then ships the widgets with the library — no separate plugin distribution, no version to keep in step, no install instructions for an agent to recite.

Two tiers, split by lifetime rather than by technology:

- **Durable widgets** live in the package and are declared as npe2 contributions. `magicgui` keeps each one to roughly ten lines.
- **Ad-hoc widgets** are injected at runtime by an agent through `viewer.window.add_dock_widget` for one task and disappear with the session. These stay out of the package by the same rule that keeps the bridge out of it.

Publishing to napari hub is a separate decision, worth making only if the widgets are meant for people outside this lab.
