# napari agent skills for multiplexed imaging — design

Date: 2026-08-15
Status: approved, not yet implemented

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

Guarantees an agent-controllable napari session and documents the two ways to get one.

- **Preferred: `napari-mcp`.** Maintained, standard, 16 tools, and its napari plugin can attach to a viewer the user opened by hand.
- **Fallback: the command bridge.** MCP servers load only at session start, so a session that began before registration — or in another directory — has no napari tools. The bridge covers that gap: a napari process with a socket server whose Qt timer drains queued code on the main thread.
- Decision rule in `SKILL.md`: check whether MCP napari tools are present; if not, start the bridge and tell the user why.

Scripts: `scripts/napari_server.py`, `scripts/nsend.py` (both validated against a 53 GB qptiff on 2026-08-15).

### 2. `viewing-multiplex-image`

The interactive skill. Purpose first, markers second.

1. Open the image lazily (pyramid levels as a multiscale list; never read a full plane).
2. Show one channel only — DAPI, or channel 0 — so the user sees tissue immediately.
3. Report the panel grouped by **purpose**, not by biology alone:
   - **Segmentation**: DAPI plus membrane / boundary markers.
   - **Annotation**: lineage-specific markers that identify cell types.
   - **Other**: functional and state markers.
4. Ask what the user is doing — segmentation, annotation, or just looking — and add the markers for that purpose.

Classification is **table plus judgement**:

- Known markers come from a curated table in the package, so the same panel classifies identically every time.
- Unknown markers are classified by the agent against the three criteria above, **proposed to the user, and confirmed before use**.
- Confirmed answers are offered back to the table, which therefore improves with use.

The segmentation group maps directly onto the nuclear + boundary marker combo that `CODEXSegmentationSkill` expects, so the two skill sets interoperate without a new format.

### 3. `converting-image-to-ome-zarr`

Batch utility, no viewer. The implementation (`wapari.convert.qptiff_to_ome_zarr`) is complete: streaming copy of every pyramid level, bounded memory, channel names, pixel size, raw per-page XML and auxiliary series preserved, output verified pixel-by-pixel.

`SKILL.md` covers only the judgement:

- Convert when the data will be read repeatedly and randomly, shared, or cropped to disk.
- Do not convert to look at a slide once; open the qptiff lazily instead.
- Always verify; silent corruption is the known failure mode here.

## Shared knowledge

Traps are documented once, in `skills/viewing-multiplex-image/references/gotchas.md`, and referenced by the other skills. Claude Code discovers only top-level `SKILL.md` files, so a `references/` subdirectory is not mistaken for a skill. Contents so far:

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
