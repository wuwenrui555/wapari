# WapaRi

Utilities and agent skills for viewing multiplexed imaging data (CODEX / Akoya qptiff, OME-TIFF, OME-Zarr) in [napari](https://napari.org).

The repository has two layers:

- **`src/wapari/`** — a plain Python library. Readers, converters and panel knowledge you would use from a notebook, with or without an agent.
- **`skills/`** — [Agent Skills](https://agentskills.dev): one directory per skill, each with a `SKILL.md`. These drive a live napari session, ask you what you want to see, and decide what to put on screen.

## What you can ask for

Skills match on intent, not on wording, and the language you ask in does not matter.

**Open a viewer you can talk to.**

> Open a napari window I can drive by talking to you.

Starts a napari session the agent can control, and tells you which mechanism it used. From here every request below happens in that same window, and you can also reach in and use the GUI yourself at any time.

**Open a slide.**

> Open `~/data/20251001_Xenium097.qptiff`.

Loads the pyramid lazily, so a 53 GB slide appears in seconds and nothing is read into memory until you look at it. Only one channel goes up at first, so you see tissue rather than a wall of colour. The agent then reports the panel grouped by what the markers are for, and asks what you want to do.

**Put up markers for a purpose, not by name.**

> I'm going to segment this — put up what I need.
>
> Show me the T cell markers.

Segmentation means DAPI plus membrane markers; annotation means the lineage-specific ones. Ask for specific markers instead and it adds exactly those.

**Ask what is in a file.**

> What channels are in this qptiff?
>
> Is there a macrophage marker in this panel?

Answered from the file's own metadata, without opening a viewer.

**Convert for repeated access.**

> Convert this qptiff to OME-Zarr.

Streams every pyramid level into a chunked OME-Zarr, preserves channel names, pixel size and the original vendor metadata, and verifies the result pixel-by-pixel before saying it worked. Worth doing when you will read the data many times; not worth doing to look at a slide once.

## Where does a new file go?

Two tests settle it. Both must agree; when they disagree, the work is probably two things that should be split.

1. **Would this still exist if there were no agent?** Yes → package. No → skill.
2. **Does it return a value, or does it maintain a session?** A value → package. A session → skill.

| Thing | Goes in | Because |
| --- | --- | --- |
| Format conversion, readers, writers | `src/wapari/` | Useful in a notebook on its own |
| Marker classification table + lookup | `src/wapari/` | Panel knowledge, also needed when writing analysis scripts |
| The napari command bridge | `skills/*/scripts/` | Exists only so an agent can drive a GUI; it is a process, not a function |
| Dialogue flow, prompts, decision rules | `SKILL.md` | Judgement, not code |
| Traps and workarounds found in practice | `skills/*/references/` | Experience, not API |

The bridge is the clearest case: it listens on a socket and executes what it receives. That belongs to an agent workflow, not to a library other people install.

## Installation

Requires Python 3.12 or newer (`napari-ome-zarr` and `ome-zarr` only support zarr 3 from versions 0.10.0 / 0.18.0, and those need >3.11).

```bash
git clone https://github.com/wuwenrui555/wapari.git
cd wapari
uv sync
```

## Making the skills discoverable

An agent finds a skill at `<skills-dir>/<skill-name>/SKILL.md`, where the directory name matches the `name:` in the frontmatter. Link them rather than copying, so `git pull` updates them in place:

```bash
ln -s "$PWD/skills/opening-napari-session" ~/.claude/skills/opening-napari-session
```

Restart the agent session afterwards, then describe what you want; the matching skill loads itself. Agents without a skills mechanism can be pointed at the `SKILL.md` paths directly — they are plain Markdown and assume no particular runtime.

For the richer napari control path, register the MCP server once per project (already done in this repository's `.mcp.json`) and start a new session, since MCP servers are loaded only at session start:

```bash
claude mcp add napari --scope project -- uv run --project "$PWD" napari-mcp run
```

## References

- [napari training course](https://github.com/sofroniewn/napari-training-course/tree/master/lessons)
- [OME-NGFF specification](https://ngff.openmicroscopy.org/latest/)
