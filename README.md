# WapaRi

Utilities and agent skills for viewing multiplexed imaging data (CODEX / Akoya qptiff, OME-TIFF, OME-Zarr) in [napari](https://napari.org).

The repository has two layers:

- **`src/wapari/`** — a plain Python library. Readers, converters and panel knowledge you would use from a notebook, with or without an agent.
- **`skills/`** — [Agent Skills](https://agentskills.dev): one directory per skill, each with a `SKILL.md`. These drive a live napari session, ask you what you want to see, and decide what to put on screen.

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

## References

- [napari training course](https://github.com/sofroniewn/napari-training-course/tree/master/lessons)
- [OME-NGFF specification](https://ngff.openmicroscopy.org/latest/)
