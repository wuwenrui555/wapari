# CLAUDE.md

## Package or skill?

Before adding a file, decide which layer it belongs to. Two tests:

1. **Would this still exist if there were no agent?** Yes → `src/wapari/`. No → `skills/`.
2. **Does it return a value, or does it maintain a session?** A value → `src/wapari/`. A session → `skills/`.

Both must agree. When they disagree, the work is probably two things that should be split. See "Where does a new file go?" in `README.md` for the worked examples.

## Conventions

- Python 3.12+. Dependencies and virtualenv are managed with `uv`.
- Follow the git-flow and TDD workflows the user's global skills define.
- All code, comments, docstrings and documentation are written in English.
- Markdown uses soft wrapping: one line per paragraph, no manual line breaks.
- Verify writes: any conversion that produces a new image file compares pixels against the source before reporting success. Silent corruption in this ecosystem is common and never raises.
