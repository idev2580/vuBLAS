# Mark pre-rename names in prompt history as former names

## What to implement

- Keep `socl` / `soclBLAS` spellings untouched inside historical
  `prompts/dev-prompt-*.md` records (no rewriting of history).
- Insert one `IMPORTANT` note directly below the first-line title of every
  history file that mentions `socl` (16 files: dev-prompt-0, 1, 2, 7, 8, 13,
  14, 17, 20, 22, 23, 25, 26, 28, 29, 30), stating that `socl` is the former
  name of `vucol` and `soclBLAS` is the former name of `vuBLAS`, so future
  readers do not confuse old and new names.
- Exclude `dev-prompt-31.md` (the rename record itself, already documents the
  mapping) and this file from the note insertion.

## How to implement

- Confirm the target list with case-insensitive `grep -li "socl" prompts/*.md`
  and check each file starts with a `# Title` first line with no pre-existing
  `IMPORTANT` note.
- Insert the blockquote note as line 2 of each target file, leaving all other
  content byte-identical; verify afterwards that every target contains the
  note exactly once and no other `prompts/` file was modified.
- `prompts/` edits need no extra approval (the AGENTS.md exception); no
  compile or execution in the agent environment, no new external libraries.
