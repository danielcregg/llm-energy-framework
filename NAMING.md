# Renaming Strategy

## Current names

| Purpose | Name |
|---------|------|
| Display / brand name | **JouleBench** |
| Repository / directory | **llm-energy-framework** |

These are intentionally kept separate so a brand rename is purely cosmetic.

## Files that use the display name

Update these if the brand name changes:

- `index.html` -- title tag, nav brand text, meta description
- `README.md` -- heading, project description, publication title
- `PLAIN_ENGLISH_SUMMARY.md` -- title and 2-3 body mentions

`paper/main.tex` does not currently use the brand name but would need
updating if the paper adopts it as a title.

## Files that use the repo name

Only update these if the GitHub repository itself is renamed:

- GitHub URLs in `index.html` and `README.md`
- `git clone` command in `README.md`
- `CLAUDE.md` references the repo name in build instructions

## Files to leave alone

These are historical records and should not be edited on rename:

- `slurm_logs/` -- historical job output
- `scripts/*.sbatch` -- SLURM job names, not the brand
- `results/` -- JSON reports and CSV data
- `prior_work/` -- reference data from the earlier project

## Quick rename command

Find every instance of the current brand name:

```bash
grep -rn "JouleBench" --include="*.html" --include="*.md" --include="*.tex"
```

Find every instance of the repo name:

```bash
grep -rn "llm-energy-framework" --include="*.html" --include="*.md" --include="*.tex"
```

## Tip

Keep the display name decoupled from the repo/directory name.
A brand rename then requires editing only 3 files and breaks no paths,
imports, or URLs.
