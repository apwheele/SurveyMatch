# SurveyMatch

A method for identifying near-duplicate survey responses by modeling the
full distribution of pairwise match counts, rather than thresholding each
pair's match percentage against a fixed rate.

The paper is `paper.pdf` (also built as `paper.docx` and `paper.md`), built
from `paper.qmd` and `references.bib`. The reusable method is the
`surveymatch` Python package under `src/`.

## Reproducing

This project uses [uv](https://docs.astral.sh/uv/) for a reproducible Python
environment and [Quarto](https://quarto.org/) to render the paper.

```bash
uv sync
uv run python -m ipykernel install --user --name surveymatch --display-name "Python (surveymatch)"
quarto render paper.qmd --to all
```

Run the test suite with:

```bash
uv run pytest
```

## Layout

- `paper.qmd`, `references.bib` -- paper source and bibliography.
- `filters/` -- pandoc Lua filters used by the Markdown and Word builds.
- `src/surveymatch/` -- the duplicate-detection package (`matching.py` is
  the statistical core; `plotting.py` and `correlated.py` support the paper).
- `tests/` -- pytest unit tests for the package.
- `data/` -- the real survey data and codebook used in the paper's example;
  see `data/README.md` for provenance.
- `original/` -- the hand-written 2023 prototype (`comp_funcs.py` and a
  notebook) that `src/surveymatch/` extends; see `paper.qmd`'s AI use
  disclosure section for how the two relate.

## License

MIT, see `LICENSE`.
