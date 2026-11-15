# mesqual

Instructions for coding agents in this repo. The README is for users; this file adds the rules for changes.

mesqual is a generic framework for multi-scenario analysis of energy market simulations: datasets, flags, collections, comparisons, KPIs, maps and dashboards. Platform packages build on it.

## Scope

- mesqual stays generic: it does not know the concepts of one platform (see `docs/architecture/index.md`).
- Examples and tests use the synthetic test platform (`mesqual.testing`), never an external platform package.
- Dependencies: (common) open-source packages only.
- `scripts/` holds repository tools, for example the logo generator. They are not part of the package and not in the docs. Modules in `src/` have no `__main__` block, except the `__main__.py` of a package with a command line, for example `mesqual.skills`.

## Architecture

The contracts of the package are in `docs/architecture/` (start at `docs/architecture/index.md`). Read them before you change code. A change of a contract updates its architecture page in the same PR. A new architecture page gets its entry in the nav of `mkdocs.yml` and a link from `docs/architecture/index.md`.

## Public API

- Each API change and bug fix gets an entry in `CHANGELOG.md`, by the version rules at its head.

## Docs

- Public modules, classes and methods have a Google-style docstring. The docstrings are the mkdocs API reference.
- A new public module gets its page in `docs/api_reference/` and its entry in the nav of `mkdocs.yml`.
- Docs, docstrings, the README, and actual code must not contradict each other.
- `tests/test_docstring_examples.py` runs every `>>>` example in a docstring of the package. An example uses the synthetic platform `mesqual.testing`. It uses the names its module defines and the shared names of `shared_names()` in that file; it imports everything else. An example changes only classes it defines, except for class configs and the quantity types of `Units`: the tests reset these two. An example that cannot run is a fenced `python` block, not a `>>>` block. An attribute docstring gets a fenced block, not a `>>>` block: the test does not collect attribute docstrings.
- `tests/test_readme_examples.py` runs the `python` blocks of the README in order, as one script. The blocks use the synthetic platform `mesqual.testing` or data they write themselves, never an external platform package.
- `tests/test_architecture_examples.py` runs the `python` blocks of `docs/architecture/example-trade-balance.md` in order, as one script, and `tests/test_guides.py` does the same for each page in `docs/guides/`, one script per page. The blocks use the synthetic platform `mesqual.testing` or data the page writes itself. Each printed output equals the `text` block after it, and in the architecture example the abridged `info` block equals the same keys of `info`.
- The landing page `docs/index.md` includes the README sections between the `<!-- --8<-- [start:name] -->` and `[end:name]` markers. Links in the README are absolute (`https://docs.mesqual.io/...`, `https://github.com/helgeesch/mesqual/...`), so they work on GitHub, on PyPI (the README is the long description) and on the site.
- The skill `src/mesqual/skills/mesqual-study/SKILL.md` follows the same rules: `tests/test_skills.py` runs its `python` blocks in order on `make_dummy_study()`, `docs/use-with-agents.md` includes its sections between `--8<--` markers, and its links must resolve on the built site (`tests/test_agent_docs.py`). `docs/use-with-agents.md` is the page of `mesqual.skills`, which has no public names.

## Writing style

- Simplified Technical English: avoid long sentences and filler.
- No em-dashes.
- Write mesqual in lowercase in prose and headings, also at the start of a sentence. Document titles (the README title, `docs/index.md`, the site name) and logo alt text may use MESQUAL.
- Docs describe the current state. Change history goes into commit messages and `CHANGELOG.md`.

## Checks

New behavior has tests. A bug fix has a regression test that fails before the fix. A test needs no network and no browser. The name of a new test says what it proves. Every warning fails the tests (`filterwarnings = ["error"]` in `pyproject.toml`): fix the cause, and catch an expected warning in its test with `pytest.warns`. Run these checks locally before every pull request:

```
uv sync --all-extras
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run mkdocs build --strict
```
