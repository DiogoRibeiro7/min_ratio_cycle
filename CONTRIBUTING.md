# Contributing to min-ratio-cycle

Thanks for your interest in improving **min-ratio-cycle**! This guide explains how to set up your environment, the project conventions, and the pull‑request process.

--------------------------------------------------------------------------------

## Ground rules

- Be respectful and follow our [Code of Conduct](./CODE_OF_CONDUCT.md).
- Favor small, focused PRs with clear motivation and tests.
- Use **Conventional Commits**: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `perf:`, `build:`, `chore:`.
- Public API is documented in the README under **API surface (stable)**; changing it requires a discussion/issue first.

--------------------------------------------------------------------------------

## Environment setup

1. **Install Poetry** (<https://python-poetry.org/docs/#installation>)

```bash
poetry --version
```

1. **Install dependencies** and enable hooks

```bash
poetry install
poetry run pre-commit install
```

1. **Run the test suite & quality checks**

```bash
poetry run pytest --cov=min_ratio_cycle
poetry run mypy min_ratio_cycle
poetry run black . && poetry run isort .
poetry run flake8 .
poetry run bandit -r min_ratio_cycle
```

> Tip: for consistent benchmarking, pin BLAS threads: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.

--------------------------------------------------------------------------------

## Development workflow

- **Branch** from `main` using a descriptive name, e.g., `feat/exact-mode-fastpath` or `fix/negcycle-offbyone`.
- **Write tests** alongside code (`tests/`), prefer small unit tests; add property tests when relevant.
- **Document** new APIs/behaviors in docstrings and the README. Keep docs building (`sphinx-build`).
- **Type hints** are required for new/changed code. We ship a `py.typed` marker for downstream tooling.
- **Performance**: if changing core loops/oracles, add a benchmark note or evidence.

--------------------------------------------------------------------------------

## Pull request checklist

- [ ] Tests added/updated and passing locally (`pytest`).
- [ ] Type checks pass (`mypy`).
- [ ] Style/lint pass (`black`, `isort`, `flake8`).
- [ ] Security scan pass (`bandit`).
- [ ] Docs updated (README / Sphinx).
- [ ] Changelog entry added if user‑visible change.
- [ ] CI is green.

--------------------------------------------------------------------------------

## Reporting bugs / proposing features

- **Bugs**: open an issue with a minimal reproducible example and environment details.
- **Features**: start a discussion/issue explaining motivation, alternatives, and expected API.

--------------------------------------------------------------------------------

## Release process (maintainers)

- Use semantic versioning. Tag releases on `main` (e.g., `v0.1.0`).
- Publish to PyPI via GitHub Actions workflow (builds must be green).

--------------------------------------------------------------------------------

## Contact

For security issues or CoC enforcement, write to **<dfr@esmad.ipp.pt>**.
