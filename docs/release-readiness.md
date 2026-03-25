# Release readiness assessment

Date: 2026-03-25

## Current status

- Test suite passes (`59 passed`), but emits a runtime warning from PyTorch because `numpy` is not installed in the environment.
- Packaging metadata is minimal and still missing common PyPI release fields.
- Distribution validation tooling (`python -m build`, `twine check`) is not yet installed/configured in the current dev setup.

## Improvements still recommended before release

1. **Add licensing artifacts**
   - Add a `LICENSE` file in the repository root.
   - Add a `license` field in `pyproject.toml` (or SPDX expression).

2. **Strengthen package metadata for PyPI discoverability**
   - Add `project.urls` (repository, issue tracker, docs).
   - Add `classifiers` and `keywords`.

3. **Add release notes discipline**
   - Introduce a `CHANGELOG.md` and capture notable user-facing changes per version.

4. **Harden the release pipeline**
   - Add `build` and `twine` to a release/dev dependency group.
   - Run `python -m build` and `twine check dist/*` in CI for release candidates.

5. **Close environment dependency gaps**
   - Add `numpy` to appropriate dependencies (or document it explicitly) to avoid PyTorch warnings and improve reproducibility.

6. **Automate quality gates in CI**
   - Ensure CI runs tests in a clean environment and optionally on a small Python/PyTorch version matrix before tagging.

## Notes

The README already includes a concise release checklist; the items above expand it into an actionable pre-release plan.
