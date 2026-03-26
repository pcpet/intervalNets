# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-25

### Added
- Initial public package metadata and release scaffolding.
- MIT license.
- Project URLs, classifiers, and keywords in `pyproject.toml`.
- Development dependencies for package build and validation (`build`, `twine`).

### Changed
- Optional `torch` extra now also includes `numpy` to avoid PyTorch runtime warnings in common workflows.
