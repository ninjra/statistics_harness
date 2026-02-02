# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- Normalized line endings across the repository for consistent Windows/WSL compatibility.

### Fixed
- Added a safe-rename hook for WSL editable installs plus one-command dev install scripts and README guidance.
- Corrected plugin loading for `plugin.py:Plugin` entrypoints, Excel ingest sheet handling, and JSON serialization of numpy bools.

### Added
- Added a `make dev` target for WSL-friendly setup.
