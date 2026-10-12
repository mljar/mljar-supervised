"""Portable relative paths stored in model metadata."""

import os
import posixpath


def portable_artifact_path(*parts):
    """Serialize artifact paths with /, accepting legacy Windows separators."""
    return posixpath.join(*(os.fspath(part).replace("\\", "/") for part in parts))


def resolve_artifact_path(results_path, artifact_path):
    """Resolve a stored path on this host without modifying the results root."""
    return os.path.join(results_path, portable_artifact_path(artifact_path))
