from __future__ import annotations

import hashlib
from pathlib import Path

from statistic_harness.core.utils import file_sha256

# Keep in sync with core.stat_plugins.registry.ALIAS_MAP, but duplicated here so
# changing cache-hash logic does not force all stat plugins to rerun.
_ALIAS_MAP: dict[str, str] = {
    "analysis_isolation_forest": "analysis_isolation_forest_anomaly",
    "analysis_log_template_drain": "analysis_log_template_mining_drain",
    "analysis_robust_pca_pcp": "analysis_robust_pca_sparse_outliers",
}


def stat_plugin_effective_code_hash(plugin_id: str) -> str | None:
    """Per-plugin effective code hash for stat-plugin wrappers (plugins/* -> run_plugin()).

    Stat plugins are commonly thin wrappers whose module file rarely changes. Their actual
    behavior lives in handlers under src/statistic_harness/core/stat_plugins/*. This helper
    hashes the relevant handler module(s) so cache reuse reflects handler changes.
    """

    if not isinstance(plugin_id, str) or not plugin_id.strip():
        return None
    resolved = _ALIAS_MAP.get(plugin_id, plugin_id)

    base = Path(__file__).resolve().parent
    files: list[Path] = []

    # Shared stat-plugin support that can affect many plugins.
    for name in ("__init__.py", "columns.py", "references.py"):
        path = base / name
        if path.exists():
            files.append(path)

    # Handler module selection.
    try:
        from statistic_harness.core.stat_plugins.topo_tda_addon import (
            HANDLERS as TOPO_TDA_ADDON_HANDLERS,
        )
    except Exception:
        TOPO_TDA_ADDON_HANDLERS = {}
    try:
        from statistic_harness.core.stat_plugins.ideaspace import (
            HANDLERS as IDEASPACE_HANDLERS,
        )
    except Exception:
        IDEASPACE_HANDLERS = {}

    if isinstance(TOPO_TDA_ADDON_HANDLERS, dict) and resolved in TOPO_TDA_ADDON_HANDLERS:
        path = base / "topo_tda_addon.py"
        if path.exists():
            files.append(path)
    elif isinstance(IDEASPACE_HANDLERS, dict) and resolved in IDEASPACE_HANDLERS:
        path = base / "ideaspace.py"
        if path.exists():
            files.append(path)
    else:
        path = base / "registry.py"
        if path.exists():
            files.append(path)

    if not files:
        return None
    h = hashlib.sha256()
    for path in sorted(set(files), key=lambda p: str(p)):
        try:
            digest = file_sha256(path)
        except Exception:
            continue
        h.update(str(path.name).encode("utf-8"))
        h.update(digest.encode("utf-8"))
    return h.hexdigest()

