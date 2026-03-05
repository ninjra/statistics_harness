from __future__ import annotations

from typing import Any


def _infer_ideaspace_roles(columns_index: list[dict[str, Any]]) -> dict[str, Any]:
    """Infer dataset "roles" from column metadata for ideaspace reporting."""

    names = [str(col.get("name") or "") for col in columns_index if isinstance(col, dict)]
    lowered = [n.lower() for n in names]
    roles = [str(col.get("role") or "").lower() for col in columns_index if isinstance(col, dict)]

    def pick(tokens: tuple[str, ...]) -> str | None:
        for name in names:
            lname = name.lower()
            if any(tok in lname for tok in tokens):
                return name
        return None

    time_col = pick(("time", "timestamp", "date", "created", "updated", "start", "end"))
    process_col = pick(("process", "activity", "task", "action", "job", "step", "workflow"))
    host_col = pick(("host", "server", "node", "instance", "machine"))
    user_col = pick(("user", "owner", "operator", "agent"))
    params = []
    for name in names:
        lname = name.lower()
        if any(tok in lname for tok in ("param", "type", "code", "variant", "reason")):
            params.append(name)
            if len(params) >= 5:
                break
    case_id_col = pick(("case", "trace", "span", "correlation", "request_id", "session"))
    if not case_id_col:
        # If a column is explicitly tagged as ID in a template, surface it.
        for idx, role in enumerate(roles):
            if role and "id" in role and idx < len(names):
                case_id_col = names[idx]
                break

    has_coords = any(any(tok in n for tok in ("lat", "lon", "coord", "longitude", "latitude")) for n in lowered)
    has_text = any(any(tok in n for tok in ("message", "error", "exception", "trace", "stack", "log")) for n in lowered)
    return {
        "time_column": time_col,
        "process_column": process_col,
        "host_column": host_col,
        "user_column": user_col,
        "params_columns": params,
        "case_id_column": case_id_col,
        "has_coords": has_coords,
        "has_text": has_text,
    }


def _ideaspace_families_summary(plugins: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize applicability (ok/na) of idea families A-F."""

    families: dict[str, list[str]] = {
        "A_tda": [
            "analysis_tda_persistent_homology",
            "analysis_tda_persistence_landscapes",
            "analysis_tda_mapper_graph",
            "analysis_tda_betti_curve_changepoint",
        ],
        "B_topographic": [
            "analysis_topographic_similarity_angle_projection",
            "analysis_topographic_angle_dynamics",
            "analysis_topographic_tanova_permutation",
            "analysis_map_permutation_test_karniski",
        ],
        "C_surface": [
            "analysis_surface_multiscale_wavelet_curvature",
            "analysis_surface_fractal_dimension_variogram",
            "analysis_surface_rugosity_index",
            "analysis_surface_terrain_position_index",
            "analysis_surface_fabric_sso_eigen",
            "analysis_surface_hydrology_flow_watershed",
            "analysis_surface_roughness_metrics",
            "analysis_monte_carlo_surface_uncertainty",
        ],
        "D_classic_auto": [
            "analysis_ttests_auto",
            "analysis_chi_square_association",
            "analysis_anova_auto",
            "analysis_regression_auto",
            "analysis_time_series_analysis_auto",
            "analysis_cluster_analysis_auto",
            "analysis_pca_auto",
        ],
        "E_uncertainty": [
            "analysis_bayesian_point_displacement",
        ],
        "F_ops_levers": [
            "analysis_actionable_ops_levers_v1",
        ],
    }

    out: list[dict[str, Any]] = []
    for family, pids in families.items():
        present = []
        for pid in pids:
            plugin = plugins.get(pid) if isinstance(plugins, dict) else None
            if not isinstance(plugin, dict):
                continue
            present.append(
                {
                    "plugin_id": pid,
                    "status": plugin.get("status"),
                    "summary": plugin.get("summary"),
                }
            )
        if not present:
            continue
        statuses = [str(p.get("status") or "") for p in present]
        applicable = any(s == "ok" for s in statuses)
        all_na = all(s in {"na", "not_applicable"} for s in statuses)
        reason = None
        if all_na:
            reason = next((p.get("summary") for p in present if isinstance(p.get("summary"), str)), None)
        out.append(
            {
                "family": family,
                "plugins": present,
                "applicable": bool(applicable),
                "all_not_applicable": bool(all_na),
                "reason": reason,
            }
        )
    return out
