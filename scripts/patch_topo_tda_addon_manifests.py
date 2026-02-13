from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PLUGINS_DIR = ROOT / "plugins"


FAMILY_D = {
    "analysis_ttests_auto": ["needs_numeric"],
    "analysis_chi_square_association": ["needs_numeric"],
    "analysis_anova_auto": ["needs_numeric"],
    "analysis_regression_auto": ["needs_numeric"],
    "analysis_time_series_analysis_auto": ["needs_numeric", "needs_timestamp"],
    "analysis_survival_time_to_event": ["needs_numeric"],
    "analysis_factor_analysis_auto": ["needs_multi_numeric"],
    "analysis_cluster_analysis_auto": ["needs_multi_numeric"],
    "analysis_pca_auto": ["needs_multi_numeric"],
}

FAMILY_B = {
    "analysis_topographic_similarity_angle_projection": ["needs_multi_numeric"],
    "analysis_topographic_angle_dynamics": ["needs_multi_numeric", "needs_timestamp"],
    "analysis_topographic_tanova_permutation": ["needs_multi_numeric"],
    "analysis_map_permutation_test_karniski": ["needs_multi_numeric"],
}

FAMILY_HEAVY = {
    "analysis_tda_persistent_homology",
    "analysis_tda_persistence_landscapes",
    "analysis_tda_mapper_graph",
    "analysis_tda_betti_curve_changepoint",
    "analysis_surface_multiscale_wavelet_curvature",
    "analysis_surface_fractal_dimension_variogram",
    "analysis_surface_rugosity_index",
    "analysis_surface_terrain_position_index",
    "analysis_surface_fabric_sso_eigen",
    "analysis_surface_hydrology_flow_watershed",
    "analysis_bayesian_point_displacement",
    "analysis_monte_carlo_surface_uncertainty",
    "analysis_surface_roughness_metrics",
}


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _dump(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> None:
    updated = 0
    for manifest in sorted(PLUGINS_DIR.glob("analysis_*/plugin.yaml")):
        data = _load(manifest)
        pid = str(data.get("id") or manifest.parent.name)
        if pid in FAMILY_D:
            data["capabilities"] = FAMILY_D[pid]
            data.setdefault("settings", {}).setdefault("description", "Classic stats auto-scan suite.")
            updated += 1
        elif pid in FAMILY_B:
            data["capabilities"] = FAMILY_B[pid]
            data.setdefault("settings", {}).setdefault("description", "Topographic similarity / permutation map tests.")
            updated += 1
        elif pid in FAMILY_HEAVY:
            data["capabilities"] = ["topo_tda_addon", "needs_multi_numeric"]
            data.setdefault("settings", {}).setdefault("description", "Topo/TDA addon (heavy; opt-in).")
            updated += 1
        else:
            continue
        _dump(manifest, data)
    print(f"updated={updated}")


if __name__ == "__main__":
    main()

