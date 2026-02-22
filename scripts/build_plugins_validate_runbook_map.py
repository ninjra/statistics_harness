from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TechniqueSpec:
    ordinal: int
    technique: str
    references: list[str]
    preferred_plugin_id: str
    alternate_plugin_ids: list[str]


TECHNIQUES: list[TechniqueSpec] = [
    TechniqueSpec(1, "Conformal prediction", ["https://arxiv.org/abs/0706.3188"], "analysis_conformal_feature_prediction", ["analysis_online_conformal_changepoint"]),
    TechniqueSpec(2, "False Discovery Rate (Benjamini-Hochberg)", ["https://doi.org/10.1111/j.2517-6161.1995.tb02031.x"], "analysis_multiple_testing_fdr", []),
    TechniqueSpec(3, "BCa bootstrap confidence intervals", ["https://doi.org/10.1080/01621459.1987.10478410"], "analysis_bootstrap_ci_effect_sizes_v1", []),
    TechniqueSpec(4, "Knockoff filter", ["https://arxiv.org/abs/1404.5609"], "analysis_gaussian_knockoffs", ["analysis_knockoff_wrapper_rf"]),
    TechniqueSpec(5, "Quantile regression", ["https://www.econ.uiuc.edu/~roger/research/rq/jasa.pdf"], "analysis_quantile_regression_duration", ["analysis_quantile_regression_forest_v1"]),
    TechniqueSpec(6, "LASSO", ["https://doi.org/10.1111/j.2517-6161.1996.tb02080.x"], "analysis_fused_lasso_trend_filtering_v1", []),
    TechniqueSpec(7, "Elastic Net", ["https://doi.org/10.1111/j.1467-9868.2005.00503.x"], "analysis_elastic_net_regularized_glm_v1", []),
    TechniqueSpec(8, "Random Forests", ["https://doi.org/10.1023/A:1010933404324"], "analysis_random_forest_action_ranker_v1", ["analysis_knockoff_wrapper_rf"]),
    TechniqueSpec(9, "Gradient Boosting Machine", ["https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation--A-gradient-boosting-machine/10.1214/aos/1013203451.full"], "analysis_quantile_loss_boosting_v1", []),
    TechniqueSpec(10, "Isolation Forest", ["https://doi.org/10.1109/ICDM.2008.17"], "analysis_isolation_forest_anomaly", ["analysis_isolation_forest"]),
    TechniqueSpec(11, "Local Outlier Factor (LOF)", ["https://doi.org/10.1145/335191.335388"], "analysis_local_outlier_factor", []),
    TechniqueSpec(12, "Minimum Covariance Determinant (MCD)", ["https://wis.kuleuven.be/stat/robust/papers/2010/wire-mcd-review.pdf"], "analysis_minimum_covariance_determinant_v1", []),
    TechniqueSpec(13, "Huber M-estimator", ["https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full"], "analysis_robust_regression_huber_ransac_v1", []),
    TechniqueSpec(14, "Gaussian Process Regression", ["https://mitpress.mit.edu/9780262182539/gaussian-processes-for-machine-learning/"], "analysis_gaussian_process_regression_v1", []),
    TechniqueSpec(15, "Generalized Additive Models (GAM)", ["https://datamining.cs.ucdavis.edu/~filkov/courses/300/winter06/readings/gam.pdf"], "analysis_gam_spline_regression_v1", []),
    TechniqueSpec(16, "Mixed-effects (random-effects) models", ["https://doi.org/10.2307/2529876"], "analysis_mixed_effects_hierarchical_v1", []),
    TechniqueSpec(17, "Cox proportional hazards", ["https://doi.org/10.1111/j.2517-6161.1972.tb00899.x"], "analysis_survival_time_to_event", ["analysis_aft_survival_lognormal_v1"]),
    TechniqueSpec(18, "Bayesian Additive Regression Trees (BART)", ["https://projecteuclid.org/journals/annals-of-applied-statistics/volume-4/issue-1/BART-Bayesian-additive-regression-trees/10.1214/09-AOAS285.full"], "analysis_bart_uplift_surrogate_v1", []),
    TechniqueSpec(19, "Dirichlet Process (nonparametric Bayes)", ["https://projecteuclid.org/journals/annals-of-statistics/volume-1/issue-2/A-Bayesian-analysis-of-some-nonparametric-problems/10.1214/aos/1176342360.full"], "analysis_dirichlet_process_mixture_v1", ["analysis_dirichlet_multinomial_categorical_overdispersion_v1"]),
    TechniqueSpec(20, "Hidden Markov Models (HMM)", ["https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf"], "analysis_hmm_latent_state_sequences", []),
    TechniqueSpec(21, "Kalman filter", ["https://doi.org/10.1115/1.3662552"], "analysis_state_space_kalman_residuals", []),
    TechniqueSpec(22, "Granger causality", ["https://doi.org/10.2307/1912791"], "analysis_granger_causality_v1", []),
    TechniqueSpec(23, "Transfer entropy", ["https://doi.org/10.1103/PhysRevLett.85.461"], "analysis_transfer_entropy_directional", []),
    TechniqueSpec(24, "kNN mutual information estimator", ["https://doi.org/10.1103/PhysRevE.69.066138"], "analysis_mutual_information_screen", []),
    TechniqueSpec(25, "Independent Component Analysis (ICA)", ["https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf"], "analysis_ica_source_separation_v1", []),
    TechniqueSpec(26, "Non-negative Matrix Factorization (NMF)", ["https://doi.org/10.1038/44565"], "analysis_nonnegative_matrix_factorization_v1", []),
    TechniqueSpec(27, "t-SNE", ["https://www.jmlr.org/papers/v9/vandermaaten08a.html"], "analysis_tsne_embedding_v1", []),
    TechniqueSpec(28, "UMAP", ["https://arxiv.org/abs/1802.03426"], "analysis_umap_embedding_v1", []),
    TechniqueSpec(29, "NOTEARS", ["https://arxiv.org/abs/1803.01422"], "analysis_notears_linear", []),
    TechniqueSpec(30, "MICE (chained equations) imputation", ["https://www.jstatsoft.org/article/view/v045i03"], "analysis_mice_imputation_chained_equations_v1", []),
]


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _collect_manifests(repo_root: Path) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted((repo_root / "plugins").glob("*/plugin.yaml")):
        payload = _read_manifest(manifest_path)
        plugin_id = str(payload.get("id") or manifest_path.parent.name)
        payload["_path"] = str(manifest_path)
        manifests[plugin_id] = payload
    return manifests


def build_payload(repo_root: Path) -> list[dict[str, Any]]:
    manifests = _collect_manifests(repo_root)
    rows: list[dict[str, Any]] = []
    for spec in TECHNIQUES:
        candidates = [spec.preferred_plugin_id, *spec.alternate_plugin_ids]
        selected = next((pid for pid in candidates if pid in manifests), spec.preferred_plugin_id)
        manifest = manifests.get(selected, {})
        rows.append(
            {
                "ordinal": spec.ordinal,
                "technique": spec.technique,
                "references": list(spec.references),
                "plugin_id": selected,
                "preferred_plugin_id": spec.preferred_plugin_id,
                "alternate_plugin_ids": list(spec.alternate_plugin_ids),
                "implemented": bool(selected in manifests),
                "manifest_path": manifest.get("_path"),
                "capabilities": list(manifest.get("capabilities") or []),
                "lane": str(manifest.get("lane") or ""),
                "decision_capable": bool(manifest.get("decision_capable"))
                if "decision_capable" in manifest
                else None,
            }
        )
    return rows


def build_dependency_rows(repo_root: Path, mapping_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifests = _collect_manifests(repo_root)
    consumers_index: dict[str, list[str]] = {}
    for plugin_id, payload in manifests.items():
        for dep in payload.get("depends_on") or []:
            dep_id = str(dep)
            consumers_index.setdefault(dep_id, []).append(plugin_id)
    rows: list[dict[str, Any]] = []
    for item in mapping_rows:
        plugin_id = str(item["plugin_id"])
        manifest = manifests.get(plugin_id, {})
        depends_on = sorted(str(dep) for dep in (manifest.get("depends_on") or []))
        consumers = sorted(consumers_index.get(plugin_id, []))
        rows.append(
            {
                "ordinal": int(item["ordinal"]),
                "technique": str(item["technique"]),
                "plugin_id": plugin_id,
                "implemented": bool(item["implemented"]),
                "depends_on": depends_on,
                "consumer_count": len(consumers),
                "consumers": consumers,
            }
        )
    return rows


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Plugins Validate Runbook Technique Map",
        "",
        "| # | Technique | Plugin ID | Implemented |",
        "|---:|---|---|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['ordinal']} | {row['technique']} | `{row['plugin_id']}` | {'Y' if row['implemented'] else 'N'} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(repo_root: Path) -> tuple[Path, Path, Path]:
    out_dir = repo_root / "docs" / "release_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_rows = build_payload(repo_root)
    dep_rows = build_dependency_rows(repo_root, mapping_rows)

    mapping_json = out_dir / "plugins_validate_runbook_technique_map.json"
    mapping_md = out_dir / "plugins_validate_runbook_technique_map.md"
    dep_csv = out_dir / "plugins_validate_runbook_dependency_matrix.csv"

    mapping_json.write_text(
        json.dumps(mapping_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    mapping_md.write_text(_to_markdown(mapping_rows), encoding="utf-8")
    with dep_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ordinal",
                "technique",
                "plugin_id",
                "implemented",
                "depends_on",
                "consumer_count",
                "consumers",
            ],
        )
        writer.writeheader()
        for row in dep_rows:
            writer.writerow(
                {
                    "ordinal": row["ordinal"],
                    "technique": row["technique"],
                    "plugin_id": row["plugin_id"],
                    "implemented": "Y" if row["implemented"] else "N",
                    "depends_on": ";".join(row["depends_on"]),
                    "consumer_count": row["consumer_count"],
                    "consumers": ";".join(row["consumers"]),
                }
            )
    return mapping_json, mapping_md, dep_csv


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mapping_json, mapping_md, dep_csv = write_outputs(repo_root)
    payload = build_payload(repo_root)
    implemented = sum(1 for row in payload if row["implemented"])
    print(
        f"rows={len(payload)} implemented={implemented} "
        f"json={mapping_json} md={mapping_md} csv={dep_csv}"
    )


if __name__ == "__main__":
    main()
