from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginResult

from .common import HAS_SKLEARN, KFold, LogisticRegression, RandomForestClassifier, StratifiedKFold, artifact, build_config, degraded, finding, prepare_data, roc_auc_score

PLUGIN_ID = "analysis_tabpfn_foundation_tabular_v1"


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    x = prepared.matrix
    if x.shape[0] < 80 or x.shape[1] < 2:
        return degraded(PLUGIN_ID, "Need >=80 rows and >=2 features for few-shot proxy", {"rows_used": int(x.shape[0]), "cols_used": int(x.shape[1])})
    y_raw = x[:, 0]
    y = (y_raw >= np.median(y_raw)).astype(int)
    x_in = x[:, 1:]
    class_counts = np.bincount(y, minlength=2)
    minority = int(np.min(class_counts))
    if minority < 2:
        rate = float(y.mean()) if y.size else 0.0
        findings = [
            finding(
                PLUGIN_ID,
                "Few-shot proxy not applicable for single-class slice",
                "Target signal has only one class in this run; use deterministic baseline uncertainty and collect more label diversity.",
                0.0,
                {"class_counts": class_counts.tolist(), "positive_rate": rate},
            )
        ]
        artifacts = [
            artifact(
                ctx,
                PLUGIN_ID,
                "tabpfn_not_applicable_single_class.json",
                {"class_counts": class_counts.tolist(), "positive_rate": rate},
            )
        ]
        return PluginResult(
            "ok",
            "Single-class target after binarization; emitted deterministic N/A fallback",
            {
                "positive_rate": rate,
                "class_0_count": int(class_counts[0]),
                "class_1_count": int(class_counts[1]),
                "not_applicable": 1,
            },
            findings,
            artifacts,
            None,
        )
    if not HAS_SKLEARN or LogisticRegression is None or RandomForestClassifier is None:
        rate = float(y.mean())
        uncertainty = float(rate * (1.0 - rate))
        artifacts = [artifact(ctx, PLUGIN_ID, "tabpfn_proxy_baseline.json", {"fallback": True, "positive_rate": rate, "uncertainty_proxy": uncertainty})]
        findings = [finding(PLUGIN_ID, "Few-shot proxy fallback executed", "Install sklearn for stronger few-shot proxy modeling; current output uses deterministic baseline uncertainty.", uncertainty, {"positive_rate": rate})]
        return PluginResult("degraded", "scikit-learn unavailable; used baseline proxy", {"positive_rate": rate, "uncertainty_proxy": uncertainty}, findings, artifacts, None)
    folds = max(3, min(6, int(config.get("n_bootstrap") or 12) // 2))
    folds = min(folds, max(2, minority))
    splitter = (
        StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(getattr(ctx, "run_seed", 0) or 0))
        if StratifiedKFold is not None
        else KFold(n_splits=folds, shuffle=True, random_state=int(getattr(ctx, "run_seed", 0) or 0))
    )
    probs = np.zeros((x_in.shape[0], 2), dtype=float)
    valid_mask = np.zeros(x_in.shape[0], dtype=bool)
    split_iter = splitter.split(x_in, y) if StratifiedKFold is not None else splitter.split(x_in)
    for train, test in split_iter:
        if np.unique(y[train]).size < 2:
            continue
        model_lr = LogisticRegression(max_iter=300, random_state=int(getattr(ctx, "run_seed", 0) or 0))
        model_rf = RandomForestClassifier(n_estimators=80, random_state=int(getattr(ctx, "run_seed", 0) or 0), min_samples_leaf=2)
        model_lr.fit(x_in[train], y[train])
        model_rf.fit(x_in[train], y[train])
        p_lr = model_lr.predict_proba(x_in[test])[:, 1]
        p_rf = model_rf.predict_proba(x_in[test])[:, 1]
        probs[test, 0] = p_lr
        probs[test, 1] = p_rf
        valid_mask[test] = True
    if not valid_mask.any():
        rate = float(y.mean()) if y.size else 0.0
        unc = np.full(x_in.shape[0], fill_value=rate * (1.0 - rate), dtype=float)
        pred = np.full(x_in.shape[0], fill_value=rate, dtype=float)
    else:
        pred = probs.mean(axis=1)
        unc = probs.std(axis=1)
    auc = float(roc_auc_score(y[valid_mask], pred[valid_mask])) if roc_auc_score is not None and valid_mask.any() else 0.5
    top_unc_idx = np.argsort(unc)[::-1][: min(20, len(unc))]
    artifacts = [artifact(ctx, PLUGIN_ID, "tabpfn_fewshot_proxy.json", {"auc_proxy": auc, "mean_uncertainty": float(np.mean(unc)), "high_uncertainty_rows": [int(i) for i in top_unc_idx]})]
    findings = [finding(PLUGIN_ID, "Few-shot uncertainty hotspots identified", "Rows with high predictive disagreement indicate unstable regimes where additional controls or features are needed.", float(np.mean(unc)), {"auc_proxy": auc, "mean_uncertainty": float(np.mean(unc))})]
    return PluginResult("ok", "Computed TabPFN-style few-shot proxy ensemble", {"auc_proxy": auc, "mean_uncertainty": float(np.mean(unc))}, findings, artifacts, None)
