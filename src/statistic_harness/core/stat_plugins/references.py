from __future__ import annotations

from typing import Any


REFERENCE_LIBRARY: dict[str, dict[str, Any]] = {
    "control_chart": {
        "title": "NIST/SEMATECH e-Handbook of Statistical Methods (Control Charts)",
        "url": "https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc3.htm",
        "doi": "",
        "notes": "EWMA/CUSUM/Individuals charts overview",
    },
    "pelt": {
        "title": "Killick et al. (2012) Optimal detection of changepoints",
        "url": "https://arxiv.org/abs/1101.1438",
        "doi": "",
        "notes": "PELT changepoint detection",
    },
    "energy": {
        "title": "Matteson & James (2014) A nonparametric approach for multiple change point analysis",
        "url": "",
        "doi": "",
        "notes": "Energy statistics changepoints",
    },
    "adwin": {
        "title": "Bifet & Gavalda (2007) ADWIN",
        "url": "",
        "doi": "",
        "notes": "Adaptive window drift detection",
    },
    "mmd": {
        "title": "Gretton et al. (2012) A kernel two-sample test",
        "url": "",
        "doi": "",
        "notes": "MMD",
    },
    "bh_fdr": {
        "title": "Benjamini & Hochberg (1995) Controlling the false discovery rate",
        "url": "",
        "doi": "",
        "notes": "BH-FDR",
    },
    "isolation_forest": {
        "title": "Liu et al. (2008) Isolation Forest",
        "url": "",
        "doi": "",
        "notes": "Isolation Forest",
    },
    "robust_pca": {
        "title": "Candès et al. (2011) Robust PCA",
        "url": "",
        "doi": "",
        "notes": "Principal Component Pursuit",
    },
    "drain": {
        "title": "He et al. (2017) Drain: Log Parsing",
        "url": "",
        "doi": "",
        "notes": "Drain log template mining",
    },
    "process_mining": {
        "title": "van der Aalst (2016) Process Mining: Data Science in Action",
        "url": "",
        "doi": "",
        "notes": "Conformance & process mining",
    },
    "lof": {
        "title": "Breunig et al. (2000) LOF",
        "url": "",
        "doi": "",
        "notes": "Local Outlier Factor",
    },
    "ocsvm": {
        "title": "Schölkopf et al. (2001) Estimating the Support of a High-Dimensional Distribution",
        "url": "",
        "doi": "",
        "notes": "One-class SVM novelty detection",
    },
    "evt": {
        "title": "Coles (2001) An Introduction to Statistical Modeling of Extreme Values",
        "url": "",
        "doi": "",
        "notes": "Extreme value theory",
    },
    "matrix_profile": {
        "title": "Yeh et al. (2016) Matrix Profile",
        "url": "",
        "doi": "",
        "notes": "Time-series motifs/discords",
    },
    "kleinberg_burst": {
        "title": "Kleinberg (2003) Bursty and Hierarchical Structure in Streams",
        "url": "",
        "doi": "",
        "notes": "Burst detection",
    },
    "hawkes": {
        "title": "Hawkes (1971) Spectra of some self-exciting and mutually exciting point processes",
        "url": "",
        "doi": "",
        "notes": "Hawkes processes",
    },
    "kalman": {
        "title": "Kalman (1960) A New Approach to Linear Filtering and Prediction Problems",
        "url": "",
        "doi": "",
        "notes": "Kalman filter",
    },
    "lda": {
        "title": "Blei et al. (2003) Latent Dirichlet Allocation",
        "url": "",
        "doi": "",
        "notes": "LDA topic modeling",
    },
    "hmm": {
        "title": "Rabiner (1989) A Tutorial on Hidden Markov Models",
        "url": "",
        "doi": "",
        "notes": "HMMs",
    },
    "transfer_entropy": {
        "title": "Schreiber (2000) Measuring information transfer",
        "url": "",
        "doi": "",
        "notes": "Transfer entropy",
    },
    "granger": {
        "title": "Granger (1969) Investigating causal relations",
        "url": "",
        "doi": "",
        "notes": "Granger causality",
    },
    "copula": {
        "title": "Nelsen (2006) An Introduction to Copulas",
        "url": "",
        "doi": "",
        "notes": "Copula dependence",
    },
    "glasso": {
        "title": "Friedman et al. (2008) Sparse inverse covariance estimation (Graphical Lasso)",
        "url": "",
        "doi": "",
        "notes": "Graphical Lasso",
    },
    "kaplan_meier": {
        "title": "Kaplan & Meier (1958) Nonparametric estimation",
        "url": "",
        "doi": "",
        "notes": "Kaplan–Meier survival",
    },
    "cox": {
        "title": "Cox (1972) Regression Models and Life-Tables",
        "url": "",
        "doi": "",
        "notes": "Cox proportional hazards",
    },
    "quantile_regression": {
        "title": "Koenker & Bassett (1978) Regression Quantiles",
        "url": "",
        "doi": "",
        "notes": "Quantile regression",
    },
    "little": {
        "title": "Little (1961) A Proof for the Queuing Formula L = λW",
        "url": "",
        "doi": "",
        "notes": "Little’s law",
    },
    "kingman": {
        "title": "Kingman (1961) The Single Server Queue in Heavy Traffic",
        "url": "",
        "doi": "",
        "notes": "Kingman approximation",
    },
    "karniski1994": {
        "title": "Karniski et al. (1994) A distribution-free test for comparing multichannel EEG/ERP topographies",
        "url": "https://link.springer.com/article/10.1007/BF01187710",
        "doi": "",
        "notes": "Permutation-based, distribution-free map comparison (preview link)",
    },
    "topotoolbox2011": {
        "title": "Tian et al. (2011) TopoToolbox: A Matlab toolbox for topographic EEG/MEG data analysis",
        "url": "https://pdfs.semanticscholar.org/6ba7/d89d6c19125fbba1a3a4bf8b845a819d112e.pdf",
        "doi": "",
        "notes": "Angle/projection similarity for multivariate maps (Figure 1 description)",
    },
    "topotoolbox2010": {
        "title": "Schwanghart & Kuhn (2010) TopoToolbox: A set of MATLAB functions for topographic analysis",
        "url": "https://www.sciencedirect.com/science/article/abs/pii/S1364815209003053",
        "doi": "",
        "notes": "Terrain/hydrology operators overview (preview link)",
    },
    "py_topo_complexity_2025": {
        "title": "Lai et al. (2025) pyTopoComplexity: an open-source Python package for topographic complexity",
        "url": "https://esurf.copernicus.org/articles/13/417/2025/",
        "doi": "",
        "notes": "Surface complexity metrics (wavelet curvature, fractal dimension, rugosity, TPI)",
    },
    "tanir2008": {
        "title": "Tanir et al. (2008) Bayesian deformation analysis (examples for geodetic monitoring networks)",
        "url": "https://nhess.copernicus.org/articles/8/335/2008/",
        "doi": "",
        "notes": "Bayesian parameter estimation for point displacement uncertainty",
    },
    "oksanen2001": {
        "title": "Oksanen & Sarjakoski (2001) Error propagation analysis for DEM-based derivative uncertainty (Monte Carlo)",
        "url": "https://icaci.org/files/documents/ICC_proceedings/ICC2001/icc2001/file/f20006.pdf",
        "doi": "",
        "notes": "Monte Carlo propagation of surface error into derived metrics",
    },
    "tda_survey": {
        "title": "Edelsbrunner & Harer (2010) Computational Topology: An Introduction",
        "url": "",
        "doi": "",
        "notes": "Persistent homology basics (general reference)",
    },
    "ttest_nist": {
        "title": "NIST/SEMATECH e-Handbook of Statistical Methods (t-tests, ANOVA, regression)",
        "url": "https://www.itl.nist.gov/div898/handbook/",
        "doi": "",
        "notes": "Classic statistical methods overview",
    },
}


def _collect(*keys: str) -> list[dict[str, Any]]:
    seen = []
    for key in keys:
        ref = REFERENCE_LIBRARY.get(key)
        if ref and ref not in seen:
            seen.append(ref)
    return seen


def default_references_for_plugin(plugin_id: str) -> list[dict[str, Any]]:
    pid = plugin_id.lower()
    refs: list[dict[str, Any]] = []
    if "actionable_ops" in pid or "ops_levers" in pid:
        refs += _collect("ttest_nist", "little", "kingman")
    if "time_series" in pid or "cluster" in pid or "pca" in pid or "factor_analysis" in pid:
        refs += _collect("ttest_nist")
    if "control_chart" in pid:
        refs += _collect("control_chart")
    if "changepoint" in pid or "pelt" in pid:
        refs += _collect("pelt")
    if "energy" in pid or "edivisive" in pid:
        refs += _collect("energy")
    if "adwin" in pid:
        refs += _collect("adwin")
    if "mmd" in pid or "kernel_two_sample" in pid:
        refs += _collect("mmd")
    if "fdr" in pid:
        refs += _collect("bh_fdr")
    if "isolation_forest" in pid:
        refs += _collect("isolation_forest")
    if "robust_pca" in pid or "pcp" in pid:
        refs += _collect("robust_pca")
    if "drain" in pid or "template_mining" in pid:
        refs += _collect("drain")
    if "conformance" in pid or "process" in pid:
        refs += _collect("process_mining")
    if "local_outlier_factor" in pid or "lof" in pid:
        refs += _collect("lof")
    if "one_class_svm" in pid or "ocsvm" in pid:
        refs += _collect("ocsvm")
    if "evt" in pid:
        refs += _collect("evt")
    if "matrix_profile" in pid:
        refs += _collect("matrix_profile")
    if "burst" in pid:
        refs += _collect("kleinberg_burst")
    if "hawkes" in pid:
        refs += _collect("hawkes")
    if "kalman" in pid:
        refs += _collect("kalman")
    if "lda" in pid or "topic_model" in pid:
        refs += _collect("lda")
    if "hmm" in pid:
        refs += _collect("hmm")
    if "transfer_entropy" in pid:
        refs += _collect("transfer_entropy")
    if "granger" in pid or "lagged_predictability" in pid:
        refs += _collect("granger")
    if "copula" in pid:
        refs += _collect("copula")
    if "graphical_lasso" in pid:
        refs += _collect("glasso")
    if "kaplan_meier" in pid or "survival" in pid:
        refs += _collect("kaplan_meier")
    if "proportional_hazards" in pid or "cox" in pid:
        refs += _collect("cox")
    if "quantile_regression" in pid:
        refs += _collect("quantile_regression")
    if "little" in pid:
        refs += _collect("little")
    if "kingman" in pid:
        refs += _collect("kingman")
    if "tda" in pid:
        refs += _collect("tda_survey")
    if "topographic" in pid or "tanova" in pid or "map_permutation" in pid:
        refs += _collect("topotoolbox2011")
    if "karniski" in pid:
        refs += _collect("karniski1994")
    if "surface" in pid or "hydrology" in pid or "rugosity" in pid or "tpi" in pid:
        refs += _collect("py_topo_complexity_2025", "topotoolbox2010")
    if "bayesian" in pid or "point_displacement" in pid:
        refs += _collect("tanir2008")
    if "monte_carlo" in pid:
        refs += _collect("oksanen2001")
    if "ttest" in pid or "anova" in pid or "regression" in pid or "chi_square" in pid:
        refs += _collect("ttest_nist", "bh_fdr")
    return refs
