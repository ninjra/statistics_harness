from __future__ import annotations
import logging
import traceback
import numpy as np
import pandas as pd
from statistic_harness.core.types import PluginResult, PluginArtifact, PluginError

logger = logging.getLogger(__name__)


class Plugin:
    def run(self, ctx) -> PluginResult:
        try:
            df = ctx.dataset_loader()
            if df.empty:
                return PluginResult("skipped", "Empty dataset", {}, [], [], None)

            # Select numeric columns and drop rows with any NaN
            num_df = df.select_dtypes(include="number").dropna()
            if num_df.shape[1] < 1:
                return PluginResult(
                    "skipped", "No numeric columns found",
                    {}, [], [], None,
                )
            if num_df.shape[0] < 20:
                return PluginResult(
                    "skipped",
                    f"Insufficient rows ({num_df.shape[0]}), need >= 20",
                    {}, [], [], None,
                )

            import torch
            import torch.nn as nn

            torch.manual_seed(ctx.run_seed)

            # Standardize
            means = num_df.mean()
            stds = num_df.std().replace(0, 1)
            standardized = (num_df - means) / stds

            data_tensor = torch.tensor(
                standardized.values, dtype=torch.float32,
            )
            n_features = data_tensor.shape[1]
            hidden_dim = max(2, n_features // 2)

            # Simple autoencoder
            encoder = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.ReLU(),
            )
            decoder = nn.Sequential(
                nn.Linear(hidden_dim, n_features),
            )
            model = nn.Sequential(encoder, decoder)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            # Train
            settings = ctx.settings or {}
            epochs = settings.get("epochs", 50)
            model.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                reconstructed = model(data_tensor)
                loss = loss_fn(reconstructed, data_tensor)
                loss.backward()
                optimizer.step()

            # Compute reconstruction error per row
            model.eval()
            with torch.no_grad():
                reconstructed = model(data_tensor)
                errors = torch.mean(
                    (data_tensor - reconstructed) ** 2, dim=1,
                ).numpy()

            # Threshold: mean + 2*std of reconstruction error
            threshold = float(np.mean(errors) + 2.0 * np.std(errors))
            anomaly_mask = errors > threshold
            anomaly_rate = float(np.mean(anomaly_mask))
            n_anomalies = int(np.sum(anomaly_mask))

            # Top anomalous rows (by error, up to 10)
            anomaly_indices = np.where(anomaly_mask)[0]
            sorted_anom = anomaly_indices[np.argsort(errors[anomaly_indices])[::-1]]
            top_rows = [
                {
                    "original_index": int(num_df.index[i]),
                    "reconstruction_error": round(float(errors[i]), 6),
                }
                for i in sorted_anom[:10]
            ]

            findings = [{
                "kind": "anomaly",
                "measurement_type": "measured",
                "anomaly_rate": round(anomaly_rate, 6),
                "n_anomalies": n_anomalies,
                "threshold": round(threshold, 6),
                "n_features": n_features,
                "top_anomalous_rows": top_rows,
                "method": "Autoencoder (PyTorch)",
            }]

            return PluginResult(
                status="ok",
                summary=(
                    f"Autoencoder anomaly detection: {n_anomalies} anomalies "
                    f"({anomaly_rate:.2%}) out of {len(errors)} rows, "
                    f"threshold={threshold:.4f}"
                ),
                metrics={
                    "n_rows": len(errors),
                    "n_features": n_features,
                    "n_anomalies": n_anomalies,
                    "anomaly_rate": round(anomaly_rate, 6),
                    "threshold": round(threshold, 6),
                    "mean_reconstruction_error": round(float(np.mean(errors)), 6),
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"Autoencoder anomaly detection failed: {e}", exc_info=True)
            return PluginResult(
                status="error",
                summary=f"Autoencoder anomaly detection failed: {e}",
                metrics={},
                findings=[],
                artifacts=[],
                error=PluginError(
                    type=type(e).__name__,
                    message=str(e),
                    traceback=traceback.format_exc(),
                ),
            )
