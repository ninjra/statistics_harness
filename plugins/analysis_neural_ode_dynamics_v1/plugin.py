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
                return PluginResult("na", "Empty dataset", {}, [], [], None)

            settings = ctx.settings or {}
            col = settings.get("column")

            # Auto-detect: pick the first numeric column with enough data
            if not col:
                for c in df.select_dtypes(include="number").columns:
                    if df[c].dropna().shape[0] >= 50:
                        col = c
                        break

            if not col or col not in df.columns:
                return PluginResult(
                    "na",
                    "No suitable numeric column found (need >= 50 non-null values)",
                    {}, [], [], None,
                )

            series = df[col].dropna().values.astype(float)
            if len(series) < 50:
                return PluginResult(
                    "na",
                    f"Insufficient data points ({len(series)}), need >= 50",
                    {}, [], [], None,
                )

            import torch
            import torch.nn as nn
            from torchdiffeq import odeint

            torch.manual_seed(ctx.run_seed)

            # Normalize to [0, 1]
            s_min, s_max = float(np.min(series)), float(np.max(series))
            if s_max - s_min < 1e-12:
                return PluginResult(
                    "na", "Constant series, nothing to model",
                    {}, [], [], None,
                )
            normed = (series - s_min) / (s_max - s_min)

            n = len(normed)
            t = torch.linspace(0, 1, n)
            y_true = torch.tensor(normed, dtype=torch.float32).unsqueeze(-1)

            # Simple ODE function: dy/dt = f(t, y)
            class ODEFunc(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(1, 32),
                        nn.Tanh(),
                        nn.Linear(32, 1),
                    )

                def forward(self, t, y):
                    return self.net(y)

            func = ODEFunc()
            optimizer = torch.optim.Adam(func.parameters(), lr=1e-2)

            y0 = y_true[0].unsqueeze(0)  # (1, 1)

            # Train with limited iterations
            n_iters = settings.get("n_iters", 100)
            best_loss = float("inf")
            for i in range(n_iters):
                optimizer.zero_grad()
                y_pred = odeint(func, y0, t).squeeze(-1).squeeze(-1)
                loss = torch.mean((y_pred - y_true.squeeze(-1)) ** 2)
                loss.backward()
                optimizer.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()

            # Final prediction
            with torch.no_grad():
                y_pred = odeint(func, y0, t).squeeze(-1).squeeze(-1)

            y_pred_np = y_pred.numpy()
            y_true_np = normed

            # Denormalize for RMSE
            y_pred_denorm = y_pred_np * (s_max - s_min) + s_min
            y_true_denorm = series

            rmse = float(np.sqrt(np.mean((y_pred_denorm - y_true_denorm) ** 2)))
            mae = float(np.mean(np.abs(y_pred_denorm - y_true_denorm)))
            series_std = float(np.std(series))
            good_fit = rmse < series_std

            findings = [{
                "kind": "time_series",
                "measurement_type": "measured",
                "column": col,
                "rmse": round(rmse, 6),
                "mae": round(mae, 6),
                "series_std": round(series_std, 6),
                "good_fit": good_fit,
                "n_points": n,
                "n_iters": n_iters,
                "method": "Neural ODE (torchdiffeq)",
            }]

            return PluginResult(
                status="ok",
                summary=(
                    f"Neural ODE on '{col}': RMSE={rmse:.4f}, MAE={mae:.4f}, "
                    f"good_fit={good_fit}"
                ),
                metrics={
                    "column": col,
                    "n_points": n,
                    "rmse": round(rmse, 6),
                    "mae": round(mae, 6),
                    "good_fit": good_fit,
                    "final_loss": round(best_loss, 6),
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"Neural ODE dynamics failed: {e}", exc_info=True)
            return PluginResult(
                status="error",
                summary=f"Neural ODE dynamics failed: {e}",
                metrics={},
                findings=[],
                artifacts=[],
                error=PluginError(
                    type=type(e).__name__,
                    message=str(e),
                    traceback=traceback.format_exc(),
                ),
            )
