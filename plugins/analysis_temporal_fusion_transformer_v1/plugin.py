from __future__ import annotations
import logging
import traceback

import numpy as np
import pandas as pd
from statistic_harness.core.types import PluginResult, PluginArtifact, PluginError

logger = logging.getLogger(__name__)

MIN_POINTS = 50


class Plugin:
    def run(self, ctx) -> PluginResult:
        try:
            df = ctx.dataset_loader()
            if df.empty:
                return PluginResult("na", "Empty dataset", {}, [], [], None)

            # Find first numeric column with enough data
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                return PluginResult("na", "No numeric columns found", {}, [], [], None)

            target_col = None
            series = None
            for col in numeric_cols:
                s = df[col].dropna().values.astype(float)
                if len(s) >= MIN_POINTS:
                    target_col = col
                    series = s
                    break

            if series is None:
                return PluginResult(
                    "na",
                    f"No numeric column with >= {MIN_POINTS} non-null values",
                    {}, [], [], None,
                )

            try:
                import torch
                import torch.nn as nn
            except ImportError:
                return PluginResult("na", "torch not installed", {}, [], [], None)

            # Normalize
            mu, sigma = float(np.mean(series)), float(np.std(series))
            if sigma < 1e-12:
                return PluginResult(
                    "na", f"Column '{target_col}' has zero variance", {}, [], [], None,
                )
            normed = (series - mu) / sigma

            # Train/test split: 80/20
            split = int(len(normed) * 0.8)
            train_data = normed[:split]
            test_data = normed[split:]

            # Build sequences for LSTM
            settings = ctx.settings or {}
            lookback = min(settings.get("lookback", 10), split - 1)
            if lookback < 2:
                lookback = 2

            def make_sequences(data, lookback):
                X, y = [], []
                for i in range(lookback, len(data)):
                    X.append(data[i - lookback: i])
                    y.append(data[i])
                return (
                    torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1),
                    torch.tensor(np.array(y), dtype=torch.float32),
                )

            X_train, y_train = make_sequences(train_data, lookback)
            if len(X_train) == 0:
                return PluginResult("na", "Not enough training data after sequencing", {}, [], [], None)

            # Simple LSTM model
            torch.manual_seed(ctx.run_seed)

            class SimpleLSTM(nn.Module):
                def __init__(self, hidden_size=32):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
                    self.fc = nn.Linear(hidden_size, 1)

                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :]).squeeze(-1)

            model = SimpleLSTM(hidden_size=32)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()

            # Train
            n_epochs = settings.get("n_epochs", 50)
            model.train()
            for _epoch in range(n_epochs):
                optimizer.zero_grad()
                pred = model(X_train)
                loss = loss_fn(pred, y_train)
                loss.backward()
                optimizer.step()

            # Predict on test set using rolling forecast
            model.eval()
            full_normed = normed
            preds = []
            with torch.no_grad():
                for i in range(split, len(full_normed)):
                    seq = full_normed[i - lookback: i]
                    inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                    p = model(inp).item()
                    preds.append(p)

            preds = np.array(preds)
            actuals = test_data

            # Trim to same length
            n = min(len(preds), len(actuals))
            preds = preds[:n]
            actuals = actuals[:n]

            # Denormalize for metrics
            preds_raw = preds * sigma + mu
            actuals_raw = actuals * sigma + mu

            rmse = float(np.sqrt(np.mean((preds_raw - actuals_raw) ** 2)))
            mae = float(np.mean(np.abs(preds_raw - actuals_raw)))

            findings = [{
                "kind": "time_series",
                "measurement_type": "measured",
                "target_column": target_col,
                "train_size": split,
                "test_size": n,
                "rmse": round(rmse, 6),
                "mae": round(mae, 6),
                "lookback": lookback,
                "n_epochs": n_epochs,
                "method": "LSTM time-series forecast",
            }]

            return PluginResult(
                status="ok",
                summary=(
                    f"LSTM forecast on '{target_col}': RMSE={rmse:.4f}, MAE={mae:.4f} "
                    f"(train={split}, test={n})"
                ),
                metrics={
                    "target_column": target_col,
                    "n_points": len(series),
                    "train_size": split,
                    "test_size": n,
                    "rmse": round(rmse, 6),
                    "mae": round(mae, 6),
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"Temporal fusion transformer (LSTM) failed: {e}", exc_info=True)
            return PluginResult(
                status="error",
                summary=f"LSTM forecast failed: {e}",
                metrics={},
                findings=[],
                artifacts=[],
                error=PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()),
            )
