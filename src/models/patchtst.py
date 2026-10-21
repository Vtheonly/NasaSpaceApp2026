"""PatchTST forecaster architecture (selected temporal-model candidate).

Reference: "A Time Series is Worth 64 Words: Long-term Forecasting with
Transformers" (Nie et al., ICLR 2023). Implementation notes:

- **Channel independence:** each channel (target series, covariates) is
  patched and encoded independently with shared weights; only the target
  channel's encoding feeds the quantile head.
- **Patching:** strided patches of ``patch_len`` days with ``stride``; defaults
  follow the documented starting values (16-day patches, stride 8, d_model 128,
  four encoder blocks, eight heads, dropout 0.1) and are explicitly *tunable,
  not proven optimal* (docs/ARCHITECTURE.md).
- **Probabilistic head:** one output per (horizon, quantile) trained with the
  pinball (quantile) loss; predictions are sorted per step so
  q_low <= q_median <= q_high always holds, then clipped at zero (FRP >= 0).
- **Missing-coverage policy:** gap days are filled with 0.0 and flagged by a
  coverage channel (0.0), so the network can distinguish "no observation" from
  "zero detections" instead of silently treating gaps as zeros.

PyTorch is imported lazily: without torch, :class:`PatchTSTForecaster` raises
:class:`TorchDependencyError` with an actionable message (graceful failure,
issue #13) while the seasonal-naive/persistence baselines remain usable.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import ForecastConfig, seed_everything
from src.data.aggregation import COVERAGE_STATUS_COLUMN, DATE_COLUMN


class TorchDependencyError(RuntimeError):
    """Raised when PyTorch is required but not installed (clear, offline-safe)."""


def require_torch():
    """Import PyTorch lazily with a clear, offline-friendly error message."""
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise TorchDependencyError(
            "PyTorch is required for the selected PatchTST forecaster. Install the "
            "optional dependency (pip install 'hotspot-harmonization[forecast]', or "
            "stage torch CPU wheels into the offline wheelhouse); the seasonal-naive "
            "and persistence baselines remain available without it."
        ) from exc
    return torch


class PatchTSTForecaster:
    """Channel-independent PatchTST with a quantile head over the horizon.

    Training windows are built chronologically (stride 1) from past-only
    windows; the chronological tail of the windows is held out for early
    stopping. Fitting is seeded and deterministic on CPU for a fixed
    environment.
    """

    name = "patchtst"
    version = "1.0.0"

    #: Channel names built from the daily frame.
    CHANNEL_VALUE = "value"
    CHANNEL_ACTIVE_PIXELS = "active_pixels"
    CHANNEL_COVERAGE = "coverage"

    def __init__(self, config: ForecastConfig | None = None, seed: int = 2026):
        self.config = config or ForecastConfig()
        self.seed = int(seed)
        self._model = None
        self.channel_mean: np.ndarray | None = None
        self.channel_std: np.ndarray | None = None
        self.training_history: dict[str, Any] | None = None
        self.channels: list[str] = []
        self.target_channel = 0
        self._last_matrix: np.ndarray | None = None
        self._last_date: pd.Timestamp | None = None

        params = self.config.patchtst
        if params.d_model % params.n_heads != 0:
            raise ValueError(
                f"PatchTST requires d_model ({params.d_model}) divisible by "
                f"n_heads ({params.n_heads})"
            )
        _ = params.n_patches  # validates patch_len <= context_len

    # ------------------------------------------------------------------
    # Channel matrix construction (documented missing-coverage policy)
    # ------------------------------------------------------------------

    def build_channel_matrix(self, daily: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Build the (T, C) channel matrix from a daily AOI/harmonized frame.

        Channels: [value (target, gaps filled with 0.0), active pixels,
        coverage indicator (1.0 observed / 0.0 gap)]. The coverage channel
        keeps the missing-vs-zero distinction explicit for the network.
        """
        daily = daily.reset_index(drop=True)
        value = pd.to_numeric(daily[self.CHANNEL_VALUE], errors="coerce").fillna(0.0)
        value = value.clip(lower=0.0).to_numpy(dtype=float)

        if {"modis_active_pixels", "viirs_active_pixels"}.issubset(daily.columns):
            pixels = (
                pd.to_numeric(daily["modis_active_pixels"], errors="coerce").fillna(0)
                + pd.to_numeric(daily["viirs_active_pixels"], errors="coerce").fillna(0)
            ).to_numpy(dtype=float)
        else:
            pixels = np.zeros(len(daily), dtype=float)

        if COVERAGE_STATUS_COLUMN in daily.columns:
            coverage = (
                daily[COVERAGE_STATUS_COLUMN].astype(str).eq("observed").to_numpy(dtype=float)
            )
        else:
            coverage = (~pd.isna(daily[self.CHANNEL_VALUE])).to_numpy(dtype=float)

        matrix = np.column_stack([value, pixels, coverage])
        channels = [self.CHANNEL_VALUE, self.CHANNEL_ACTIVE_PIXELS, self.CHANNEL_COVERAGE]
        self._last_matrix = matrix
        self._last_date = pd.to_datetime(daily[DATE_COLUMN]).max()
        return matrix, channels

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def make_windows(
        self, series_matrix: np.ndarray, horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a (T, C) series matrix into supervised windows.

        Returns X with shape (N, C, context_len) and y with shape (N, horizon).
        Windows never peek beyond their end index, so no future leakage occurs.
        """
        context_len = self.config.patchtst.context_len
        T, C = series_matrix.shape
        if T < context_len + horizon:
            raise ValueError(
                f"series length {T} is shorter than context_len ({context_len}) + "
                f"horizon ({horizon}); need at least {context_len + horizon} days"
            )
        xs, ys = [], []
        for start in range(0, T - context_len - horizon + 1):
            end = start + context_len
            xs.append(series_matrix[start:end, :].T)
            ys.append(series_matrix[end : end + horizon, self.target_channel])
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    # ------------------------------------------------------------------
    # Torch network
    # ------------------------------------------------------------------

    def _make_net(self):
        torch = require_torch()
        from torch import nn  # noqa: PLC0415

        params = self.config.patchtst
        horizon = self.config.horizon_days
        n_quantiles = len(self.config.quantiles)
        target_channel = self.target_channel

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.patch_embed = nn.Linear(params.patch_len, params.d_model)
                self.pos_embedding = nn.Parameter(
                    torch.zeros(1, params.n_patches, params.d_model)
                )
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=params.d_model,
                    nhead=params.n_heads,
                    dim_feedforward=params.dim_feedforward,
                    dropout=params.dropout,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=params.n_encoder_blocks,
                    norm=nn.LayerNorm(params.d_model),
                    enable_nested_tensor=False,  # incompatible with norm_first=True
                )
                self.head = nn.Linear(params.n_patches * params.d_model, horizon * n_quantiles)
                self.horizon = horizon
                self.n_quantiles = n_quantiles
                self.patch_len = params.patch_len
                self.stride = params.stride
                self.target_channel = target_channel

            def forward(self, x):
                # x: (B, C, L) -> channel-independent encoding (B*C, P, d_model)
                B, C, L = x.shape
                patches = x.unfold(-1, self.patch_len, self.stride)  # (B, C, P, patch_len)
                P = patches.shape[2]
                flat = patches.reshape(B * C, P, self.patch_len)
                embedded = self.patch_embed(flat) + self.pos_embedding[:, :P, :]
                encoded = self.encoder(embedded)
                out = self.head(encoded.reshape(B * C, -1))
                out = out.reshape(B, C, self.horizon, self.n_quantiles)
                return out[:, self.target_channel, :, :]

        net = _Net()
        return net

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, daily: pd.DataFrame) -> PatchTSTForecaster:
        """Train on the daily history. Deterministic for a fixed seed/env."""
        torch = require_torch()
        seed_everything(self.seed)
        torch.manual_seed(self.seed)

        matrix, channels = self.build_channel_matrix(daily)
        self.channels = channels
        horizon = self.config.horizon_days
        X, y = self.make_windows(matrix, horizon)
        if len(X) < 8:
            raise ValueError(
                f"PatchTST needs at least 8 supervised windows; got {len(X)} "
                f"(provide more history or shorten context_len)"
            )

        # Per-channel standardization from the training matrix only.
        self.channel_mean = matrix.mean(axis=0)
        self.channel_std = np.maximum(matrix.std(axis=0), 1e-8)
        Xs = (X - self.channel_mean[None, :, None]) / self.channel_std[None, :, None]
        ys = (y - self.channel_mean[self.target_channel]) / self.channel_std[self.target_channel]

        n_val = max(2, int(len(Xs) * self.config.trainer.validation_fraction))
        n_val = min(n_val, len(Xs) - 4) if len(Xs) > 6 else max(1, len(Xs) // 4)
        X_train, y_train = Xs[:-n_val], ys[:-n_val]
        X_val, y_val = Xs[-n_val:], ys[-n_val:]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = self._make_net().to(device)
        quantiles = torch.tensor(self.config.quantiles, dtype=torch.float32, device=device)

        def pinball(y_true, y_pred):
            # y_true: (B, H); y_pred: (B, H, Q)
            diff = y_true.unsqueeze(-1) - y_pred
            loss = torch.maximum(quantiles * diff, (quantiles - 1.0) * diff)
            return loss.mean()

        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=self.config.trainer.learning_rate,
            weight_decay=self.config.trainer.weight_decay,
        )

        rng = np.random.default_rng(self.seed)
        batch_size = self.config.trainer.batch_size
        patience = self.config.trainer.patience
        best_val = float("inf")
        best_epoch = -1
        best_state: dict | None = None
        train_losses: list[float] = []
        val_losses: list[float] = []

        n_train = len(X_train)
        for epoch in range(self.config.trainer.epochs):
            net.train()
            order = rng.permutation(n_train)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_train, batch_size):
                idx = order[start : start + batch_size]
                xb = torch.tensor(X_train[idx], dtype=torch.float32, device=device)
                yb = torch.tensor(y_train[idx], dtype=torch.float32, device=device)
                optimizer.zero_grad()
                loss = pinball(yb, net(xb))
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            train_losses.append(epoch_loss / max(n_batches, 1))

            net.eval()
            with torch.no_grad():
                vb = torch.tensor(X_val, dtype=torch.float32, device=device)
                vy = torch.tensor(y_val, dtype=torch.float32, device=device)
                val_loss = float(pinball(vy, net(vb)).item())
            val_losses.append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(net.state_dict())
            elif epoch - best_epoch >= patience:
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self._model = net
        self._device = device
        self.training_history = {
            "epochs_run": len(train_losses),
            "best_epoch": int(best_epoch),
            "best_val_loss": best_val,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "n_windows": int(len(Xs)),
            "n_train_windows": int(n_train),
            "n_val_windows": int(n_val),
            "device": str(device),
            "seed": self.seed,
        }
        return self

    def predict(self, daily: pd.DataFrame | None = None) -> dict[str, Any]:
        """Forecast the next ``horizon_days`` from the end of the fitted history.

        ``daily`` optionally supplies/refreshes the history frame (e.g. after
        loading a saved forecaster) before the forward pass.

        Returns a dict with ``point`` (H,) median and ``quantiles`` (H, Q)
        arrays, sorted per step and clipped at zero, plus provenance fields
        (model name/version and, when available, the training history summary).
        """
        if self._model is None:
            raise RuntimeError("PatchTSTForecaster.predict called before fit")
        torch = require_torch()
        if daily is not None:
            matrix, _ = self.build_channel_matrix(daily)
            self._last_matrix = matrix
        matrix = self._last_matrix
        if matrix is None:
            raise RuntimeError("No history matrix available; fit the forecaster first")
        context_len = self.config.patchtst.context_len
        window = matrix[-context_len:, :].T  # (C, L)
        window = (window - self.channel_mean[:, None]) / self.channel_std[:, None]
        x = torch.tensor(window[None, ...], dtype=torch.float32, device=self._device)
        with torch.no_grad():
            raw = self._model(x).cpu().numpy().astype(float)  # (1, H, Q)
        quantiles_sorted = np.sort(raw[0], axis=-1)
        std = self.channel_std[self.target_channel]
        mean = self.channel_mean[self.target_channel]
        physical = quantiles_sorted * std + mean
        physical = np.clip(physical, 0.0, None)
        q_index = list(self.config.quantiles).index(0.5)
        point = physical[:, q_index]
        if self._last_date is None:
            raise RuntimeError("No history dates available; fit or predict(daily) first")
        horizon_dates = pd.date_range(
            self._last_date + pd.Timedelta(days=1), periods=self.config.horizon_days, freq="D"
        )
        return {
            "model_name": self.name,
            "model_version": self.version,
            "point": point,
            "quantiles": physical,
            "quantile_levels": list(self.config.quantiles),
            "horizon_dates": [d.date().isoformat() for d in horizon_dates],
            "training_history": self.training_history,
        }

    # ------------------------------------------------------------------
    # Serialization support (artifact contract)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Export everything needed to restore the forecaster (torch tensors as lists)."""
        if self._model is None:
            raise RuntimeError("state_dict called before fit")
        return {
            "model_name": self.name,
            "model_version": self.version,
            "config": self.config.to_dict(),
            "channels": list(self.channels),
            "target_channel": int(self.target_channel),
            "channel_mean": self.channel_mean.tolist(),
            "channel_std": self.channel_std.tolist(),
            "seed": int(self.seed),
            "training_history": self.training_history,
            "torch_state": {k: v.tolist() for k, v in self._model.state_dict().items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> PatchTSTForecaster:
        """Restore from :meth:`state_dict` output (weights verified by artifact hash)."""
        torch = require_torch()
        self.config = ForecastConfig.from_dict(state["config"])
        self.channels = list(state["channels"])
        self.target_channel = int(state["target_channel"])
        self.channel_mean = np.asarray(state["channel_mean"], dtype=float)
        self.channel_std = np.asarray(state["channel_std"], dtype=float)
        self.seed = int(state["seed"])
        self.training_history = state.get("training_history")
        net = self._make_net()
        net.load_state_dict(
            {k: torch.tensor(np.asarray(v)) for k, v in state["torch_state"].items()}
        )
        net.eval()
        self._model = net
        self._device = next(net.parameters()).device
        return self


__all__ = ["PatchTSTForecaster", "TorchDependencyError", "require_torch"]
