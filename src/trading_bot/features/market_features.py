"""Leak-safe market feature engineering from stored OHLCV candles."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from trading_bot.logging_config import get_logger
from trading_bot.schemas.features import (
    INDEX_COLUMNS,
    get_expected_market_feature_columns,
    validate_required_columns,
)

logger = get_logger(__name__)

_SUPPORTED_WINDOWS: tuple[int, ...] = (1, 2, 4, 8)


def compute_market_features(
    df: pd.DataFrame,
    windows: Sequence[int] | None = None,
    *,
    include_breakout_features: bool = True,
    include_calendar_features: bool = True,
) -> pd.DataFrame:
    """Build deterministic market features from candle history only.

    Every feature at timestamp ``t`` uses only candle information available at
    or before ``t``. Early rows preserve ``NaN`` values when there is not
    enough history to compute a rolling statistic.
    """
    validate_required_columns(
        df,
        [
            "symbol",
            "timeframe",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
        dataset_name="market candle data",
    )

    if df.empty:
        return df[INDEX_COLUMNS].copy()

    enabled_windows = _resolve_windows(windows)
    required_rolling_windows = [window for window in enabled_windows if window >= 4]

    market = df.copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    market = market.sort_values(INDEX_COLUMNS).reset_index(drop=True)

    logger.info(
        "market_features_start",
        rows=len(market),
        windows=enabled_windows,
        include_breakout_features=include_breakout_features,
        include_calendar_features=include_calendar_features,
    )

    group_keys = ["symbol", "timeframe"]
    grouped = market.groupby(group_keys, sort=False, group_keys=False)

    close = market["close"].astype(float)
    open_ = market["open"].astype(float)
    high = market["high"].astype(float)
    low = market["low"].astype(float)
    volume = market["volume"].astype(float)

    candle_range = high - low
    safe_close = close.replace(0.0, np.nan)
    safe_range = candle_range.replace(0.0, np.nan)

    features = market[INDEX_COLUMNS].copy()

    for window in enabled_windows:
        features[f"ret_{window}"] = grouped["close"].pct_change(window)

    if 1 in enabled_windows:
        features["log_ret_1"] = np.log(close / grouped["close"].shift(1))
        features["volume_change_1"] = grouped["volume"].pct_change(1)
    else:
        features["log_ret_1"] = np.nan
        features["volume_change_1"] = np.nan

    features["range_pct"] = candle_range / safe_close
    features["body_pct"] = (close - open_) / safe_range
    features["upper_wick_pct"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / safe_range
    features["lower_wick_pct"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / safe_range
    features["high_low_spread"] = (high / low.replace(0.0, np.nan)) - 1.0

    ret_1_series = (
        features["ret_1"] if "ret_1" in features else pd.Series(np.nan, index=market.index)
    )
    for window in required_rolling_windows:
        rolling_returns = ret_1_series.groupby([market["symbol"], market["timeframe"]]).rolling(
            window=window,
            min_periods=window,
        )
        features[f"rolling_volatility_{window}"] = rolling_returns.std().reset_index(
            level=[0, 1],
            drop=True,
        )
        features[f"rolling_mean_return_{window}"] = rolling_returns.mean().reset_index(
            level=[0, 1],
            drop=True,
        )

        rolling_volume = grouped["volume"].rolling(window=window, min_periods=window)
        features[f"rolling_volume_mean_{window}"] = rolling_volume.mean().reset_index(
            level=[0, 1],
            drop=True,
        )

        rolling_sma = grouped["close"].rolling(window=window, min_periods=window).mean()
        rolling_sma_series = rolling_sma.reset_index(level=[0, 1], drop=True)
        features[f"close_vs_sma_{window}"] = (close - rolling_sma_series) / rolling_sma_series

    if 8 in enabled_windows:
        rolling_volume_mean_8 = grouped["volume"].rolling(window=8, min_periods=8).mean()
        rolling_volume_std_8 = grouped["volume"].rolling(window=8, min_periods=8).std()
        volume_mean_series = rolling_volume_mean_8.reset_index(level=[0, 1], drop=True)
        volume_std_series = rolling_volume_std_8.reset_index(level=[0, 1], drop=True)
        features["volume_zscore_8"] = (volume - volume_mean_series) / volume_std_series.replace(
            0.0,
            np.nan,
        )
    else:
        features["volume_zscore_8"] = np.nan

    if include_breakout_features and 8 in enabled_windows:
        rolling_max_8 = grouped["close"].rolling(window=8, min_periods=8).max()
        rolling_min_8 = grouped["close"].rolling(window=8, min_periods=8).min()
        max_close_series = rolling_max_8.reset_index(level=[0, 1], drop=True)
        min_close_series = rolling_min_8.reset_index(level=[0, 1], drop=True)

        features["rolling_max_close_8"] = max_close_series
        features["rolling_min_close_8"] = min_close_series
        features["breakout_distance_high_8"] = (close - max_close_series) / max_close_series
        features["breakout_distance_low_8"] = (close - min_close_series) / min_close_series

    if include_calendar_features:
        features["hour_of_day"] = market["timestamp"].dt.hour
        features["day_of_week"] = market["timestamp"].dt.dayofweek

    output_columns = INDEX_COLUMNS + get_expected_market_feature_columns(
        include_breakout_features=include_breakout_features,
        include_calendar_features=include_calendar_features,
    )
    result = features[output_columns].copy()

    logger.info(
        "market_features_complete",
        rows=len(result),
        feature_columns=len(result.columns) - len(INDEX_COLUMNS),
        start_timestamp=result["timestamp"].min().isoformat(),
        end_timestamp=result["timestamp"].max().isoformat(),
    )

    return result


def _resolve_windows(windows: Sequence[int] | None) -> list[int]:
    """Return supported windows in stable ascending order.

    The step-4 schema depends on the canonical 1/2/4/8 windows, so we always
    include them even if the config omits one.
    """
    requested = set(windows or _SUPPORTED_WINDOWS) | set(_SUPPORTED_WINDOWS)
    resolved = [window for window in _SUPPORTED_WINDOWS if window in requested]
    return resolved if resolved else list(_SUPPORTED_WINDOWS)
