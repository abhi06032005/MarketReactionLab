"""Market feature engineering and technical indicator computation module.

Computes strictly backward-looking candlestick indicators, rolling statistics,
and broad-market context relative to SPY. No future data is used.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "rolling_vol_20d",
    "rolling_avg_vol_20d",
    "relative_volume",
    "rolling_hl_range_20d",
    "candle_body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "rel_perf_spy_5d",
    "rel_perf_spy_20d",
    "spy_ret_20d",
    "spy_vol_20d",
]


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Computes standard Wilder-smoothed Relative Strength Index (RSI).
    Returns array of same length as prices, with NaN for initial uncomputable elements.
    """
    n = len(prices)
    rsi = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return rsi

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        idx = i + 1
        if avg_loss == 0.0:
            rsi[idx] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[idx] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def compute_ema(series: np.ndarray, span: int) -> np.ndarray:
    """Computes Exponential Moving Average with span."""
    n = len(series)
    ema = np.full(n, np.nan, dtype=np.float64)
    if n < span:
        return ema

    alpha = 2.0 / (span + 1.0)
    # Start with simple moving average of first 'span' values
    ema[span - 1] = np.mean(series[:span])
    for i in range(span, n):
        ema[i] = alpha * series[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def compute_macd(
    prices: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes MACD line, signal line, and MACD histogram."""
    n = len(prices)
    macd = np.full(n, np.nan, dtype=np.float64)
    sig = np.full(n, np.nan, dtype=np.float64)
    hist = np.full(n, np.nan, dtype=np.float64)

    if n < slow + signal:
        return macd, sig, hist

    ema_fast = compute_ema(prices, fast)
    ema_slow = compute_ema(prices, slow)
    macd_raw = ema_fast - ema_slow

    # Compute signal line on valid MACD elements
    valid_mask = ~np.isnan(macd_raw)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) >= signal:
        macd_valid = macd_raw[valid_idx]
        sig_valid = compute_ema(macd_valid, signal)
        macd[valid_idx] = macd_valid
        sig[valid_idx] = sig_valid
        hist[valid_idx] = macd_valid - sig_valid

    return macd, sig, hist


class MarketFeatureComputer:
    """Computes candlestick and market features for historical price sequences."""

    def __init__(self, benchmark_symbol: str = "SPY"):
        self.benchmark_symbol = benchmark_symbol

    def compute_symbol_features(self, df_sym: pl.DataFrame) -> pl.DataFrame:
        """
        Computes rolling candlestick and indicator features for a single symbol's price history.
        Assumes df_sym is sorted chronologically ascending by 'date'.
        """
        if len(df_sym) == 0:
            return df_sym

        c = df_sym["close"].to_numpy().astype(np.float64)
        o = df_sym["open"].to_numpy().astype(np.float64)
        h = df_sym["high"].to_numpy().astype(np.float64)
        l = df_sym["low"].to_numpy().astype(np.float64)
        v = df_sym["volume"].to_numpy().astype(np.float64)
        n = len(c)

        # Intraday candlestick shape metrics
        hl_range = h - l
        eps = 1e-6
        denom = hl_range + eps

        candle_body_ratio = np.abs(c - o) / denom
        upper_wick_ratio = (h - np.maximum(o, c)) / denom
        lower_wick_ratio = (np.minimum(o, c) - l) / denom

        # Return lags
        def calc_ret(lag: int) -> np.ndarray:
            ret = np.full(n, np.nan, dtype=np.float64)
            if n > lag:
                ret[lag:] = (c[lag:] - c[:-lag]) / c[:-lag]
            return ret

        ret_1d = calc_ret(1)
        ret_3d = calc_ret(3)
        ret_5d = calc_ret(5)
        ret_10d = calc_ret(10)
        ret_20d = calc_ret(20)

        # Rolling 20-day volatility of 1-day returns (sample std)
        rolling_vol_20d = np.full(n, np.nan, dtype=np.float64)
        for i in range(20, n):
            window = ret_1d[i - 19 : i + 1]
            if not np.any(np.isnan(window)):
                rolling_vol_20d[i] = np.std(window, ddof=1)

        # Rolling 20-day average volume & relative volume
        rolling_avg_vol_20d = np.full(n, np.nan, dtype=np.float64)
        for i in range(20, n):
            rolling_avg_vol_20d[i] = np.mean(v[i - 19 : i + 1])
        relative_volume = np.where(
            np.isnan(rolling_avg_vol_20d),
            np.nan,
            v / (rolling_avg_vol_20d + eps),
        )

        # Rolling 20-day high-low normalized range
        hl_pct = (h - l) / c
        rolling_hl_range_20d = np.full(n, np.nan, dtype=np.float64)
        for i in range(20, n):
            rolling_hl_range_20d[i] = np.mean(hl_pct[i - 19 : i + 1])

        # RSI(14)
        rsi_14 = compute_rsi(c, period=14)

        # MACD(12, 26, 9)
        macd, macd_signal, macd_hist = compute_macd(c, fast=12, slow=26, signal=9)

        # Append computed series
        computed = df_sym.with_columns([
            pl.Series("ret_1d", ret_1d),
            pl.Series("ret_3d", ret_3d),
            pl.Series("ret_5d", ret_5d),
            pl.Series("ret_10d", ret_10d),
            pl.Series("ret_20d", ret_20d),
            pl.Series("rolling_vol_20d", rolling_vol_20d),
            pl.Series("rolling_avg_vol_20d", rolling_avg_vol_20d),
            pl.Series("relative_volume", relative_volume),
            pl.Series("rolling_hl_range_20d", rolling_hl_range_20d),
            pl.Series("candle_body_ratio", candle_body_ratio),
            pl.Series("upper_wick_ratio", upper_wick_ratio),
            pl.Series("lower_wick_ratio", lower_wick_ratio),
            pl.Series("rsi_14", rsi_14),
            pl.Series("macd", macd),
            pl.Series("macd_signal", macd_signal),
            pl.Series("macd_hist", macd_hist),
        ])
        return computed

    def compute_all_with_benchmark(self, df_prices: pl.DataFrame) -> pl.DataFrame:
        """
        Computes features for all symbols including relative performance against SPY benchmark.
        """
        # First process SPY separately
        df_spy = df_prices.filter(pl.col("ticker") == self.benchmark_symbol).sort("date")
        if len(df_spy) == 0:
            raise ValueError(f"Benchmark {self.benchmark_symbol} not found in price dataset!")

        spy_features = self.compute_symbol_features(df_spy)
        spy_ref = spy_features.select([
            pl.col("date"),
            pl.col("ret_5d").alias("spy_ret_5d"),
            pl.col("ret_20d").alias("spy_ret_20d"),
            pl.col("rolling_vol_20d").alias("spy_vol_20d"),
        ])

        # Process each symbol
        all_symbols = df_prices["ticker"].unique().to_list()
        processed_dfs = []

        for sym in all_symbols:
            df_sym = df_prices.filter(pl.col("ticker") == sym).sort("date")
            feats_sym = self.compute_symbol_features(df_sym)

            # Join benchmark context on date
            joined = feats_sym.join(spy_ref, on="date", how="left")

            # Relative performance
            joined = joined.with_columns([
                (pl.col("ret_5d") - pl.col("spy_ret_5d")).alias("rel_perf_spy_5d"),
                (pl.col("ret_20d") - pl.col("spy_ret_20d")).alias("rel_perf_spy_20d"),
            ])
            processed_dfs.append(joined)

        full_features = pl.concat(processed_dfs).sort(["ticker", "date"])
        logger.info(
            f"Computed features for {len(full_symbols := all_symbols)} symbols, "
            f"total {len(full_features):,} rows."
        )
        return full_features


def build_market_features(
    clean_prices_path: str = "data/interim/clean_prices.parquet",
    benchmark_symbol: str = "SPY"
) -> pl.DataFrame:
    """Convenience function to load prices, compute features, and return DataFrame."""
    df_prices = pl.read_parquet(clean_prices_path)
    computer = MarketFeatureComputer(benchmark_symbol=benchmark_symbol)
    return computer.compute_all_with_benchmark(df_prices)


def compute_market_features(pdf):
    """Convenience helper to compute market features directly on a pandas price DataFrame."""
    import polars as pl
    lower_df = pdf.copy()
    lower_df.columns = [c.lower() for c in lower_df.columns]
    computer = MarketFeatureComputer()
    pl_df = pl.from_pandas(lower_df)
    feats = computer.compute_symbol_features(pl_df).to_pandas()
    for col in FEATURE_COLUMNS:
        if col in feats.columns:
            pdf[f"feat_{col}"] = feats[col].values
    return pdf

