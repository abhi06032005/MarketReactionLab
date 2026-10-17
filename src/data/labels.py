"""Training label computation and market adjustment module.

Computes raw returns, benchmark (SPY) returns, next-session market-adjusted returns,
and three-class (UP/NEUTRAL/DOWN) target labels with outcome validation.
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.005  # 0.5% market-adjusted return boundary
LABEL_MAPPING = {
    "DOWN": 0,
    "NEUTRAL": 1,
    "UP": 2,
}
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}


class LabelComputer:
    """Computes next-session market-adjusted returns and discrete classification labels."""

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        benchmark_symbol: str = "SPY",
    ):
        self.threshold = threshold
        self.benchmark_symbol = benchmark_symbol

    def compute_event_labels(
        self,
        df_events: pl.DataFrame,
        df_prices: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Computes reference close, next-session close, raw return, benchmark return,
        market-adjusted return, and discrete class_label for each event.

        df_events must have columns: ['ticker', 'reference_session', 'target_session']
        df_prices must have columns: ['ticker', 'date', 'close']
        """
        # Build price lookup table: (ticker, date) -> close
        price_lookup = {
            (row["ticker"], row["date"]): float(row["close"])
            for row in df_prices.select(["ticker", "date", "close"]).iter_rows(named=True)
        }

        # Build benchmark lookup table: date -> close
        benchmark_lookup = {
            row["date"]: float(row["close"])
            for row in df_prices.filter(pl.col("ticker") == self.benchmark_symbol)
            .select(["date", "close"])
            .iter_rows(named=True)
        }

        ref_closes = []
        target_closes = []
        raw_returns = []
        bench_returns = []
        adj_returns = []
        class_labels = []
        class_names = []
        outcome_valid = []

        for row in df_events.iter_rows(named=True):
            sym = row["ticker"]
            ref_date = row["reference_session"]
            tgt_date = row["target_session"]

            p_ref = price_lookup.get((sym, ref_date))
            p_tgt = price_lookup.get((sym, tgt_date))
            b_ref = benchmark_lookup.get(ref_date)
            b_tgt = benchmark_lookup.get(tgt_date)

            if (
                p_ref is not None and p_ref > 0
                and p_tgt is not None and p_tgt > 0
                and b_ref is not None and b_ref > 0
                and b_tgt is not None and b_tgt > 0
            ):
                raw_ret = (p_tgt - p_ref) / p_ref
                bench_ret = (b_tgt - b_ref) / b_ref
                adj_ret = raw_ret - bench_ret

                if adj_ret > self.threshold:
                    label = 2
                    cname = "UP"
                elif adj_ret < -self.threshold:
                    label = 0
                    cname = "DOWN"
                else:
                    label = 1
                    cname = "NEUTRAL"

                ref_closes.append(round(p_ref, 4))
                target_closes.append(round(p_tgt, 4))
                raw_returns.append(round(raw_ret, 6))
                bench_returns.append(round(bench_ret, 6))
                adj_returns.append(round(adj_ret, 6))
                class_labels.append(label)
                class_names.append(cname)
                outcome_valid.append(True)
            else:
                ref_closes.append(None)
                target_closes.append(None)
                raw_returns.append(None)
                bench_returns.append(None)
                adj_returns.append(None)
                class_labels.append(None)
                class_names.append(None)
                outcome_valid.append(False)

        labeled_df = df_events.with_columns([
            pl.Series("reference_close", ref_closes),
            pl.Series("target_close", target_closes),
            pl.Series("raw_return", raw_returns),
            pl.Series("benchmark_return", bench_returns),
            pl.Series("market_adjusted_return", adj_returns),
            pl.Series("class_label", class_labels, dtype=pl.Int32),
            pl.Series("class_name", class_names),
            pl.Series("outcome_valid", outcome_valid),
        ])

        valid_count = sum(outcome_valid)
        logger.info(
            f"Computed labels for {len(df_events):,} events: "
            f"{valid_count:,} valid outcomes ({len(df_events) - valid_count:,} missing price sessions)."
        )
        return labeled_df

    @staticmethod
    def get_distribution_summary(df: pl.DataFrame) -> Dict[str, Any]:
        """Calculates class balance, percentages, and return statistics."""
        valid_df = df.filter(pl.col("outcome_valid") == True)
        total = len(valid_df)
        if total == 0:
            return {"total_valid": 0}

        counts = valid_df["class_name"].value_counts().to_dicts()
        dist = {c["class_name"]: c["count"] for c in counts}
        pcts = {k: round(v / total * 100, 2) for k, v in dist.items()}

        ret_series = valid_df["market_adjusted_return"].to_numpy()
        return {
            "total_valid_events": total,
            "counts": dist,
            "percentages": pcts,
            "mean_adjusted_return": round(float(np.mean(ret_series)), 6),
            "median_adjusted_return": round(float(np.median(ret_series)), 6),
            "std_adjusted_return": round(float(np.std(ret_series)), 6),
            "min_adjusted_return": round(float(np.min(ret_series)), 6),
            "max_adjusted_return": round(float(np.max(ret_series)), 6),
        }


def compute_market_adjusted_label(
    stock_return: float,
    benchmark_return: float,
    threshold: float = 0.005,
) -> Tuple[int, str, float]:
    """Computes market-adjusted return and returns (class_label, class_name, adjusted_return)."""
    adj_ret = stock_return - benchmark_return
    if adj_ret > threshold:
        return 2, "UP", adj_ret
    elif adj_ret < -threshold:
        return 0, "DOWN", adj_ret
    else:
        return 1, "NEUTRAL", adj_ret


def analyze_class_distribution(df: pl.DataFrame, label_col: str = "class_label") -> Dict[str, Any]:
    """Analyzes class balance and returns summary dictionary."""
    total = len(df)
    counts = {}
    percentages = {}
    for row in df[label_col].value_counts().iter_rows(named=True):
        lbl = row[label_col]
        name = REVERSE_LABEL_MAPPING.get(lbl, str(lbl))
        cnt = row["count"]
        counts[name] = cnt
        percentages[name] = round((cnt / total) * 100.0, 2)
    return {
        "total": total,
        "counts": counts,
        "percentages": percentages,
        "is_imbalanced": any(p < 15.0 or p > 70.0 for p in percentages.values()),
    }
