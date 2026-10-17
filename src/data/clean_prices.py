"""Price cleaning, validation, and schema normalization module."""

import os
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import polars as pl

logger = logging.getLogger(__name__)


class PriceCleaner:
    """Validates and standardizes OHLCV candlestick records for equities and market benchmark."""

    def __init__(self, raw_prices_dir: str = "data/raw/prices"):
        self.raw_prices_dir = Path(raw_prices_dir)
        self.rejection_reasons: Dict[str, int] = {
            "unparseable_row": 0,
            "non_positive_price": 0,
            "high_less_than_max": 0,
            "low_greater_than_min": 0,
            "negative_volume": 0,
            "duplicate_symbol_date": 0,
        }

    def clean_single_symbol(self, symbol: str) -> pl.DataFrame:
        """Loads and validates historical daily OHLCV CSV for a single symbol."""
        csv_path = self.raw_prices_dir / f"{symbol}.csv"
        if not csv_path.exists():
            logger.warning(f"Price CSV for {symbol} not found at {csv_path}")
            return pl.DataFrame()

        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return pl.DataFrame()

            h_lower = [c.lower().strip() for c in header]
            try:
                col_date = h_lower.index("date")
                col_open = h_lower.index("open")
                col_high = h_lower.index("high")
                col_low = h_lower.index("low")
                col_close = h_lower.index("close")
                col_volume = h_lower.index("volume")
                col_adj_close = h_lower.index("adj close") if "adj close" in h_lower else None
            except ValueError as e:
                logger.error(f"Missing essential OHLCV column in {csv_path}: {e}")
                return pl.DataFrame()

            valid_rows = []
            seen_dates = set()

            for row in reader:
                if len(row) <= max(col_date, col_open, col_high, col_low, col_close, col_volume):
                    self.rejection_reasons["unparseable_row"] += 1
                    continue

                d_str = row[col_date].strip()
                if not d_str or len(d_str) < 10:
                    self.rejection_reasons["unparseable_row"] += 1
                    continue

                try:
                    open_p = float(row[col_open])
                    high_p = float(row[col_high])
                    low_p = float(row[col_low])
                    close_p = float(row[col_close])
                    volume = float(row[col_volume])
                    adj_close_p = float(row[col_adj_close]) if col_adj_close is not None else close_p
                except ValueError:
                    self.rejection_reasons["unparseable_row"] += 1
                    continue

                # Mathematical validity checks
                if open_p <= 0 or high_p <= 0 or low_p <= 0 or close_p <= 0:
                    self.rejection_reasons["non_positive_price"] += 1
                    continue

                if volume < 0:
                    self.rejection_reasons["negative_volume"] += 1
                    continue

                eps = 1e-4
                if high_p < max(open_p, close_p, low_p) - eps:
                    self.rejection_reasons["high_less_than_max"] += 1
                    continue

                if low_p > min(open_p, close_p, high_p) + eps:
                    self.rejection_reasons["low_greater_than_min"] += 1
                    continue

                # Duplicate date check within symbol
                date_norm = d_str[:10]
                if date_norm in seen_dates:
                    self.rejection_reasons["duplicate_symbol_date"] += 1
                    continue
                seen_dates.add(date_norm)

                valid_rows.append({
                    "ticker": symbol.upper(),
                    "date": date_norm,
                    "open": round(open_p, 4),
                    "high": round(high_p, 4),
                    "low": round(low_p, 4),
                    "close": round(close_p, 4),
                    "adj_close": round(adj_close_p, 4),
                    "volume": float(volume),
                })

        if not valid_rows:
            return pl.DataFrame()

        df = pl.DataFrame(valid_rows)
        # Sort chronologically ascending
        df = df.sort("date")
        return df

    def clean_all(self, symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """Cleans price records for all available symbols and concatenates."""
        if symbols is None:
            csv_files = list(self.raw_prices_dir.glob("*.csv"))
            symbols = [p.stem for p in csv_files]

        dfs = []
        for sym in symbols:
            df_sym = self.clean_single_symbol(sym)
            if len(df_sym) > 0:
                dfs.append(df_sym)

        if not dfs:
            return pl.DataFrame()

        full_df = pl.concat(dfs).sort(["ticker", "date"])
        logger.info(
            f"Prices cleaned: {len(full_df):,} daily records across {full_df['ticker'].n_unique()} symbols. "
            f"Rejections: {self.rejection_reasons}"
        )
        return full_df


def clean_and_save_prices(
    raw_dir: str = "data/raw/prices",
    out_path: str = "data/interim/clean_prices.parquet"
) -> pl.DataFrame:
    """Cleans all price CSVs and saves combined interim parquet file."""
    cleaner = PriceCleaner(raw_dir)
    df = cleaner.clean_all()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.write_parquet(out_path)
    logger.info(f"Clean prices saved to {out_path} ({len(df):,} records).")
    return df


if __name__ == "__main__":
    clean_and_save_prices()
