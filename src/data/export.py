"""Dataset export module: saves training-ready Parquets, feature schemas, and manifests."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import polars as pl
from datetime import datetime

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "rolling_vol_20d",
    "relative_volume",
    "rolling_hl_range_20d",
    "candle_body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "rel_perf_spy_20d",
    "spy_ret_5d",
    "spy_ret_20d",
    "spy_vol_20d",
]

LABEL_MAPPING = {
    "DOWN": 0,
    "NEUTRAL": 1,
    "UP": 2
}


def export_processed_dataset(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    output_dir: str = "data/processed",
    threshold: float = 0.005,
    benchmark_symbol: str = "SPY"
) -> Dict[str, str]:
    """Exports train/validation/test parquet files, feature schema, and dataset manifest."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_file = out_path / "train.parquet"
    val_file = out_path / "validation.parquet"
    test_file = out_path / "test.parquet"
    schema_file = out_path / "feature_schema.json"
    label_file = out_path / "label_mapping.json"
    manifest_file = out_path / "dataset_manifest.json"

    logger.info("Writing processed parquet splits...")
    train_df.write_parquet(train_file)
    val_df.write_parquet(val_file)
    test_df.write_parquet(test_file)

    # Feature Schema
    feature_schema = {
        "version": "1.0.0",
        "numerical_features_count": len(FEATURE_NAMES),
        "numerical_feature_names": FEATURE_NAMES,
        "text_field": "news_text",
        "id_field": "sample_id",
        "ticker_field": "ticker",
        "date_field": "event_date",
        "target_return_field": "market_adjusted_return",
        "target_label_field": "class_label",
        "threshold": threshold,
        "benchmark_symbol": benchmark_symbol,
    }
    with open(schema_file, "w", encoding="utf-8") as f:
        json.dump(feature_schema, f, indent=2)

    # Label Mapping
    with open(label_file, "w", encoding="utf-8") as f:
        json.dump(LABEL_MAPPING, f, indent=2)

    # Dataset Manifest
    total_samples = len(train_df) + len(val_df) + len(test_df)
    symbols = sorted(list(set(train_df["ticker"].unique().to_list() + val_df["ticker"].unique().to_list() + test_df["ticker"].unique().to_list())))

    manifest = {
        "dataset_name": "MarketReactionLab_Phase1",
        "generated_at": datetime.now().isoformat(),
        "total_samples": total_samples,
        "symbols_count": len(symbols),
        "symbols": symbols,
        "splits": {
            "train": {
                "count": len(train_df),
                "earliest_date": train_df["event_date"].min() if len(train_df) > 0 else "",
                "latest_date": train_df["event_date"].max() if len(train_df) > 0 else "",
                "file": str(train_file),
            },
            "validation": {
                "count": len(val_df),
                "earliest_date": val_df["event_date"].min() if len(val_df) > 0 else "",
                "latest_date": val_df["event_date"].max() if len(val_df) > 0 else "",
                "file": str(val_file),
            },
            "test": {
                "count": len(test_df),
                "earliest_date": test_df["event_date"].min() if len(test_df) > 0 else "",
                "latest_date": test_df["event_date"].max() if len(test_df) > 0 else "",
                "file": str(test_file),
            },
        },
        "target": {
            "task": "Next-session market-adjusted direction classification (UP/NEUTRAL/DOWN)",
            "benchmark": benchmark_symbol,
            "threshold": threshold,
            "label_mapping": LABEL_MAPPING,
        },
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest saved to {manifest_file}")
    return {
        "train": str(train_file),
        "validation": str(val_file),
        "test": str(test_file),
        "feature_schema": str(schema_file),
        "label_mapping": str(label_file),
        "dataset_manifest": str(manifest_file),
    }
