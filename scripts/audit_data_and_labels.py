"""Audit Script 1: Forensic verification of samples, schemas, provenance, labels, and text leakage."""

import json
from pathlib import Path
import polars as pl
import numpy as np

def run_audit_data_and_labels():
    proc_dir = Path("data/processed")
    train_path = proc_dir / "train.parquet"
    val_path = proc_dir / "validation.parquet"
    test_path = proc_dir / "test.parquet"
    manifest_path = proc_dir / "dataset_manifest.json"

    print("=== CHECK 1: Sample Counts, Schemas, Provenance & Splits ===")
    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)
    test_df = pl.read_parquet(test_path)

    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    total = n_train + n_val + n_test
    print(f"Sample Counts: Train={n_train:,}, Val={n_val:,}, Test={n_test:,}, Total={total:,}")

    # Check manifest consistency
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"Manifest reported total: {manifest.get('total_samples')}")
    print(f"Manifest splits: {manifest.get('splits')}")

    # Chronological boundaries
    print("\nDate Ranges (event_date):")
    print(f"  Train: {train_df['event_date'].min()} to {train_df['event_date'].max()}")
    print(f"  Val:   {val_df['event_date'].min()} to {val_df['event_date'].max()}")
    print(f"  Test:  {test_df['event_date'].min()} to {test_df['event_date'].max()}")

    # Check for split overlap
    train_ids = set(train_df["sample_id"].to_list())
    val_ids = set(val_df["sample_id"].to_list())
    test_ids = set(test_df["sample_id"].to_list())
    overlap_tv = len(train_ids & val_ids)
    overlap_vt = len(val_ids & test_ids)
    overlap_tt = len(train_ids & test_ids)
    print(f"Sample ID Overlap: Train/Val={overlap_tv}, Val/Test={overlap_vt}, Train/Test={overlap_tt}")

    # Check for date leakage across splits
    leak_tv = (train_df["event_date"].max() >= val_df["event_date"].min())
    leak_vt = (val_df["event_date"].max() >= test_df["event_date"].min())
    print(f"Date Boundary Leak: Train/Val={leak_tv}, Val/Test={leak_vt}")

    # Check schema columns
    print("\nParquet Schema Columns:")
    print(train_df.columns)

    print("\n=== CHECK 2: Independent Recalculation of Labels from Raw Prices ===")
    clean_prices = pl.read_parquet("data/interim/clean_prices.parquet")
    price_lookup = {
        (row["ticker"], row["date"]): float(row["close"])
        for row in clean_prices.select(["ticker", "date", "close"]).iter_rows(named=True)
    }

    # Verify on sample of 1000 records from test and train
    full_df = pl.concat([train_df, val_df, test_df])
    mismatches = 0
    max_return_diff = 0.0
    label_mismatches = 0

    threshold = 0.005
    for row in full_df.iter_rows(named=True):
        sym = row["ticker"]
        ref_d = row["reference_session"]
        tgt_d = row["target_session"]

        p_ref = price_lookup.get((sym, ref_d))
        p_tgt = price_lookup.get((sym, tgt_d))
        spy_ref = price_lookup.get(("SPY", ref_d))
        spy_tgt = price_lookup.get(("SPY", tgt_d))

        if not (p_ref and p_tgt and spy_ref and spy_tgt):
            continue

        raw_calc = (p_tgt / p_ref) - 1.0
        spy_calc = (spy_tgt / spy_ref) - 1.0
        adj_calc = raw_calc - spy_calc

        if adj_calc > threshold:
            lbl_calc = 2
        elif adj_calc < -threshold:
            lbl_calc = 0
        else:
            lbl_calc = 1

        ret_diff = abs(adj_calc - row["market_adjusted_return"])
        if ret_diff > max_return_diff:
            max_return_diff = ret_diff
        if ret_diff > 1e-4:
            mismatches += 1
        if lbl_calc != row["class_label"]:
            label_mismatches += 1

    print(f"Total rows audited: {len(full_df):,}")
    print(f"Max return difference: {max_return_diff:.8f}")
    print(f"Return mismatches (>1e-4): {mismatches}")
    print(f"Label mismatches: {label_mismatches}")

    print("\n=== CHECK 3: News Headline Leakage & Retrospective Summaries Audit ===")
    movement_patterns = [
        "biggest movers", "stocks that hit", "shares are trading",
        "shares drop", "shares surge", "stocks moving in", "mid-day session",
        "market wrap", "stocks to watch", "why .* shares are", "options trading"
    ]
    import re
    compiled = [re.compile(p, re.IGNORECASE) for p in movement_patterns]

    flagged_samples = []
    for row in full_df.iter_rows(named=True):
        text = row["news_text"]
        for regex in compiled:
            if regex.search(text):
                flagged_samples.append((row["sample_id"], row["event_date"], text[:100]))
                break

    print(f"Total samples with retrospective / movement-only headlines: {len(flagged_samples):,} / {len(full_df):,} ({len(flagged_samples)/len(full_df)*100:.2f}%)")
    if flagged_samples:
        print("Sample flagged headlines:")
        for sid, dt, snip in flagged_samples[:5]:
            print(f"  [{sid}] ({dt}): {snip}")

    print("\n=== CHECK 4: Temporal Feature Precedence Verification ===")
    # Check whether target_session > reference_session for 100% of rows
    inv_session = full_df.filter(pl.col("target_session") <= pl.col("reference_session"))
    print(f"Target session <= Reference session violations: {len(inv_session)}")

    # Check whether reference_session <= event_date
    inv_ref = full_df.filter(pl.col("reference_session") > pl.col("event_date"))
    print(f"Reference session > Event date violations: {len(inv_ref)}")

if __name__ == "__main__":
    run_audit_data_and_labels()
