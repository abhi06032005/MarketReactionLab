"""End-to-end integration tests for MarketReactionLab pipeline and PyTorch DataLoader."""

import json
from pathlib import Path
import pytest
import polars as pl
import torch
from src.training.collate import get_market_dataloader
from src.data.validation import PipelineValidator


def test_processed_dataset_files_exist():
    proc_dir = Path("data/processed")
    assert (proc_dir / "train.parquet").exists()
    assert (proc_dir / "validation.parquet").exists()
    assert (proc_dir / "test.parquet").exists()
    assert (proc_dir / "feature_schema.json").exists()
    assert (proc_dir / "label_mapping.json").exists()
    assert (proc_dir / "dataset_manifest.json").exists()


def test_manifest_and_splits_consistency():
    proc_dir = Path("data/processed")
    with open(proc_dir / "dataset_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_df = pl.read_parquet(proc_dir / "train.parquet")
    val_df = pl.read_parquet(proc_dir / "validation.parquet")
    test_df = pl.read_parquet(proc_dir / "test.parquet")

    assert len(train_df) == manifest["splits"]["train"]["count"]
    assert len(val_df) == manifest["splits"]["validation"]["count"]
    assert len(test_df) == manifest["splits"]["test"]["count"]
    assert manifest["total_samples"] == len(train_df) + len(val_df) + len(test_df)


def test_pytorch_dataloader_integration():
    proc_dir = Path("data/processed")
    loader = get_market_dataloader(str(proc_dir / "train.parquet"), batch_size=32, shuffle=False)

    batch = next(iter(loader))
    assert "numerical_features" in batch
    assert "class_labels" in batch
    assert "market_adjusted_returns" in batch
    assert "news_texts" in batch

    # Check tensor shapes
    assert batch["numerical_features"].shape == (32, 19)
    assert batch["class_labels"].shape == (32,)
    assert batch["market_adjusted_returns"].shape == (32,)
    assert len(batch["news_texts"]) == 32

    # Check label classes are valid (0, 1, or 2)
    labels = batch["class_labels"].tolist()
    assert all(l in [0, 1, 2] for l in labels)


def test_pipeline_leakage_audit_clean():
    validator = PipelineValidator(processed_dir="data/processed", reports_dir="reports")
    res = validator.run_leakage_audit()

    assert res["leakage_free"] is True
    assert res["target_strictly_after_reference"] is True
    assert res["chronological_split_isolated"] is True
    assert res["train_val_id_overlap"] == 0
    assert res["val_test_id_overlap"] == 0
