"""Master command-line interface and orchestrator for MarketReactionLab Phase 1 pipeline."""

import os
import sys
import yaml
import logging
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

from src.data.inspect_sources import run_source_audit
from src.data.download import TargetedDownloader
from src.data.clean_news import clean_and_save_news
from src.data.clean_prices import clean_and_save_prices
from src.data.market_features import build_market_features
from src.data.time_alignment import align_and_build_samples
from src.data.splits import create_chronological_splits
from src.data.export import export_processed_dataset
from src.data.validation import PipelineValidator

app = typer.Typer(help="MarketReactionLab Data Engineering Pipeline")
console = Console()
logger = logging.getLogger("marketreactionlab")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@app.command()
def inspect(output_report: str = "reports/source_audit.md"):
    """Audits remote Hugging Face dataset files, schemas, and sizes without full downloads."""
    console.print("[bold cyan]=== MarketReactionLab: Source Inspection ===[/bold cyan]")
    report = run_source_audit(output_path=output_report)
    console.print(f"[bold green]Source audit completed! Written to {output_report}[/bold green]")


@app.command()
def download(
    config: str = "configs/data.yaml",
    dry_run: bool = False,
    sample: bool = False,
    limit_symbols: int = typer.Option(None, help="Limit number of symbols to download")
):
    """Selectively downloads prices and news records for configured symbols."""
    console.print("[bold cyan]=== MarketReactionLab: Targeted Download ===[/bold cyan]")
    limit = 5 if sample and not limit_symbols else limit_symbols
    downloader = TargetedDownloader(config_path=config)
    res = downloader.run_all(limit_symbols=limit, dry_run=dry_run)
    console.print(f"[bold green]Download finished: {res}[/bold green]")


@app.command()
def prepare(
    config: str = "configs/data.yaml",
    threshold: float = 0.005,
    sample: bool = False
):
    """Processes, aligns, and exports training-ready Parquet datasets."""
    console.print("[bold cyan]=== MarketReactionLab: Data Preparation & Alignment ===[/bold cyan]")
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Clean News
    console.print("[yellow][1/5] Cleaning raw news records...[/yellow]")
    clean_news_df = clean_and_save_news(
        raw_path=os.path.join(cfg["paths"]["raw_news_dir"], "raw_news_all.csv"),
        out_path=os.path.join(cfg["paths"]["interim_dir"], "clean_news.parquet")
    )

    # 2. Clean Prices
    console.print("[yellow][2/5] Cleaning and validating OHLCV prices...[/yellow]")
    clean_prices_df = clean_and_save_prices(
        raw_dir=cfg["paths"]["raw_prices_dir"],
        out_path=os.path.join(cfg["paths"]["interim_dir"], "clean_prices.parquet")
    )

    # 3. Market Features
    console.print("[yellow][3/5] Engineering derived technical and market features...[/yellow]")
    features_df = build_market_features(
        clean_prices_path=os.path.join(cfg["paths"]["interim_dir"], "clean_prices.parquet"),
        benchmark_symbol=cfg["market"]["benchmark"]
    )

    # 4. Strict Leakage-Aware Time Alignment
    console.print("[yellow][4/5] Aligning news and prices with historical cutoff protocol...[/yellow]")
    samples_df = align_and_build_samples(
        clean_news_path=os.path.join(cfg["paths"]["interim_dir"], "clean_news.parquet"),
        features_df=features_df,
        max_articles_per_cutoff=cfg["news_processing"].get("max_articles_per_cutoff", 5),
        threshold=threshold,
        benchmark_symbol=cfg["market"]["benchmark"]
    )

    # 5. Chronological Splits & Export
    console.print("[yellow][5/5] Partitioning chronological splits and exporting parquets...[/yellow]")
    split_cfg = cfg["splits"]["chronological"]
    train_df, val_df, test_df, _ = create_chronological_splits(
        samples_df,
        train_end=split_cfg["train_end"],
        val_start=split_cfg["val_start"],
        val_end=split_cfg["val_end"],
        test_start=split_cfg["test_start"],
        test_end=split_cfg["test_end"]
    )

    export_res = export_processed_dataset(
        train_df, val_df, test_df,
        output_dir=cfg["paths"]["processed_dir"],
        threshold=threshold,
        benchmark_symbol=cfg["market"]["benchmark"]
    )

    table = Table(title="Generated Processed Datasets")
    table.add_column("Split / Asset", style="cyan")
    table.add_column("File Path", style="green")
    table.add_column("Count", style="magenta")
    table.add_row("Train Split", export_res["train"], f"{len(train_df):,}")
    table.add_row("Validation Split", export_res["validation"], f"{len(val_df):,}")
    table.add_row("Test Split", export_res["test"], f"{len(test_df):,}")
    table.add_row("Feature Schema", export_res["feature_schema"], "19 features")
    table.add_row("Label Mapping", export_res["label_mapping"], "3 classes")
    table.add_row("Manifest", export_res["dataset_manifest"], "JSON metadata")
    console.print(table)
    console.print("[bold green]Preparation complete![/bold green]")


@app.command()
def validate(config: str = "configs/data.yaml"):
    """Performs rigorous automated leakage audit and quality checks."""
    console.print("[bold cyan]=== MarketReactionLab: Validation & Leakage Audit ===[/bold cyan]")
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    validator = PipelineValidator(
        processed_dir=cfg["paths"]["processed_dir"],
        reports_dir=cfg["paths"]["reports_dir"]
    )

    leakage_res = validator.run_leakage_audit()
    stats_res = validator.generate_statistical_reports()

    table = Table(title="Audit and Integrity Check Summary")
    table.add_column("Check Description", style="cyan")
    table.add_column("Result", style="green" if leakage_res["leakage_free"] else "red")
    table.add_row("Total Processed Samples", f"{leakage_res['total_samples_audited']:,}")
    table.add_row("Target Session > Reference Session", "PASSED (0 violations)" if leakage_res["target_strictly_after_reference"] else "FAILED")
    table.add_row("Chronological Split Boundary Isolation", "PASSED" if leakage_res["chronological_split_isolated"] else "FAILED")
    table.add_row("Sample ID Split Disjointness", "PASSED (0 overlap)" if (leakage_res["train_val_id_overlap"] + leakage_res["val_test_id_overlap"]) == 0 else "FAILED")
    table.add_row("Overall Leakage Status", "[bold green]100% LEAKAGE FREE[/bold green]" if leakage_res["leakage_free"] else "[bold red]LEAKAGE DETECTED[/bold red]")
    console.print(table)


@app.command()
def all(
    config: str = "configs/data.yaml",
    dry_run: bool = False,
    sample: bool = False,
    limit_symbols: int = typer.Option(None, help="Limit number of symbols")
):
    """Executes the complete end-to-end data pipeline from inspection to validation."""
    console.print("[bold magenta]************************************************************[/bold magenta]")
    console.print("[bold magenta]  MARKETREACTIONLAB — FULL END-TO-END DATA PIPELINE RUN     [/bold magenta]")
    console.print("[bold magenta]************************************************************[/bold magenta]")
    inspect()
    download(config=config, dry_run=dry_run, sample=sample, limit_symbols=limit_symbols)
    prepare(config=config, sample=sample)
    validate(config=config)
    console.print("[bold green]All pipeline phases successfully finished![/bold green]")


if __name__ == "__main__":
    app()
