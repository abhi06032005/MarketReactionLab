"""Large-scale data ingestion module for MarketReactionLab inside Kaggle.

Downloads raw FNSPID historical prices and streams multi-gigabyte news CSV
incrementally from Hugging Face Hub (Zihan1004/FNSPID), filtering toward
150-300 liquid U.S. stocks without overwhelming RAM or disk.
"""

import gc
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set
import urllib.request
import pandas as pd
import numpy as np

logger = logging.getLogger("KaggleIngestion")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HF_PRICE_URL = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip"
HF_NEWS_URL = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/All_external.csv"


def check_disk_space(path: str = "/kaggle/working", min_gb_required: float = 10.0) -> Dict[str, float]:
    """Check available disk space in GB."""
    stat = shutil.disk_usage(path if os.path.exists(path) else ".")
    total_gb = stat.total / (1024 ** 3)
    used_gb = stat.used / (1024 ** 3)
    free_gb = stat.free / (1024 ** 3)
    logger.info(f"Disk storage check at {path}: Total={total_gb:.1f}GB, Free={free_gb:.1f}GB, Used={used_gb:.1f}GB")
    if free_gb < min_gb_required:
        logger.warning(f"Available disk ({free_gb:.1f}GB) is below recommended ({min_gb_required:.1f}GB).")
    return {"total_gb": total_gb, "used_gb": used_gb, "free_gb": free_gb}


def download_with_progress(url: str, output_path: Path, max_retries: int = 3) -> Path:
    """Download file with retry logic and stream chunking."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 1024 * 1024:
        logger.info(f"File already exists at {output_path} ({output_path.stat().st_size / (1024**2):.1f} MB). Skipping download.")
        return output_path

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading from {url} to {output_path} (Attempt {attempt}/{max_retries})...")
            
            # Use urllib request with User-Agent header
            req = urllib.request.Request(url, headers={"User-Agent": "MarketReactionLab-Kaggle/1.0"})
            with urllib.request.urlopen(req, timeout=120) as response, open(output_path, "wb") as out_file:
                chunk_size = 1024 * 1024 * 8  # 8 MB chunks
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (64 * 1024 * 1024) == 0:
                        logger.info(f"Downloaded {downloaded / (1024**2):.1f} MB...")
            
            logger.info(f"Successfully downloaded {output_path} ({output_path.stat().st_size / (1024**2):.1f} MB)")
            return output_path
        except Exception as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if output_path.exists():
                output_path.unlink()
            if attempt == max_retries:
                raise RuntimeError(f"Failed to download {url} after {max_retries} attempts: {e}")


def extract_target_prices(zip_path: Path, extract_dir: Path, target_symbols: List[str]) -> List[Path]:
    """Extract only the target stocks and benchmark CSVs from full_history.zip."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted_files: List[Path] = []
    symbol_set = {s.upper() for s in target_symbols}
    
    # Check if already extracted
    existing_files = list(extract_dir.glob("*.csv"))
    existing_symbols = {f.stem.upper() for f in existing_files}
    if len(existing_symbols.intersection(symbol_set)) >= len(symbol_set) * 0.8:
        logger.info(f"Found {len(existing_files)} existing price CSVs in {extract_dir}. Skipping extraction.")
        return [f for f in existing_files if f.stem.upper() in symbol_set]

    logger.info(f"Opening price archive {zip_path} for selective extraction of {len(symbol_set)} symbols...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        name_list = zf.namelist()
        logger.info(f"Archive contains {len(name_list):,} total files.")
        
        for name in name_list:
            base_name = os.path.basename(name)
            if not base_name.endswith(".csv"):
                continue
            symbol = base_name.replace(".csv", "").upper()
            if symbol in symbol_set:
                target_dest = extract_dir / f"{symbol}.csv"
                if not target_dest.exists():
                    with zf.open(name) as src, open(target_dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                extracted_files.append(target_dest)
                
    logger.info(f"Extracted {len(extracted_files)} target price files to {extract_dir}.")
    return extracted_files


def process_matched_news_parquet(
    matched_parquet_path: Path,
    output_parquet: Path,
    target_symbols: Optional[List[str]] = None,
    filter_retrospective: bool = True
) -> int:
    """Process existing matched_news.parquet with schemas: Date, Article_title, Stock_symbol, Url, Publisher."""
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading pre-matched news from {matched_parquet_path}...")
    df = pd.read_parquet(matched_parquet_path)
    logger.info(f"Loaded {len(df):,} raw rows. Columns: {list(df.columns)}")

    # Standardize column mapping
    col_map = {}
    for col in df.columns:
        c_low = col.lower().strip()
        if c_low in ["stock_symbol", "ticker", "symbol", "stock"]:
            col_map[col] = "ticker"
        elif c_low in ["article_title", "headline", "title", "text"]:
            col_map[col] = "headline"
        elif c_low in ["date", "published_at", "timestamp"]:
            col_map[col] = "published_at"

    df = df.rename(columns=col_map)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["headline"] = df["headline"].fillna("").astype(str).str.strip()

    # Filter target symbols if provided
    if target_symbols:
        sym_set = {s.upper() for s in target_symbols}
        df = df[df["ticker"].isin(sym_set)]
        logger.info(f"Filtered to {len(df):,} rows matching target universe of {len(sym_set)} symbols.")

    df = df[df["headline"].str.len() > 10]

    if filter_retrospective:
        import re
        retrospective_patterns = [
            r'\bplunges?\b', r'\bsurges?\b', r'\bup\s+\d+%', r'\bdown\s+\d+%',
            r'\bgainers?\b', r'\blosers?\b', r'\bbiggest movers\b', r'\bslides?\b',
            r'\brallies\b', r'\btumbles?\b', r'\bjumps?\b', r'\bdips?\b'
        ]
        retro_regex = re.compile('|'.join(retrospective_patterns), re.IGNORECASE)
        mask_non_retro = ~df["headline"].apply(lambda t: bool(retro_regex.search(t)))
        before_count = len(df)
        df = df[mask_non_retro]
        logger.info(f"Purged {before_count - len(df):,} retrospective leakage headlines. {len(df):,} remain.")

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.dropna(subset=["published_at"])
    df = df.drop_duplicates(subset=["ticker", "headline"])
    df = df.sort_values("published_at").reset_index(drop=True)

    df.to_parquet(output_parquet, index=False, compression="snappy")
    logger.info(f"Saved {len(df):,} clean news rows to {output_parquet}")
    return len(df)


def stream_and_filter_news(
    news_csv_path: Path,
    output_parquet: Path,
    target_symbols: List[str],
    chunk_size: int = 50000,
    filter_retrospective: bool = True,
    max_records: Optional[int] = None
) -> int:
    """Stream multi-gigabyte news CSV chunk-by-chunk to prevent RAM exhaustion.
    
    Applies:
    1. Ticker filtering to target universe
    2. Retrospective headline filtering (e.g., 'down 2%', 'surges', 'movers')
    3. Point-in-time date validation
    4. Chunk-wise incremental Parquet writing
    """
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    symbol_set = {s.upper() for s in target_symbols}
    
    # Regex for retrospective commentary target leakage
    retrospective_patterns = [
        r'\bplunges?\b', r'\bsurges?\b', r'\bup\s+\d+%', r'\bdown\s+\d+%',
        r'\bgainers?\b', r'\blosers?\b', r'\bbiggest movers\b', r'\bslides?\b',
        r'\brallies\b', r'\btumbles?\b', r'\bjumps?\b', r'\bdips?\b'
    ]
    import re
    retro_regex = re.compile('|'.join(retrospective_patterns), re.IGNORECASE)

    total_kept = 0
    total_processed = 0
    chunks_written = []
    temp_dir = output_parquet.parent / "temp_news_chunks"
    temp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Streaming news CSV from {news_csv_path} in chunks of {chunk_size:,}...")
    
    try:
        reader = pd.read_csv(
            news_csv_path,
            chunksize=chunk_size,
            dtype=str,
            on_bad_lines="skip",
            low_memory=False
        )

        for chunk_idx, chunk in enumerate(reader):
            total_processed += len(chunk)
            
            # Standardize columns
            chunk.columns = [c.lower().strip() for c in chunk.columns]
            
            # Map ticker column
            ticker_col = None
            for col in ["ticker", "symbol", "stock"]:
                if col in chunk.columns:
                    ticker_col = col
                    break
            
            headline_col = None
            for col in ["headline", "title", "news_title", "text"]:
                if col in chunk.columns:
                    headline_col = col
                    break

            date_col = None
            for col in ["date", "datetime", "release_date", "timestamp"]:
                if col in chunk.columns:
                    date_col = col
                    break

            if not (ticker_col and headline_col and date_col):
                logger.warning(f"Chunk {chunk_idx} missing required columns. Found: {list(chunk.columns)}")
                continue

            # 1. Filter by target symbols
            filtered = chunk[chunk[ticker_col].str.upper().isin(symbol_set)].copy()
            if filtered.empty:
                continue

            filtered.rename(columns={
                ticker_col: "ticker",
                headline_col: "headline",
                date_col: "published_at"
            }, inplace=True)

            filtered["ticker"] = filtered["ticker"].str.upper().str.strip()
            filtered["headline"] = filtered["headline"].fillna("").astype(str).str.strip()
            filtered = filtered[filtered["headline"].str.len() > 10]

            # 2. Filter retrospective headlines to avoid target leakage
            if filter_retrospective:
                mask_non_retro = ~filtered["headline"].apply(lambda t: bool(retro_regex.search(t)))
                filtered = filtered[mask_non_retro]

            # 3. Clean publication dates
            filtered["published_at"] = pd.to_datetime(filtered["published_at"], errors="coerce")
            filtered = filtered.dropna(subset=["published_at"])
            filtered = filtered.drop_duplicates(subset=["ticker", "headline"])

            if not filtered.empty:
                chunk_file = temp_dir / f"chunk_{chunk_idx:05d}.parquet"
                filtered.to_parquet(chunk_file, index=False)
                chunks_written.append(chunk_file)
                total_kept += len(filtered)

            if chunk_idx % 20 == 0:
                logger.info(f"Processed {total_processed:,} lines. Kept {total_kept:,} target news records so far.")
                gc.collect()

            if max_records and total_kept >= max_records:
                logger.info(f"Reached max records threshold ({max_records:,}). Stopping stream.")
                break

    except Exception as e:
        logger.error(f"Error during news streaming: {e}")
        raise

    logger.info(f"Streaming complete. Merging {len(chunks_written)} intermediate chunks into {output_parquet}...")
    if chunks_written:
        merged_dfs = [pd.read_parquet(cf) for cf in chunks_written]
        final_df = pd.concat(merged_dfs, ignore_index=True)
        final_df = final_df.sort_values("published_at").reset_index(drop=True)
        final_df.to_parquet(output_parquet, index=False, compression="snappy")
        logger.info(f"Successfully saved {len(final_df):,} curated news articles to {output_parquet}")
        
        # Cleanup temporary chunks
        shutil.rmtree(temp_dir, ignore_errors=True)
        return len(final_df)
    else:
        logger.warning("No articles matched the filter criteria.")
        return 0
