"""Targeted, resumable downloader for MarketReactionLab.

Downloads selective price and news records for configured symbols from Hugging Face Hub
using HTTP range requests, completely avoiding the 29.6 GB full dataset download.
"""

import os
import sys
import zlib
import struct
import logging
import requests
import yaml
from pathlib import Path
from typing import List, Set, Dict, Optional, Tuple, Any
from huggingface_hub import hf_hub_url
from src.data.sources import RemoteSourceInspector, HF_REPO_ID, REPO_TYPE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class TargetedDownloader:
    """Selective, resumable downloader using HTTP range requests."""

    def __init__(self, config_path: str = "configs/data.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        symbols_cfg_path = self.cfg["paths"]["symbols_config"]
        with open(symbols_cfg_path, "r", encoding="utf-8") as f:
            sym_data = yaml.safe_load(f)

        self.benchmark = sym_data["benchmark"]
        self.symbols = list(sym_data["primary_symbols"])
        if self.benchmark not in self.symbols:
            self.symbols.append(self.benchmark)

        self.raw_prices_dir = Path(self.cfg["paths"]["raw_prices_dir"])
        self.raw_news_dir = Path(self.cfg["paths"]["raw_news_dir"])
        self.raw_prices_dir.mkdir(parents=True, exist_ok=True)
        self.raw_news_dir.mkdir(parents=True, exist_ok=True)

        self.inspector = RemoteSourceInspector(self.cfg["source"]["repo_id"])

    def download_prices(self, symbols: Optional[List[str]] = None, dry_run: bool = False) -> Dict[str, str]:
        """Downloads historical OHLCV CSVs directly from remote zip archive."""
        target_symbols = symbols or self.symbols
        logger.info(f"Checking historical prices for {len(target_symbols)} symbols...")

        # Parse central directory once
        zip_entries = self.inspector.parse_remote_zip_central_directory(self.cfg["source"]["prices_file"])
        url = hf_hub_url(repo_id=self.cfg["source"]["repo_id"], filename=self.cfg["source"]["prices_file"], repo_type=REPO_TYPE)

        saved_files = {}
        for sym in target_symbols:
            out_file = self.raw_prices_dir / f"{sym}.csv"
            if out_file.exists() and out_file.stat().st_size > 0:
                logger.info(f"Symbol {sym}: already cached at {out_file}")
                saved_files[sym] = str(out_file)
                continue

            entry_name = f"full_history/{sym}.csv"
            if entry_name not in zip_entries:
                logger.warning(f"Symbol {sym} not found in remote full_history.zip!")
                continue

            cm, cs, us, ho = zip_entries[entry_name]
            logger.info(f"Downloading {sym} ({us:,} bytes uncompressed, offset {ho})...")

            if dry_run:
                logger.info(f"[DRY-RUN] Would fetch bytes {ho} to {ho + cs + 150} for {sym}")
                saved_files[sym] = "[DRY-RUN]"
                continue

            # Fetch local header + compressed payload
            range_end = ho + 30 + 100 + cs
            r = requests.get(url, headers={"Range": f"bytes={ho}-{range_end}"}, allow_redirects=True, timeout=60)
            data = r.content

            # Parse local header
            loc_fname_len = struct.unpack("<H", data[26:28])[0]
            loc_extra_len = struct.unpack("<H", data[28:30])[0]
            comp_start = 30 + loc_fname_len + loc_extra_len
            comp_data = data[comp_start : comp_start + cs]

            if cm == 8:
                uncompressed = zlib.decompress(comp_data, -15)
            else:
                uncompressed = comp_data

            with open(out_file, "wb") as f:
                f.write(uncompressed)

            saved_files[sym] = str(out_file)

        logger.info(f"Prices ready: {len(saved_files)}/{len(target_symbols)} symbols available.")
        return saved_files

    def download_news(self, symbols: Optional[List[str]] = None, max_bytes: Optional[int] = None, dry_run: bool = False) -> str:
        """Downloads targeted news headlines matching configured symbols from All_external.csv."""
        target_symbols = set(symbols or self.symbols)
        limit_bytes = max_bytes or self.cfg["source"].get("max_news_bytes", 35000000)
        out_file = self.raw_news_dir / "raw_news_all.csv"

        if out_file.exists() and out_file.stat().st_size > 0 and not dry_run:
            logger.info(f"Raw news already downloaded at {out_file} ({out_file.stat().st_size:,} bytes).")
            return str(out_file)

        if dry_run:
            logger.info(f"[DRY-RUN] Would stream first {limit_bytes:,} bytes of {self.cfg['source']['news_file']}")
            return "[DRY-RUN]"

        logger.info(f"Downloading news chunk (up to {limit_bytes / (1024*1024):.1f} MB) for {len(target_symbols)} symbols...")
        url = hf_hub_url(repo_id=self.cfg["source"]["repo_id"], filename=self.cfg["source"]["news_file"], repo_type=REPO_TYPE)

        r = requests.get(url, headers={"Range": f"bytes=0-{limit_bytes - 1}"}, allow_redirects=True, timeout=120)
        raw_text = r.content.decode("utf-8", errors="replace")
        lines = raw_text.splitlines()

        if not lines:
            raise RuntimeError("Received empty news stream from Hugging Face Hub")

        header = lines[0]
        import csv
        reader = csv.reader(lines[1:])
        matched_rows = []
        rejected_rows = 0

        for row in reader:
            if len(row) >= 3:
                sym = row[2].strip()
                if sym in target_symbols:
                    matched_rows.append(row)
            else:
                rejected_rows += 1

        logger.info(f"News extracted: {len(matched_rows):,} headlines matched target symbols ({rejected_rows} malformed lines ignored).")

        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header.split(","))
            writer.writerows(matched_rows)

        logger.info(f"Saved filtered news to: {out_file}")
        return str(out_file)

    def run_all(self, limit_symbols: Optional[int] = None, dry_run: bool = False) -> Dict[str, Any]:
        """Runs targeted download for prices and news."""
        syms = self.symbols[:limit_symbols] if limit_symbols else self.symbols
        if self.benchmark not in syms:
            syms.append(self.benchmark)

        prices_res = self.download_prices(symbols=syms, dry_run=dry_run)
        news_res = self.download_news(symbols=syms, dry_run=dry_run)

        return {
            "prices_downloaded": len(prices_res),
            "news_file": news_res,
            "symbols": syms,
        }


if __name__ == "__main__":
    downloader = TargetedDownloader()
    downloader.run_all()
