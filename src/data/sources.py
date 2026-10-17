"""Data source discovery, provenance tracking, and remote metadata inspection."""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import logging
import struct
import io
import requests
from huggingface_hub import HfApi, hf_hub_url

logger = logging.getLogger(__name__)

HF_REPO_ID = "Zihan1004/FNSPID"
REPO_TYPE = "dataset"
LICENSE_ID = "CC BY-NC-4.0"
PROVENANCE_URL = "https://huggingface.co/datasets/Zihan1004/FNSPID"
ORIGINAL_PAPER_ARXIV = "2402.06698"


@dataclass
class SourceFileMetadata:
    """Metadata for a remote or local data source file."""
    path: str
    size_bytes: int
    size_mb: float
    description: str
    provenance: str
    license: str
    is_accessible: bool = True
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RemoteSourceInspector:
    """Inspects remote datasets on Hugging Face Hub without full downloads."""

    def __init__(self, repo_id: str = HF_REPO_ID):
        self.repo_id = repo_id
        self.api = HfApi()

    def list_files(self) -> List[SourceFileMetadata]:
        """List files and sizes from the remote repository."""
        tree = list(self.api.list_repo_tree(repo_id=self.repo_id, repo_type=REPO_TYPE, recursive=True))
        results = []
        for item in tree:
            size = getattr(item, "size", 0) or 0
            desc = "Dataset file"
            if item.path.endswith("All_external.csv"):
                desc = "Aggregated external financial news headlines (5.73 GB)"
            elif item.path.endswith("nasdaq_exteral_data.csv"):
                desc = "Nasdaq full corporate news articles with text bodies (23.23 GB)"
            elif item.path.endswith("full_history.zip"):
                desc = "Daily OHLCV historical stock prices for 7,600+ symbols (589 MB)"
            elif item.path.endswith("README.md"):
                desc = "Dataset documentation and citation details"

            results.append(SourceFileMetadata(
                path=item.path,
                size_bytes=size,
                size_mb=round(size / (1024 * 1024), 2),
                description=desc,
                provenance=PROVENANCE_URL,
                license=LICENSE_ID,
                is_accessible=True,
            ))
        return results

    def fetch_byte_range(self, filename: str, start_byte: int, end_byte: int) -> bytes:
        """Fetch a specific byte range from a Hugging Face Hub file using HTTP Range headers."""
        url = hf_hub_url(repo_id=self.repo_id, filename=filename, repo_type=REPO_TYPE)
        headers = {"Range": f"bytes={start_byte}-{end_byte}"}
        resp = requests.get(url, headers=headers, allow_redirects=True, timeout=30)
        resp.raise_for_status()
        return resp.content

    def inspect_csv_header(self, filename: str, preview_bytes: int = 4096) -> Tuple[List[str], str]:
        """Inspect the CSV header and first lines using a partial range download."""
        raw = self.fetch_byte_range(filename, 0, preview_bytes)
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        header = [c.strip() for c in lines[0].split(",")] if lines else []
        preview = "\n".join(lines[:5])
        return header, preview

    def parse_remote_zip_central_directory(self, zip_filename: str = "Stock_price/full_history.zip") -> Dict[str, Tuple[int, int, int, int]]:
        """
        Parses the ZIP central directory of a remote zip file without downloading the file.
        Returns a dict mapping filename -> (compress_method, compress_size, uncompress_size, header_offset).
        """
        url = hf_hub_url(repo_id=self.repo_id, filename=zip_filename, repo_type=REPO_TYPE)
        head_resp = requests.head(url, allow_redirects=True, timeout=30)
        head_resp.raise_for_status()
        total_size = int(head_resp.headers["content-length"])

        # Fetch last 64KB containing EOCD
        tail_size = min(65536, total_size)
        tail_data = requests.get(
            url,
            headers={"Range": f"bytes={total_size - tail_size}-{total_size - 1}"},
            allow_redirects=True,
            timeout=30
        ).content

        eocd_idx = tail_data.rfind(b"PK\x05\x06")
        if eocd_idx == -1:
            raise ValueError(f"Could not locate End of Central Directory in {zip_filename}")

        cd_size = struct.unpack("<I", tail_data[eocd_idx + 12 : eocd_idx + 16])[0]
        cd_offset = struct.unpack("<I", tail_data[eocd_idx + 16 : eocd_idx + 20])[0]

        # Fetch central directory
        cd_data = requests.get(
            url,
            headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}"},
            allow_redirects=True,
            timeout=60
        ).content

        entries = {}
        pos = 0
        while pos < len(cd_data):
            if cd_data[pos : pos + 4] != b"PK\x01\x02":
                break
            comp_method = struct.unpack("<H", cd_data[pos + 10 : pos + 12])[0]
            comp_size = struct.unpack("<I", cd_data[pos + 20 : pos + 24])[0]
            uncomp_size = struct.unpack("<I", cd_data[pos + 24 : pos + 28])[0]
            fname_len = struct.unpack("<H", cd_data[pos + 28 : pos + 30])[0]
            extra_len = struct.unpack("<H", cd_data[pos + 30 : pos + 32])[0]
            comm_len = struct.unpack("<H", cd_data[pos + 32 : pos + 34])[0]
            header_offset = struct.unpack("<I", cd_data[pos + 42 : pos + 46])[0]
            fname = cd_data[pos + 46 : pos + 46 + fname_len].decode("utf-8", errors="ignore")
            entries[fname] = (comp_method, comp_size, uncomp_size, header_offset)
            pos += 46 + fname_len + extra_len + comm_len

        logger.info(f"Successfully parsed remote zip index: {len(entries)} entries found.")
        return entries
