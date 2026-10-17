"""CLI script to verify dataset integrity and audit for leakage."""
from src.data.pipeline import validate

if __name__ == "__main__":
    validate()
