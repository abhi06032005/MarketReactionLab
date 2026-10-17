"""Curated stock universe for large-scale MarketReactionLab research on Kaggle.

Spans ~200 highly liquid U.S. equities across all 11 GICS sectors plus SPY benchmark.
"""

from typing import Dict, List, Set

GICS_SECTORS: Dict[str, List[str]] = {
    "Information_Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CRM", "CSCO", "INTC", "AMD", "QCOM",
        "TXN", "AMAT", "INTU", "IBM", "NOW", "LRCX", "ADI", "MU", "SNPS", "CDNS",
        "KLAC", "ANET", "MCHP", "NXPI", "APH", "MSI", "TEL", "HPQ", "GLW", "KEYS"
    ],
    "Communication_Services": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "CHTR", "TMUS", "EA",
        "WBD", "TTWO", "OMC", "IPG", "FOXA", "NWSA"
    ],
    "Consumer_Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG", "TJX", "TGT",
        "F", "GM", "MAR", "ORLY", "AZO", "ROST", "DHI", "LEN", "EBAY", "YUM"
    ],
    "Consumer_Staples": [
        "WMT", "PG", "COST", "KO", "PEP", "PM", "MDLZ", "MO", "CL", "EL",
        "KMB", "GIS", "SYY", "STZ", "ADM", "HSY", "K", "CAG", "TSN", "CLX"
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "USB",
        "PNC", "TFC", "CB", "MMC", "PGR", "MET", "PRU", "AFL", "ALL", "TRV",
        "AIG", "AJG", "COF", "BK", "STT"
    ],
    "Health_Care": [
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "BMY",
        "AMGN", "GILD", "CVS", "ISRG", "CI", "MDT", "SYK", "REGN", "VRTX", "BDX",
        "ZTS", "EW", "HUM", "BSX", "CNC", "MCK", "CAH", "BAX", "A", "ALXN"
    ],
    "Industrials": [
        "GE", "CAT", "UNP", "HON", "RTX", "BA", "LMT", "DE", "UPS", "EMR",
        "ETN", "ITW", "NSC", "CSX", "WM", "FDX", "GD", "NOC", "PH", "CMI",
        "TT", "PCAR", "ROK", "FAST", "CTAS"
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HES",
        "KMI", "WMB", "HAL", "BKR", "DVN", "FANG", "MRO", "APA"
    ],
    "Materials": [
        "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "CTVA", "DOW", "DD", "PPG",
        "ALB", "NUE", "VMC", "MLM", "BALL", "FMC"
    ],
    "Real_Estate": [
        "PLD", "AMT", "EQIX", "PSA", "CCI", "SPG", "O", "WELL", "DLR", "SBAC",
        "CBRE", "WY", "AVB", "EQR", "VTR"
    ],
    "Utilities": [
        "NEE", "SO", "DUK", "CEG", "SRE", "AEP", "D", "PEG", "EXC", "ED",
        "XEL", "PCG", "WEC", "ETR", "ES", "AEE"
    ],
}

BENCHMARK_SYMBOLS: List[str] = ["SPY", "QQQ", "IWM"]


def get_target_universe(max_stocks: int = 250) -> List[str]:
    """Return a flat, deduplicated list of target stocks plus benchmark."""
    stocks: List[str] = []
    for sector, tickers in GICS_SECTORS.items():
        stocks.extend(tickers)
    
    unique_stocks = list(dict.fromkeys(stocks))
    if len(unique_stocks) > max_stocks:
        unique_stocks = unique_stocks[:max_stocks]
        
    for bench in BENCHMARK_SYMBOLS:
        if bench not in unique_stocks:
            unique_stocks.append(bench)
            
    return unique_stocks


def get_sector_mapping() -> Dict[str, str]:
    """Map ticker symbol to its GICS sector."""
    mapping = {}
    for sector, tickers in GICS_SECTORS.items():
        for ticker in tickers:
            mapping[ticker] = sector
    for bench in BENCHMARK_SYMBOLS:
        mapping[bench] = "Benchmark"
    return mapping
