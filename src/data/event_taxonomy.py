"""Deterministic rule-based event taxonomy, multi-company detection, and price-movement filters."""

import re
from typing import Dict, List, Tuple, Any, Optional

TAXONOMY_CATEGORIES = [
    "Earnings",
    "Revenue/profit guidance",
    "Analyst rating changes",
    "Mergers/acquisitions",
    "Regulatory/legal developments",
    "Product announcements",
    "Management changes",
    "Debt/financing",
    "Macroeconomic developments",
    "Other/uncertain",
]

# Regex rules for each category
TAXONOMY_RULES = {
    "Earnings": [
        r"\b(?:q[1-4]\s+)?earnings\b",
        r"\beps\b",
        r"\bquarterly results\b",
        r"\breports\s+q[1-4]\b",
        r"\bprofit\s+rises\b",
        r"\bprofit\s+falls\b",
        r"\bnet\s+income\b",
        r"\bearnings\s+beat\b",
        r"\bearnings\s+miss\b",
    ],
    "Revenue/profit guidance": [
        r"\bguidance\b",
        r"\boutlook\b",
        r"\braises\s+(?:full[- ]year\s+)?forecast\b",
        r"\blowers\s+(?:full[- ]year\s+)?forecast\b",
        r"\bsees\s+fy\d{2,4}\b",
        r"\brevenue\s+target\b",
        r"\bpreannounces\b",
    ],
    "Analyst rating changes": [
        r"\bupgrades?\b",
        r"\bdowngrades?\b",
        r"\bmaintains\s+(?:buy|neutral|hold|overweight|underweight)\b",
        r"\breiterates\s+(?:buy|neutral|hold|overweight|underweight)\b",
        r"\binitiates\s+coverage\b",
        r"\bprice\s+target\b",
        r"\bpt\s+raised\b",
        r"\bpt\s+lowered\b",
        r"\boutperform\b",
        r"\bunderperform\b",
    ],
    "Mergers/acquisitions": [
        r"\bacqui(?:re|res|red|sition)\b",
        r"\bmerger\b",
        r"\bbuyout\b",
        r"\btakeover\b",
        r"\bto\s+buy\b",
        r"\bsells\s+unit\b",
        r"\bdivest(?:ment|iture|s)?\b",
        r"\bdeal\s+valued\b",
    ],
    "Regulatory/legal developments": [
        r"\bfda\b",
        r"\bsec\b",
        r"\blawsuit\b",
        r"\bsues\b",
        r"\bsued\b",
        r"\bpatent\b",
        r"\bsettlement\b",
        r"\bantitrust\b",
        r"\binvestigation\b",
        r"\bcompliance\b",
        r"\bapprov(?:al|ed|es)\b",
        r"\bfined?\b",
    ],
    "Product announcements": [
        r"\bunveils?\b",
        r"\blaunches?\b",
        r"\bintroduces?\b",
        r"\bnew\s+product\b",
        r"\bnew\s+service\b",
        r"\bplatform\b",
        r"\brolls?\s+out\b",
        r"\bfeature\b",
        r"\bpartnership\b",
        r"\bcollaborat(?:ion|es)\b",
    ],
    "Management changes": [
        r"\bceo\b",
        r"\bcfo\b",
        r"\bexecutive\b",
        r"\bappoints?\b",
        r"\bnames?\s+new\b",
        r"\bstep(?:s)?\s+down\b",
        r"\bresigns?\b",
        r"\bretires?\b",
        r"\bboard\s+of\s+directors\b",
    ],
    "Debt/financing": [
        r"\bnotes\s+offering\b",
        r"\bdebt\b",
        r"\bcredit\s+facility\b",
        r"\bshares?\s+offering\b",
        r"\bstock\s+repurchase\b",
        r"\bbuyback\b",
        r"\bdividend\b",
        r"\byield\b",
        r"\brefinanc(?:ing|es)\b",
    ],
    "Macroeconomic developments": [
        r"\bfed\b",
        r"\bfederal\s+reserve\b",
        r"\binterest\s+rates?\b",
        r"\binflation\b",
        r"\btariff\b",
        r"\btrade\s+war\b",
        r"\bgdp\b",
        r"\bjobless\s+claims\b",
        r"\bcentral\s+bank\b",
        r"\bcoronavirus\b",
        r"\bcovid[- ]?19\b",
        r"\bpandemic\b",
    ],
}

# Price-movement headlines (post-outcome summaries rather than news events)
PRICE_MOVEMENT_PATTERNS = [
    r"\bshares\s+(?:are\s+)?trading\s+(?:higher|lower|up|down)\b",
    r"\bshares\s+(?:are\s+)?(?:dropping|surging|falling|rising|tumbling|plunging|rallying)\b",
    r"\bwhy\s+.*?\s+(?:is|are)\s+(?:rising|falling|dropping|surging|tumbling|up|down)\b",
    r"\b\d+\s+biggest\s+movers\b",
    r"\b\d+\s+stocks\s+moving\b",
    r"\bmid-day\s+session\b",
    r"\bpre-market\s+session\b",
    r"\bmarket\s+wrap\b",
    r"\bstocks\s+that\s+hit\s+52-week\b",
    r"\btrading\s+higher\s+after\b",
    r"\btrading\s+lower\s+after\b",
    r"\boptions?\s+trading\b",
    r"\bput\s+and\s+call\s+options\b",
    r"\ba\s+peek\s+into\s+the\s+markets\b",
    r"\bmorning\s+market\s+stats\b",
    r"\bstock\s+market\s+today\b",
]

# Multiple company markers
MULTI_COMPANY_PATTERNS = [
    r"\bpeers\b",
    r"\bsector\s+stocks\b",
    r"\b(?:tech|chip|bank|airline|retail)\s+stocks\b",
    r"\bvs\.?\b",
    r"\broundtable\s+picks\b",
    r"\bbulls\s+and\s+bears\b",
    r"\bmovers\s+from\b",
    r"\b(?:stocks|companies)\s+to\s+watch\b",
]


def classify_event(headline: str) -> Tuple[str, float]:
    """
    Classifies a financial headline into one of 10 standard event taxonomy categories.
    Returns (category, confidence).
    """
    if not headline:
        return "Other/uncertain", 0.0

    text = headline.lower()

    for category, patterns in TAXONOMY_RULES.items():
        for pat in patterns:
            if re.search(pat, text):
                # High confidence if clear keyword matched
                return category, 0.85

    return "Other/uncertain", 0.20


def is_price_movement_headline(headline: str) -> bool:
    """Detects whether headline is merely reporting stock price movement or market session summary."""
    if not headline:
        return False
    text = headline.lower()
    for pat in PRICE_MOVEMENT_PATTERNS:
        if re.search(pat, text):
            return True
    return False


def is_multi_company_headline(headline: str) -> bool:
    """Detects whether headline refers to multiple companies, baskets, or entire sectors."""
    if not headline:
        return False
    text_lower = headline.lower()
    for pat in MULTI_COMPANY_PATTERNS:
        if re.search(pat, text_lower):
            return True
    # Check for multiple tickers or company comma lists in original text
    # e.g. "Apple, Google and Amazon" or "AAPL, MSFT, GOOG"
    if re.search(r"\b[A-Z][a-zA-Z]+,\s+[A-Z][a-zA-Z]+(?:\s+and\s+[A-Z][a-zA-Z]+)?\b", headline):
        return True
    if re.search(r"\b[A-Z]{2,5},\s*[A-Z]{2,5}\b", headline):
        return True
    return False



class EventTaxonomy:
    """Class wrapper for event taxonomy classification."""
    CATEGORIES = TAXONOMY_CATEGORIES

    @staticmethod
    def classify(headline: str) -> Tuple[str, float]:
        return classify_event(headline)


class EventTaxonomyClassifier(EventTaxonomy):
    """Classifier returning primary category string."""
    def classify(self, headline: str) -> str:
        cat, _ = classify_event(headline)
        return cat


# Conservative company name patterns for ambiguous single-letter and short tickers
TICKER_COMPANY_PATTERNS = {
    "A": re.compile(r"\bagilent\b", re.IGNORECASE),
    "C": re.compile(r"\bciti(?:group)?\b", re.IGNORECASE),
    "F": re.compile(r"\bford\b", re.IGNORECASE),
    "K": re.compile(r"\bkellogg\b", re.IGNORECASE),
    "T": re.compile(r"\bat&t\b|\batt\b", re.IGNORECASE),
    "V": re.compile(r"\bvisa\b", re.IGNORECASE),
}

# Explicit competitor mentions that indicate a false tag when target company is absent
TICKER_FALSE_TAG_PATTERNS = {
    "A": re.compile(r"\bthermo\s*fisher\b|\btmo\b|\bdanaher\b|\bperkinelmer\b|\bwaters\s+corp\b", re.IGNORECASE),
    "T": re.compile(r"\bt-mobile\b|\btmus\b|\bverizon\b|\bvz\b|\bsprint\b", re.IGNORECASE),
}


def validate_ticker_headline_attribution(ticker: str, headline: str) -> Tuple[bool, Optional[str]]:
    """Audit ticker-to-headline mapping to reject misattributed ticker tags.
    
    Specifically handles:
    1. Ticker 'A' (Agilent Technologies) falsely matched to English indefinite article 'a'
       or headlines concerning Thermo Fisher Scientific / competitors.
    2. Ticker 'T' (AT&T) falsely tagged to T-Mobile stories.
    3. Single-letter tickers tagged on generic commentary without mentioning the target company.
    
    Returns (is_valid, rejection_reason). Note: This is a conservative filter, not a claim of
    perfect entity matching.
    """
    if not headline or not ticker:
        return False, "empty_headline_or_ticker"

    ticker_upper = ticker.upper().strip()

    # Special audit for ticker 'A' (Agilent)
    if ticker_upper == "A":
        has_agilent = bool(TICKER_COMPANY_PATTERNS["A"].search(headline))
        has_thermo = bool(TICKER_FALSE_TAG_PATTERNS["A"].search(headline))
        if has_thermo and not has_agilent:
            return False, "misattributed_a_thermo_fisher"
        if not has_agilent:
            # Single-letter 'A' in headline is almost always an English word or false match
            return False, "misattributed_a_no_company_name"

    # Special audit for ticker 'T' (AT&T vs T-Mobile)
    elif ticker_upper == "T":
        has_att = bool(TICKER_COMPANY_PATTERNS["T"].search(headline))
        has_tmobile = bool(TICKER_FALSE_TAG_PATTERNS["T"].search(headline))
        if has_tmobile and not has_att:
            return False, "misattributed_t_tmobile"

    # General single-letter ticker audit (C, F, K, V)
    elif ticker_upper in TICKER_COMPANY_PATTERNS:
        company_pat = TICKER_COMPANY_PATTERNS[ticker_upper]
        if not company_pat.search(headline):
            # Check if exact ticker appears with market context (e.g., '(Symbol: C)' or 'NYSE:C')
            context_match = re.search(rf"\b(?:symbol|nyse|nasdaq|ticker)[:\s]+{ticker_upper}\b", headline, re.IGNORECASE)
            if not context_match:
                return False, f"misattributed_{ticker_upper.lower()}_no_company_name"

    return True, None


