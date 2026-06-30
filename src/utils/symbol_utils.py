import re

_BTP_ISIN_PATTERN = re.compile(r'^IT[A-Z0-9]{10}$')


def is_btp_isin(symbol: str) -> bool:
    """Check if a symbol is a BTP bond ISIN.

    Accepts symbols with or without a /QUOTE suffix
    (e.g., 'IT0001234567' or 'IT0001234567/EUR').
    """
    if not symbol:
        return False
    base = symbol.split("/")[0] if "/" in symbol else symbol
    return bool(_BTP_ISIN_PATTERN.match(base))
