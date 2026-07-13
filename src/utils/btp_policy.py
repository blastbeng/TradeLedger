import logging
from typing import Optional
from src.config.settings import settings
from src.utils.symbol_utils import is_btp_isin

logger = logging.getLogger(__name__)


class BTPPolicy:
    """Centralizes all BTP-specific trading rules and policies."""

    PAR_VALUE = 100.0

    @staticmethod
    def is_btp(symbol: str) -> bool:
        """Check if a symbol is a BTP bond."""
        return is_btp_isin(symbol)

    @staticmethod
    def supports_trailing_stop(symbol: str) -> bool:
        """Trailing stops are not supported for BTPs on Intesa Sanpaolo Investo."""
        return not BTPPolicy.is_btp(symbol)

    @staticmethod
    def get_max_take_profit_pct(symbol: str) -> Optional[float]:
        """Return the maximum take-profit percentage for a symbol, or None if no cap."""
        if BTPPolicy.is_btp(symbol):
            return settings.BTP_MAX_TAKE_PROFIT_PCT
        return None

    @staticmethod
    def get_hard_max_loss_pct(symbol: str, timeframe: Optional[str] = None) -> float:
        """Determine the hard max loss percentage based on asset type and timeframe."""
        if not BTPPolicy.is_btp(symbol):
            return 0.0  # Non-BTP handled by caller

        default_loss = settings.BTP_HARD_MAX_LOSS_PCT
        tf_loss = 0.0
        if timeframe == "1h":
            tf_loss = settings.BTP_HARD_MAX_LOSS_PCT_1H
        elif timeframe == "1d":
            tf_loss = settings.BTP_HARD_MAX_LOSS_PCT_1D
        elif timeframe == "1w":
            tf_loss = settings.BTP_HARD_MAX_LOSS_PCT_1W
        elif timeframe == "1M":
            tf_loss = settings.BTP_HARD_MAX_LOSS_PCT_1M
        elif timeframe == "3M":
            tf_loss = settings.BTP_HARD_MAX_LOSS_PCT_3M
        elif timeframe in ("6M", "1Y", "3Y", "5Y"):
            tf_loss = settings.BTP_HARD_MAX_LOSS_PCT_6M_1Y

        return tf_loss if tf_loss > 0 else default_loss

    @staticmethod
    def compute_breakeven_price(symbol: str, entry_price: float, amount: float) -> float:
        """Compute the breakeven stop price for a position."""
        if BTPPolicy.is_btp(symbol):
            return entry_price
        return entry_price  # Fallback, non-BTP handled by caller with fee calc

    @staticmethod
    def compute_fees(operation_type: str, gross_value: float) -> float:
        """Compute BTP fees. Assumes caller has verified it's a BTP."""
        if settings.BTP_IS_PRIMARY_ISSUANCE:
            return 0.0
        raw_fee = gross_value * settings.BTP_FEE_PERC
        return max(raw_fee, settings.BTP_MIN_FEE)
