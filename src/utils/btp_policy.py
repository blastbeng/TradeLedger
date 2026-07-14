import logging
from typing import Dict, Optional
from src.config.settings import settings
from src.utils.symbol_utils import is_btp_isin

logger = logging.getLogger(__name__)


class BTPPolicy:
    """Centralizes all BTP-specific trading rules and policies."""

    PAR_VALUE = 100.0
    BTP_SLIPPAGE_PCT = 0.001  # 0.1% fixed slippage for BTPs
    BTP_MAX_YIELD_SHIFT_BPS = 100  # Max expected yield shift in basis points for risk modeling

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
    def get_max_stop_loss_pct(symbol: str) -> Optional[float]:
        """Return the maximum stop-loss percentage for a symbol, or None if no cap."""
        if BTPPolicy.is_btp(symbol):
            return settings.BTP_MAX_STOP_LOSS_PCT
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

    @staticmethod
    def get_slippage_pct(symbol: str) -> Optional[float]:
        """Return a fixed slippage percentage for BTPs, or None for non-BTPs."""
        if BTPPolicy.is_btp(symbol):
            return BTPPolicy.BTP_SLIPPAGE_PCT
        return None

    @staticmethod
    def compute_btp_metrics(symbol: str, price: float) -> Optional[Dict[str, float]]:
        """Compute Macaulay duration, modified duration, convexity, and YTM for a BTP."""
        if not BTPPolicy.is_btp(symbol):
            return None
        from src.database import get_btp_details_from_db, compute_btp_ytm
        
        base_symbol = symbol.split("/")[0]
        details = get_btp_details_from_db([base_symbol])
        info = details.get(base_symbol)
        if not info or not info.get("maturity") or info.get("coupon") is None:
            return None
        
        maturity_str = info["maturity"]
        coupon = info["coupon"]
        ytm = compute_btp_ytm(coupon, maturity_str, price)
        if ytm is None:
            return None
        
        ytm_frac = ytm / 100.0
        try:
            from datetime import datetime
            maturity_date = datetime.strptime(maturity_str, "%Y-%m-%d")
            years_to_maturity = (maturity_date - datetime.now()).days / 365.25
            if years_to_maturity <= 0:
                return None
            
            periods = int(years_to_maturity * 2)
            if periods == 0:
                return None
            
            coupon_payment = (coupon / 100) * 100 / 2
            par_value = 100.0
            
            weighted_pv = 0.0
            total_pv = 0.0
            convexity = 0.0
            
            for i in range(1, periods + 1):
                pv = coupon_payment / ((1 + ytm_frac / 2) ** i)
                if i == periods:
                    pv += par_value / ((1 + ytm_frac / 2) ** i)
                weighted_pv += (i / 2) * pv
                total_pv += pv
                convexity += (i * (i + 1) / 4) * pv
            
            if total_pv <= 0:
                return None
            
            macaulay_duration = weighted_pv / total_pv
            modified_duration = macaulay_duration / (1 + ytm_frac / 2)
            convexity = convexity / (total_pv * (1 + ytm_frac / 2) ** 2)
            
            return {
                "ytm": ytm,
                "macaulay_duration": macaulay_duration,
                "modified_duration": modified_duration,
                "convexity": convexity,
            }
        except (ValueError, TypeError):
            return None

    @staticmethod
    def compute_btp_price_change(symbol: str, price: float, yield_shift_bps: float) -> Optional[float]:
        """Estimate percentage price change for a given yield shift (in basis points)."""
        metrics = BTPPolicy.compute_btp_metrics(symbol, price)
        if not metrics:
            return None
        
        delta_y = yield_shift_bps / 10000.0
        modified_duration = metrics["modified_duration"]
        convexity = metrics["convexity"]
        
        # % price change = -D_mod * dy + 0.5 * C * dy^2
        pct_change = -modified_duration * delta_y + 0.5 * convexity * (delta_y ** 2)
        return pct_change
