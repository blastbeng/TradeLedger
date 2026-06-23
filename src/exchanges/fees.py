import logging
from typing import Dict, Any
import re

logger = logging.getLogger(__name__)


def calculate_transaction_costs(operation_type: str, stock_price: float, quantity: float, symbol: str = None) -> Dict[str, float]:
    """
    Calculate transaction costs (bank fees and Tobin tax) for Italian stocks
    based on the Intesa Sanpaolo Investo Standard Profile.

    Args:
        operation_type: 'BUY' or 'SELL'
        stock_price: Price per share in EUR
        quantity: Number of shares (can be fractional)
        symbol: The trading symbol or ISIN (used to detect BTP bonds)

    Returns:
        dict with keys: 'gross_value', 'bank_fee', 'tobin_tax', 'total_costs', 'net_value'
    """
    operation_type = operation_type.upper()
    if operation_type not in ("BUY", "SELL"):
        raise ValueError("operation_type must be 'BUY' or 'SELL'")

    gross_value = stock_price * quantity

    # --- BTP Bond Fee Logic ---
    is_btp = symbol is not None and re.match(r'^IT[A-Z0-9]{10}$', symbol.split("/")[0]) is not None
    if is_btp:
        from src.config.settings import settings
        if settings.BTP_IS_PRIMARY_ISSUANCE:
            bank_fee = 0.0
        else:
            raw_fee = gross_value * settings.BTP_FEE_PERC
            bank_fee = max(raw_fee, settings.BTP_MIN_FEE)
        
        # Sovereign bonds are exempt from Tobin tax
        tobin_tax = 0.0
        total_costs = bank_fee
        
        if operation_type == "BUY":
            net_value = gross_value + total_costs
        else:
            net_value = gross_value - bank_fee
            
        return {
            "gross_value": gross_value,
            "bank_fee": bank_fee,
            "tobin_tax": tobin_tax,
            "total_costs": total_costs,
            "net_value": net_value
        }

    # --- Standard Stock/ETF Fee Logic ---
    # Bank Commission (Intesa Sanpaolo Investo Standard Profile)
    percentage_commission = gross_value * 0.0024
    bank_fee = max(3.50, percentage_commission) + 2.50

    # State Tax (Italian Tobin Tax)
    tobin_tax = (gross_value * 0.0012) if operation_type == "BUY" else 0.0

    total_costs = bank_fee + tobin_tax

    if operation_type == "BUY":
        net_value = gross_value + total_costs
    else:  # SELL
        net_value = gross_value - bank_fee

    return {
        "gross_value": gross_value,
        "bank_fee": bank_fee,
        "tobin_tax": tobin_tax,
        "total_costs": total_costs,
        "net_value": net_value
    }
