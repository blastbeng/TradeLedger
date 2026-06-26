import logging
import json
from src.exchanges.market_data import discover_btp_bonds, _get_quotes_impl, get_quotes_cached

logging.basicConfig(level=logging.INFO)

def main():
    print("Discovering BTP bonds...")
    btp_bonds = discover_btp_bonds()
    
    # Filter for BTPs starting with 'IT'
    btp_symbols = [b["isin"] for b in btp_bonds if b.get("isin", "").startswith("IT")]
    
    print(f"Found {len(btp_symbols)} BTP symbols starting with 'IT'.")
    
    if btp_symbols:
        print("\n--- Testing _get_quotes_impl ---")
        quotes_impl = _get_quotes_impl(btp_symbols)
        print(json.dumps(quotes_impl, indent=2))
        
        print("\n--- Testing get_quotes_cached ---")
        quotes_cached = get_quotes_cached(btp_symbols)
        print(json.dumps(quotes_cached, indent=2))
    else:
        print("No BTP symbols found to test.")

if __name__ == "__main__":
    main()
