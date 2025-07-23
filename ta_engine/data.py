import pandas as pd
from core.dataloader import load_data


def load_price_data(filepath: str) -> pd.DataFrame:
    """Load price data from a CSV file."""
    return load_data(filepath)
