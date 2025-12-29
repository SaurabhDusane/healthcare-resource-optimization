"""Helper utility functions."""

import pandas as pd
import numpy as np
from typing import Any, Dict
import json

def save_json(data: Dict, filepath: str):
    """Save dictionary to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath: str) -> Dict:
    """Load JSON file to dictionary."""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_percentage_change(old_val: float, new_val: float) -> float:
    """Calculate percentage change."""
    if old_val == 0:
        return 0
    return ((new_val - old_val) / old_val) * 100

def format_large_number(num: int) -> str:
    """Format large numbers with K, M suffixes."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)
