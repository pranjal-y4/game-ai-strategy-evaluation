"""
utils/serialization.py
Utilities for saving/loading results.
"""

import json
import csv
import os
from datetime import datetime


def save_csv(data: list, filename: str, fieldnames: list):
    """Save list of dicts to CSV."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def save_json(data: dict, filename: str):
    """Save dict to JSON."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def get_timestamp():
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")