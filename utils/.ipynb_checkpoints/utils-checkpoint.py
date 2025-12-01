# utils.py

import re
import numpy as np
import ast

TITLE_PREFIX_PATTERN = re.compile(
    r"^(mr\.?|ms\.?|mrs\.?|dr\.?|prof\.?)\s+",
    re.IGNORECASE
)

def clean_name(name):
    if not isinstance(name, str):
        return name
    return TITLE_PREFIX_PATTERN.sub("", name).strip().lower()


def extract_officer_names(arr):
    """
    Returns list of strings: "clean_name|yearBorn"
    """
    results = []

    if isinstance(arr, (list, np.ndarray)):
        for o in arr:
            if isinstance(o, dict) and "name" in o:
                cleaned = clean_name(o["name"])
                yob = o.get("yearBorn", np.nan)
                results.append(f"{cleaned}|{yob}")

    return results


def parse_list(cell):
    """
    Parse a cell from the raw string format into
    a Python list of dictionaries. Handles Pandas Timestamp(...) objects.
    
    Parameters
    ----------
    cell : str or object
        The raw cell value containing a stringified list of dicts.

    Returns
    -------
    list or None
        Parsed list of dictionaries, or None if parsing fails.
    """
    if not isinstance(cell, str):
        return cell

    # Remove outer array formatting, e.g. '[" ... "]'
    cleaned = cell.strip()

    # Case: cell looks like ["[...]"]
    if cleaned.startswith('["') and cleaned.endswith('"]'):
        cleaned = cleaned[2:-2]

    # Replace Pandas Timestamp('...') → '...'
    cleaned = re.sub(r"Timestamp\('([^']+)'\)", r"'\1'", cleaned)

    # Try literal_eval
    try:
        parsed = ast.literal_eval(cleaned)
        return parsed  # returns a list of dicts
    except Exception:
        return None


def extract_institution_names(arr):
    """
    Returns list of institution names
    """
    results = []
    if isinstance(arr, str):
        arr = parse_list(arr)
    if isinstance(arr, (list, np.ndarray)):
        for o in arr:
            if isinstance(o, dict) and "Holder" in o:
                inst = o.get("Holder")
                results.append(inst.lower().strip())
    return results


def after_first_hyphen(text):
    # Regex: capture before-first-hyphen as group 1, after-first-hyphen as group 2
    match = re.search(r'^([^-]*)-(.*)', text)
    if not match:
        return text

    before = match.group(1).strip()
    after = match.group(2).strip()

    # Condition: if the prefix is exactly "Bridgeway Funds, Inc."
    if before == "Bridgeway Funds, Inc.":
        return f"Bridgeway {after}"
    elif before == 'TIAA-CREF Funds':
        return f"{after.replace('CREF Funds-', '')}"
    elif before == 'SPDR SERIES TRUST':
        return f"{after.replace('(R)', '')}"
    elif before == 'DFA INVESTMENT DIMENSIONS GROUP INC':
        return text
    elif after.startswith('Price (T.Rowe)'):
        remove_words = ['Markets', 'Trust', 'Fund', 'Stock', 'Fd.', 'Equity']
        after = ' '.join(word for word in after.split() if word not in remove_words)
        return f"T.Rowe Price {after.strip()}"
    else:
        return after


def extract_mutualfund_names(arr):
    """
    Returns list of mutual funds names
    """
    results = []
    if isinstance(arr, str):
        arr = parse_list(arr)
    if isinstance(arr, (list, np.ndarray)):
        for o in arr:
            if isinstance(o, dict) and "Holder" in o:
                fund = o.get("Holder")
                fund = after_first_hyphen(fund)
                
                results.append(fund.lower().strip())
    return results