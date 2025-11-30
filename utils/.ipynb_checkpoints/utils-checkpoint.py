# utils.py

import re
import numpy as np

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