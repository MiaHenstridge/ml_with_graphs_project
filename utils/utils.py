# utils.py

import re
import numpy as np
import ast
import datetime
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import os 
import json
from typing import *

####################################
# DATA ENGINEERING 
####################################

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


def years_since_timestamp(timestamp: float) -> float:
    """
    Calculate the number of years elapsed since a given Unix timestamp.
    Handles null values (NaN) safely.
    
    Parameters:
    - timestamp: float or int, Unix timestamp in seconds
    
    Returns:
    - float: years elapsed, or np.nan if timestamp is null
    """
    if pd.isna(timestamp):
        return np.nan
    
    date = datetime.datetime.utcfromtimestamp(timestamp)
    now = datetime.datetime.utcnow()
    delta_days = (now - date).days
    years_elapsed = delta_days / 365.25  # account for leap years
    return years_elapsed


# ==============================================================================
# Preparing data for link prediction
# ==============================================================================
def invert_nested_dict(d: dict) -> dict:
    """
    Inverts a nested dictionary where inner values become new keys.
    Assumes the input is {ntype: {old_key: new_key, ...}}
    Returns: {ntype: {new_key: old_key, ...}}
    """
    inverted = {}
    for outer_key, inner_dict in d.items():
        # Inner values are expected to be unique if they become keys
        inverted[outer_key] = {v: k for k, v in inner_dict.items()}
    return inverted


def localize_and_create_inverse_edges(
    data: HeteroData,
    id_mapping_path: str,
    edge_types_original: list,
) -> tuple[HeteroData, list, list]:
    """
    1. Localizes edge indices in HeteroData using a global-to-local ID map.
    2. Creates and adds the corresponding inverse edge types.

    Args:
        data (HeteroData): The PyG data object with global IDs.
        id_mapping_path (str): Path to the JSON file containing the global ID map.
        edge_types_original (list): List of original edge triplets (src, rel, dst).
        invert_map_func (callable): Function to invert the raw ID map structure.

    Returns:
        tuple: (data with localized edges, forward edge list, all edge list)
    """
    
    # --- Setup and Mapping Load ---
    
    # 1. Load and prepare the ID map
    with open(id_mapping_path) as f:
        # Load and invert the map (e.g., from local2global to global2local)
        raw_global2local = invert_nested_dict(json.load(f))

    # 2. Convert ALL keys and values to integers (CRITICAL STEP)
    global2local = {}
    for ntype, g_to_l_map in raw_global2local.items():
        # Convert global ID keys (strings) and local ID values (strings) to integers
        global2local[ntype] = {
            int(global_id): int(local_id) 
            for global_id, local_id in g_to_l_map.items()
        }
    
    # Get device from an existing tensor
    if not edge_types_original:
        print("Error: edge_types_original list is empty.")
        return data, [], []
        
    # Find the device of the graph data
    device = data.edge_index_dict[edge_types_original[0]].device 

    final_edge_types = []
    forward_edge_types = []

    print("Starting localization of edge indices and creating inverse relations...")

    for src, rel, dst in edge_types_original:
        original_edge_type = (src, rel, dst)
        
        # 1. Check and Retrieve Data
        if original_edge_type not in data.edge_types:
            print(f"Warning: Skipping {original_edge_type} - not found in data.")
            continue
            
        # Get the original index tensor (assumed to hold global IDs).
        # Move to CPU for efficient dictionary lookup, then back to device later.
        original_index_global = data[original_edge_type].edge_index.cpu() 
        
        src_map = global2local.get(src, {})
        dst_map = global2local.get(dst, {})

        # Create lists to hold the new, localized indices
        new_src_indices = []
        new_dst_indices = []
        
        # Map the global IDs to local IDs
        # We process the edge index using NumPy/Python lists for speed in mapping
        global_src_ids = original_index_global[0].tolist()
        global_dst_ids = original_index_global[1].tolist()
        
        for global_src_id, global_dst_id in zip(global_src_ids, global_dst_ids):
            try:
                new_src_indices.append(src_map[global_src_id])
                new_dst_indices.append(dst_map[global_dst_id])
            except KeyError:
                # Skip the problematic edge if ID is missing in the map
                continue

        # 2. Overwrite the Original Edge with the localized index
        if not new_src_indices:
             print(f"Warning: No valid edges found for {original_edge_type}. Skipping.")
             continue
             
        new_index_local = torch.stack([
            torch.tensor(new_src_indices, dtype=torch.long),
            torch.tensor(new_dst_indices, dtype=torch.long)
        ]).to(device)

        data[original_edge_type].edge_index = new_index_local
        final_edge_types.append(original_edge_type)
        forward_edge_types.append(original_edge_type)

        
        # 3. Create the Inverse Edge
        inverse_rel = f'rev_{rel}'
        inverse_edge_type = (dst, inverse_rel, src)
        
        # The inverse index is the new_index_local flipped!
        # PyG automatically handles creating the new attribute slot.
        data[inverse_edge_type].edge_index = new_index_local.flip(0)
        final_edge_types.append(inverse_edge_type)
        
    print(f"✅ Localization complete. Forward edge types: {len(forward_edge_types)}, Total edge types: {len(final_edge_types)}")
    
    return data, forward_edge_types, final_edge_types


def split_and_prepare_features(
    data: HeteroData,
    hake_embed_dim: int = 32,
    default_fixed_dim: int = 1,
    default_fixed_value: float = 1.0,
) -> HeteroData:
    """
    Splits concatenated node features (.x) into fixed features (.fixed_x),
    trainable embeddings (.hake_embeds), and index tensors (.idx) for GNN input.
    
    It assumes all nodes have at least HAKE embeddings and explicitly sets 
    data[ntype].num_nodes to prevent split errors.

    Args:
        data (HeteroData): The PyG data object with concatenated features in .x.
        hake_embed_dim (int): The dimension of the HAKE embeddings.
        default_fixed_dim (int): Dimension for the placeholder fixed feature 
                                 if no original fixed features exist.
        default_fixed_value (float): Constant value for the placeholder.

    Returns:
        HeteroData: The modified data object.
    """
    
    print(f"Starting feature split (HAKE D={hake_embed_dim}, Placeholder D={default_fixed_dim})...")
    
    # Get the device from the data object
    device = data.device if hasattr(data, 'device') else torch.device('cpu')

    for ntype in data.node_types:
        # Check if the node type has features to process
        if not hasattr(data[ntype], 'x') or data[ntype].x is None:
             print(f"  - {ntype}: Skipping, no original features (.x) found.")
             continue

        full_tensor = data[ntype].x
        N_nodes = full_tensor.size(0)
        
        # CRITICAL: Explicitly set num_nodes to prevent errors in RandomLinkSplit
        data[ntype].num_nodes = N_nodes
        
        # Calculate the dimension of the fixed features (assuming HAKE is at the end)
        D_fixed = full_tensor.size(1) - hake_embed_dim

        # --- 1. Store HAKE embeddings (last HAKE_EMBED_DIM columns) ---
        # This is the part that will be made trainable in the FeatureManager
        data[ntype].hake_embeds = full_tensor[:, -hake_embed_dim:].clone().detach()

        # --- 2. Create the fixed_x tensor based on D_fixed ---
        if D_fixed > 0:
            # Case A: Node has fixed features (D_fixed > 0). Split the tensor.
            data[ntype].fixed_x = full_tensor[:, :D_fixed].clone().detach()
            print(f"  - {ntype}: Split features. D_fixed={D_fixed}.")

        else:
            # Case B: Node has ONLY HAKE features (D_fixed <= 0).
            # Create a minimal constant tensor for fixed_x to satisfy the GNN architecture.
            data[ntype].fixed_x = torch.full((N_nodes, default_fixed_dim), 
                                             default_fixed_value, 
                                             dtype=torch.float, 
                                             device=device)
            print(f"  - {ntype}: Only HAKE found. Assigned default fixed_x (D={default_fixed_dim}).")

        # --- 3. Create the index tensor (applies to all cases) ---
        # This tensor is used by the FeatureManager to look up the trainable HAKE embeddings
        data[ntype].idx = torch.arange(N_nodes, dtype=torch.long, device=device)
        
        # Remove the original concatenated .x to clean up the input
        del data[ntype].x
        
    print("✅ Feature preparation complete.")
    return data