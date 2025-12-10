import os
import json
import torch
from collections import defaultdict
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import HeteroData # Ensure this is imported

# set seed
from torch_geometric import seed_everything
seed_everything(42)


# --- PROVIDED FUNCTIONS (NO MODIFICATION) ---

def invert_nested_dict(nested_dict):
    """Inverts the inner dictionaries of a nested dict."""
    return {outer_k: {str(v): str(k) for k, v in inner_d.items()} 
            for outer_k, inner_d in nested_dict.items()}

def invert_dict(dict_):
    return {v: k for k, v in dict_.items()}


# ----------- id mappings ----------
ID_MAPPING_DIR = './data/entity_id_map'
############ get global entity type mapping ###########
with open(os.path.join(ID_MAPPING_DIR, 'global_type_map.json')) as f:
    global_type = json.load(f)

########## get local to global id mapping ############
with open(os.path.join(ID_MAPPING_DIR, 'global_id.json')) as f:
    global_id = json.load(f)

# convert to global to local id mapping
LOCAL_ID = invert_nested_dict(global_id)

########### grab entity id mappings ###################
with open(os.path.join(ID_MAPPING_DIR, 'company2id.json')) as f:
    company2id = json.load(f)

with open(os.path.join(ID_MAPPING_DIR, 'stocksymbol2id.json')) as f:
    stocksymbol2id = json.load(f)

with open(os.path.join(ID_MAPPING_DIR, 'industry2id.json')) as f:
    industry2id = json.load(f)

with open(os.path.join(ID_MAPPING_DIR, 'sector2id.json')) as f:
    sector2id = json.load(f)

with open(os.path.join(ID_MAPPING_DIR, 'institution2id.json')) as f:
    institution2id = json.load(f)

with open(os.path.join(ID_MAPPING_DIR, 'mutualfund2id.json')) as f:
    fund2id = json.load(f)

with open(os.path.join(ID_MAPPING_DIR, 'fundsymbol2id.json')) as f:
    fundsymbol2id = json.load(f)

ID_DICT_MAP = {
    'company': invert_dict(company2id), 
    'stock_symbol': invert_dict(stocksymbol2id), 
    'industry': invert_dict(industry2id), 
    'sector': invert_dict(sector2id),
    'institution': invert_dict(institution2id), 
    'fund': invert_dict(fund2id), 
    'fund_symbol': invert_dict(fundsymbol2id),
}


def get_entity_lid_name(gid, entity_type):
    # get local id 
    lid = LOCAL_ID[entity_type][str(gid)]
    # get entity name from local id
    name = ID_DICT_MAP[entity_type][int(lid)]
    return lid, name

# ------------- Prepare data: localize indices and create reverse edges -------------

def prepare_data(hetero_data: HeteroData):
    """
    1. Infer the correct local node counts (num_nodes) by examining all edge IDs 
       and mapping them to the local ID system.
    2. Convert all global edge IDs to local IDs in-place.
    
    Args:
        hetero_data (HeteroData): The graph data object (modified in-place).
    """
    EDGE_TYPES_ORIGINAL = list(hetero_data.edge_index_dict.keys())
    
    # --- PASS 1: INFER AND SET FINAL LOCAL NODE COUNTS (The essential fix) ---
    
    unique_global_ids = defaultdict(set)
    final_local_node_counts = {}
    
    # 1. Collect all unique global IDs involved in the graph
    for src, rel, dst in EDGE_TYPES_ORIGINAL:
        edge_index = hetero_data[(src, rel, dst)].edge_index.cpu() 
        unique_global_ids[src].update(edge_index[0].unique().tolist())
        unique_global_ids[dst].update(edge_index[1].unique().tolist())
        
    # 2. Determine and set num_nodes for validation/splitting
    for ntype, global_ids_set in unique_global_ids.items():
        max_local_id = -1
        
        for global_id in global_ids_set:
            try:
                local_id_str, _ = get_entity_lid_name(global_id, ntype)
                local_id = int(local_id_str)
                if local_id > max_local_id:
                    max_local_id = local_id
            except (KeyError, ValueError):
                continue
                
        count = max_local_id + 1
        final_local_node_counts[ntype] = count
        
        # # CRITICAL FIX: Explicitly set num_nodes for the node type
        # if ntype not in hetero_data:
        #     hetero_data[ntype] = {} # Initialize storage if it doesn't exist

        hetero_data[ntype].num_nodes = count
        
        # Defensive fix: Ensure an empty feature tensor exists if none were loaded
        if not hasattr(hetero_data[ntype], 'x') or hetero_data[ntype].x is None:
             hetero_data[ntype].x = torch.empty(count, 0, dtype=torch.float)

    # -------------------------------------------------------------------

    # --- PASS 2: LOCALIZATION (Your original, working logic) ---
    
    for edge_type in EDGE_TYPES_ORIGINAL:
        htype, rtype, ttype = edge_type
        
        head_ids, tail_ids = hetero_data.edge_index_dict[edge_type]
        target_device = hetero_data[edge_type].edge_index.device
        
        # Move global indices to CPU for efficient dictionary lookup
        head_ids_cpu = head_ids.cpu()
        tail_ids_cpu = tail_ids.cpu()

        new_src_indices = []
        new_dst_indices = []
        
        for h, t in zip(head_ids_cpu, tail_ids_cpu):
            try:
                h_local_id, _ = get_entity_lid_name(h.item(), htype)
                t_local_id, _ = get_entity_lid_name(t.item(), ttype)
                
                new_src_indices.append(int(h_local_id))
                new_dst_indices.append(int(t_local_id))
            except (KeyError, ValueError):
                # Skip edge if ID is unmapped or unparsable
                continue
            
        # 2. Overwrite the Original Edge with the localized index
        new_index_local = torch.stack([
            torch.tensor(new_src_indices, dtype=torch.long),
            torch.tensor(new_dst_indices, dtype=torch.long)
        ]).to(target_device) # Move back to the original device

        hetero_data[edge_type].edge_index = new_index_local

    # Final validation check
    hetero_data.validate()
    return hetero_data
    
    
# ------ CREATE entities.dict and relations.dict from full graph ----------
def create_hake_dicts(hetero_data, out_dir):
    entities = set()
    relations = set()
    entity2gid = {}
    for edge_type in hetero_data.edge_index_dict.keys():
        htype, rtype, ttype = edge_type
        relations.add(rtype)
        # grab (head id, tail id)
        head_ids, tail_ids = hetero_data.edge_index_dict[edge_type]
        for h, t in zip(head_ids, tail_ids):
            _, h_name = get_entity_lid_name(h.item(), htype)
            _, t_name = get_entity_lid_name(t.item(), ttype)

            # add to entities set
            entities.add(h_name)
            entities.add(t_name)
            # add to entity name to gid mapping
            if h_name not in entity2gid:
                entity2gid[h_name] = int(h.item())
            if t_name not in entity2gid:
                entity2gid[t_name] = int(t.item())
                
    # Write entities.dict
    with open(os.path.join(out_dir, "entities.dict"), "w", encoding="utf-8") as f:
        for eid, entity in enumerate(entities):
            f.write(f"{eid}\t{entity}\n")
    # Write relations.dict
    with open(os.path.join(out_dir, "relations.dict"), "w", encoding="utf-8") as f:
        for rid, relation in enumerate(relations):
            f.write(f"{rid}\t{relation}\n")

    return entity2gid


# --------- create train.txt, valid.txt and test.txt ------------
def split_data(data, num_val, num_test, edge_types):
    transform = RandomLinkSplit(
        num_val=num_val,
        num_test=num_test,
        add_negative_train_samples=False,
        neg_sampling_ratio=0.0,
        is_undirected=True, # CRITICAL: Ensures symmetric splitting
        edge_types=edge_types
    )
    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data


def create_hake_txt_files(hetero_data, num_val, num_test, out_dir):
    # localize ids before splitting the data
    data = prepare_data(hetero_data)
    # split data
    train_data, val_data, test_data = split_data(data, num_val, num_test, edge_types=list(data.edge_index_dict.keys()))
    
    for data_split, filename in zip([train_data, val_data, test_data], ['train.txt', 'valid.txt', 'test.txt']):
        triples = []
        for edge_type in list(data_split.edge_index_dict.keys()):
            htype, rtype, ttype = edge_type
            # grab (head id, tail id) - localized
            head_ids, tail_ids = data_split.edge_index_dict[edge_type]
            # grab name of entites
            for h, t in zip(head_ids, tail_ids):
                h_name = ID_DICT_MAP[htype][int(h)]
                t_name = ID_DICT_MAP[ttype][int(t)]
                triples.append((h_name, rtype, t_name))
            # write to file
            with open(os.path.join(out_dir, filename), 'w', encoding="utf-8") as f:
                for h, r, t in triples:
                    f.write(f"{h}\t{r}\t{t}\n")

    return
        
    
            
            
    