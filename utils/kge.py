from typing import *
from torch_geometric.data import HeteroData
import torch
from collections import defaultdict

def build_global_id_map(entity_id_maps, entity_types):
    """
    entity_id_maps: list of dicts [{local_name: local_id}, ...]
    entity_types: list of strings ["company", "symbol", "officer"]

    Returns:
        global_id_map: {entity_type: {local_name: global_id}}
        type_map: {global_id: entity_type}
        offsets: {entity_type: offset_start}
    """
    assert len(entity_id_maps) == len(entity_types), \
        "entity_id_maps and entity_types must be the same length"

    global_id_map = {}
    type_map = {}
    offsets = {}

    current_offset = 0

    for local_map, entity_type in zip(entity_id_maps, entity_types):
        offsets[entity_type] = current_offset
        global_id_map[entity_type] = {}

        for name, local_id in local_map.items():
            global_id = current_offset + local_id
            global_id_map[entity_type][local_id] = global_id
            type_map[global_id] = entity_type

        current_offset += len(local_map)

    return global_id_map, type_map, offsets


def build_global_triples(
    edge_indices_list: List[Dict],
    entity2global: Dict[str, Dict[int, int]]
    ) -> List[Tuple[int, str, int]]:
    """
    Convert local edge indices (PyTorch tensors) of multiple relation types to global triples.
    
    Parameters:
    - edge_indices_list: list of dicts, each with:
        {
            "relation": str,
            "head_type": str,
            "tail_type": str,
            "edge_index": torch.LongTensor of shape [2, num_edges] (head, tail)
        }
    - entity2global: dict mapping entity type -> local_id -> global_id
    
    Returns:
    - List of triples: (global_head_id, relation_type_str, global_tail_id)
    """
    global_triples = []
    
    for rel in edge_indices_list:
        relation = rel["relation"]
        head_type = rel["head_type"]
        tail_type = rel["tail_type"]
        edge_index = rel["edge_index"]  # [2, num_edges]
        
        # Convert local IDs to global IDs
        h_global = [entity2global[head_type][h.item()] for h in edge_index[0]]
        t_global = [entity2global[tail_type][t.item()] for t in edge_index[1]]
        
        # Append triples
        global_triples.extend([(h, relation, t) for h, t in zip(h_global, t_global)])
    
    return global_triples


def build_hetero_graph(global_triples, type_map):
    """
    global_triples: List of (head_id, relation, tail_id)
    type_map: dict {global_id: entity_type}
    
    Returns:
    - HeteroData object
    """
    # Group edges by (head_type, relation, tail_type)
    edge_dict = defaultdict(list)
    for h, r, t in global_triples:
        head_type = type_map[str(h)]
        tail_type = type_map[str(t)]
        edge_dict[(head_type, r, tail_type)].append((h, t))
    
    data = HeteroData()
    
    # Add edges per relation
    for (src_type, rel, dst_type), edges in edge_dict.items():
        edges = torch.tensor(edges, dtype=torch.long).t().contiguous()  # shape [2, num_edges]
        data[src_type, rel, dst_type].edge_index = edges
    
    return data
