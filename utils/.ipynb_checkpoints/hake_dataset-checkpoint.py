import os
import json
from collections import defaultdict

def invert_nested_dict(nested_dict):
    """
    Inverts the inner dictionaries of a nested dict.
    
    Input: {entity: {local_id: global_id}}
    Output: {entity: {global_id: local_id}}
    """
    return {outer_k: {v: k for k, v in inner_d.items()} 
            for outer_k, inner_d in nested_dict.items()}

def invert_dict(dict_):
    return {v: k for k, v in dict_.items()}

def make_hake_triples(id_mapping_dir, edge_index_dir):
    """
    Prepare triples input: list of (head, relation, tail) for making hake dataset
    """
    # os.makedirs(out_dir, exist_ok=True)
    
    ########### get global triples ##########
    with open(os.path.join(edge_index_dir, 'global_triples.json')) as f:
        global_triples = json.load(f)

    ############ get global entity type mapping ###########
    with open(os.path.join(id_mapping_dir, 'global_type_map.json')) as f:
        global_type = json.load(f)

    ########## get local to global id mapping ############
    with open(os.path.join(id_mapping_dir, 'global_id.json')) as f:
        global_id = json.load(f)

    # convert to global to local id mapping
    local_id = invert_nested_dict(global_id)

    ########### grab entity id mappings ###################
    with open(os.path.join(id_mapping_dir, 'company2id.json')) as f:
        company2id = json.load(f)
    
    with open(os.path.join(id_mapping_dir, 'stocksymbol2id.json')) as f:
        stocksymbol2id = json.load(f)
    
    with open(os.path.join(id_mapping_dir, 'industry2id.json')) as f:
        industry2id = json.load(f)
    
    with open(os.path.join(id_mapping_dir, 'sector2id.json')) as f:
        sector2id = json.load(f)
    
    with open(os.path.join(id_mapping_dir, 'institution2id.json')) as f:
        institution2id = json.load(f)
    
    with open(os.path.join(id_mapping_dir, 'mutualfund2id.json')) as f:
        fund2id = json.load(f)
    
    with open(os.path.join(id_mapping_dir, 'fundsymbol2id.json')) as f:
        fundsymbol2id = json.load(f)

    id_dict_map = {
        'company': invert_dict(company2id), 
        'stock_symbol': invert_dict(stocksymbol2id), 
        'industry': invert_dict(industry2id), 
        'sector': invert_dict(sector2id),
        'institution': invert_dict(institution2id), 
        'fund': invert_dict(fund2id), 
        'fund_symbol': invert_dict(fundsymbol2id),
    }

    hake_triples = []
    # entity2id = {}
    # rel2id = {}

    for (h, r, t) in global_triples:
        h_type_ = global_type[str(h)]
        t_type_ = global_type[str(t)]
        
        # get local ids of h and t w.r.t their entity type
        h_local_ = local_id[h_type_][h]
        t_local_ = local_id[t_type_][t]
        try:
            # get name of h and t using their local ids
            h_name = id_dict_map[h_type_][int(h_local_)]
            t_name = id_dict_map[t_type_][int(t_local_)]
    
            hake_triples.append((h_name, r, t_name))
        except KeyError:
            print(f"Head node: Type {h_type_}, local id: {h_local_}")
            print(f"Tail node: Type {t_type_}, local id: {t_local_}")
            return

    return hake_triples
    


def make_hake_dataset(triples, out_dir):
    """
    triples: list of (head, relation, tail) — all strings
    out_dir: directory where HAKE-formatted files will be written

    Produces:
        entities.dict      --> global_entity_id \t entity_name
        relations.dict     --> relation_id \t relation_name
        train.txt          --> head_name \t relation_name \t tail_name
    """

    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. Build entity dictionary ----
    entity_set = set()
    relation_set = set()

    for h, r, t in triples:
        entity_set.add(h)
        entity_set.add(t)
        relation_set.add(r)

    entity_list = sorted(list(entity_set))
    relation_list = sorted(list(relation_set))

    entity2id = {e: i for i, e in enumerate(entity_list)}
    rel2id = {r: i for i, r in enumerate(relation_list)}

    # ---- 2. Write entities.dict ----
    with open(os.path.join(out_dir, "entities.dict"), "w", encoding="utf-8") as f:
        for entity, eid in entity2id.items():
            f.write(f"{eid}\t{entity}\n")

    # ---- 3. Write relations.dict ----
    with open(os.path.join(out_dir, "relations.dict"), "w", encoding="utf-8") as f:
        for rel, rid in rel2id.items():
            f.write(f"{rid}\t{rel}\n")

    # ---- 4. Write train.txt (HAKE format uses names, not IDs) ----
    with open(os.path.join(out_dir, "train.txt"), "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")

    return
