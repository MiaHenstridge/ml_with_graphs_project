"""
Convert the heterogeneous graph (PyTorch tensors + ID mappings) to RDF format.

Entity types:
- Company
- StockSymbol
- Exchange
- Industry
- Sector
- Officer

Relationships:
- company --hasSymbol--> stockSymbol
- stockSymbol --listedOn--> exchange
- company --belongsTo--> industry
- industry --partOf--> sector
- company --employs--> officer
- institution --holds--> stockSymbol
"""

import json
import torch
import os
from pathlib import Path
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD
import logging
from urllib.parse import quote

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = './data'
ID_MAPPING = os.path.join(DATA_PATH, 'entity_id_map')
EDGE_INDEX = os.path.join(DATA_PATH, 'edge_index')
RDF_OUTPUT = os.path.join(DATA_PATH, 'rdf')
os.makedirs(RDF_OUTPUT, exist_ok=True)

# Define namespaces
GRAPH = Namespace("http://example.org/graph/")
COMPANY = Namespace("http://example.org/company/")
SYMBOL = Namespace("http://example.org/symbol/")
EXCHANGE = Namespace("http://example.org/exchange/")
INDUSTRY = Namespace("http://example.org/industry/")
SECTOR = Namespace("http://example.org/sector/")
OFFICER = Namespace("http://example.org/officer/")
INSTITUTION = Namespace("http://example.org/institution/")

# Define relationship predicates
HAS_SYMBOL = GRAPH.hasSymbol
LISTED_ON = GRAPH.listedOn
BELONGS_TO = GRAPH.belongsTo
PART_OF = GRAPH.partOf
EMPLOYS = GRAPH.employs
HOLDS = GRAPH.holds


def load_id_mapping(filename):
    """Load an ID mapping from JSON file."""
    filepath = os.path.join(ID_MAPPING, filename)
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return {}
    with open(filepath, 'r') as f:
        mapping = json.load(f)
    return {v: k for k, v in mapping.items()}  # Invert: id -> name


def create_uri(namespace, name):
    """Create a URI-safe identifier using URL encoding."""
    # URL-encode special characters, but preserve common ones
    safe_name = quote(str(name), safe='_-')
    return namespace[safe_name]


def add_entity_nodes(graph, namespace, id_mapping, entity_type):
    """Add entity nodes to the graph."""
    count = 0
    for entity_id, entity_name in id_mapping.items():
        uri = create_uri(namespace, entity_name)
        graph.add((uri, RDF.type, GRAPH[entity_type]))
        graph.add((uri, RDFS.label, Literal(entity_name)))
        count += 1
    return count


def add_edges_from_tensor(graph, tensor_path, src_mapping, dst_mapping, 
                         src_namespace, dst_namespace, predicate):
    """Load edges from a PyTorch tensor file and add to RDF graph."""
    if not os.path.exists(tensor_path):
        logger.warning(f"Tensor file not found: {tensor_path}")
        return 0
    
    edge_index = torch.load(tensor_path)
    src_ids = edge_index[0].tolist()
    dst_ids = edge_index[1].tolist()
    
    count = 0
    for src_id, dst_id in zip(src_ids, dst_ids):
        if src_id in src_mapping and dst_id in dst_mapping:
            src_uri = create_uri(src_namespace, src_mapping[src_id])
            dst_uri = create_uri(dst_namespace, dst_mapping[dst_id])
            graph.add((src_uri, predicate, dst_uri))
            count += 1
    
    return count


def main():
    """Build the complete RDF graph from PyTorch tensors and ID mappings."""
    
    logger.info("Creating RDF graph from heterogeneous graph...")
    
    # Initialize RDF graph with all namespaces
    graph = Graph()
    graph.bind('graph', GRAPH)
    graph.bind('company', COMPANY)
    graph.bind('symbol', SYMBOL)
    graph.bind('exchange', EXCHANGE)
    graph.bind('industry', INDUSTRY)
    graph.bind('sector', SECTOR)
    graph.bind('officer', OFFICER)
    graph.bind('institution', INSTITUTION)
    graph.bind('rdf', RDF)
    graph.bind('rdfs', RDFS)
    
    # Load all ID mappings (inverted: id -> name)
    logger.info("Loading entity ID mappings...")
    company2id = load_id_mapping('company2id.json')
    symbol2id = load_id_mapping('stocksymbol2id.json')
    exchange2id = load_id_mapping('exchange2id.json')
    industry2id = load_id_mapping('industry2id.json')
    sector2id = load_id_mapping('sector2id.json')
    officer2id = load_id_mapping('officer2id.json')
    institution2id = load_id_mapping('institution2id.json')
    
    # Add entity nodes
    logger.info("Adding entity nodes...")
    c = add_entity_nodes(graph, COMPANY, company2id, 'Company')
    logger.info(f"  Added {c} company nodes")
    
    c = add_entity_nodes(graph, SYMBOL, symbol2id, 'StockSymbol')
    logger.info(f"  Added {c} stock symbol nodes")
    
    c = add_entity_nodes(graph, EXCHANGE, exchange2id, 'Exchange')
    logger.info(f"  Added {c} exchange nodes")
    
    c = add_entity_nodes(graph, INDUSTRY, industry2id, 'Industry')
    logger.info(f"  Added {c} industry nodes")
    
    c = add_entity_nodes(graph, SECTOR, sector2id, 'Sector')
    logger.info(f"  Added {c} sector nodes")
    
    c = add_entity_nodes(graph, OFFICER, officer2id, 'Officer')
    logger.info(f"  Added {c} officer nodes")
    
    c = add_entity_nodes(graph, INSTITUTION, institution2id, 'Institution')
    logger.info(f"  Added {c} institution nodes")
    
    # Add edges from tensors
    logger.info("Adding edges from tensor files...")
    
    # company --hasSymbol--> stockSymbol
    c = add_edges_from_tensor(
        graph,
        os.path.join(EDGE_INDEX, 'comp2sym.pt'),
        company2id, symbol2id,
        COMPANY, SYMBOL,
        HAS_SYMBOL
    )
    logger.info(f"  Added {c} hasSymbol edges")
    
    # stockSymbol --listedOn--> exchange
    c = add_edges_from_tensor(
        graph,
        os.path.join(EDGE_INDEX, 'sym2ex.pt'),
        symbol2id, exchange2id,
        SYMBOL, EXCHANGE,
        LISTED_ON
    )
    logger.info(f"  Added {c} listedOn edges")
    
    # company --belongsTo--> industry
    c = add_edges_from_tensor(
        graph,
        os.path.join(EDGE_INDEX, 'comp2ind.pt'),
        company2id, industry2id,
        COMPANY, INDUSTRY,
        BELONGS_TO
    )
    logger.info(f"  Added {c} belongsTo edges")
    
    # industry --partOf--> sector
    c = add_edges_from_tensor(
        graph,
        os.path.join(EDGE_INDEX, 'ind2sec.pt'),
        industry2id, sector2id,
        INDUSTRY, SECTOR,
        PART_OF
    )
    logger.info(f"  Added {c} partOf edges")
    
    # company --employs--> officer
    c = add_edges_from_tensor(
        graph,
        os.path.join(EDGE_INDEX, 'comp2off.pt'),
        company2id, officer2id,
        COMPANY, OFFICER,
        EMPLOYS
    )
    logger.info(f"  Added {c} employs edges")
    
    # institution --holds--> stockSymbol
    c = add_edges_from_tensor(
        graph,
        os.path.join(EDGE_INDEX, 'inst2sym.pt'),
        institution2id, symbol2id,
        INSTITUTION, SYMBOL,
        HOLDS
    )
    logger.info(f"  Added {c} holds edges")
    
    # Save as Turtle format
    output_path = os.path.join(RDF_OUTPUT, 'heterogeneous_graph.ttl')
    logger.info(f"Serializing RDF graph to {output_path}...")
    graph.serialize(destination=output_path, format='turtle')
    
    # Also save as N-Triples for compatibility
    nt_path = os.path.join(RDF_OUTPUT, 'heterogeneous_graph.nt')
    graph.serialize(destination=nt_path, format='nt')
    
    # Print graph statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"RDF Graph Statistics")
    logger.info(f"{'='*60}")
    logger.info(f"Total triples: {len(graph)}")
    logger.info(f"Unique subjects: {len(set(s for s, _, _ in graph))}")
    logger.info(f"Unique predicates: {len(set(p for _, p, _ in graph))}")
    logger.info(f"Unique objects: {len(set(o for _, _, o in graph))}")
    logger.info(f"\nRDF files saved to: {RDF_OUTPUT}")
    logger.info(f"  - Turtle format: {output_path}")
    logger.info(f"  - N-Triples format: {nt_path}")
    logger.info(f"{'='*60}\n")
    
    return graph


if __name__ == '__main__':
    graph = main()
