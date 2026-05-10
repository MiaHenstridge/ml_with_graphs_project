# Hierarchy-Aware Link Prediction in Financial Knowledge Graphs

Predicting latent fund–stock and institution–stock holdings using hierarchy-aware GNN embeddings on a heterogeneous financial knowledge graph.

📄 [Paper](https://miaworld.neocities.org/files/hierarchy_aware_link_prediction_financialKG.pdf)

## Overview

Financial markets are organized in natural hierarchies (Sector → Industry → Company), but standard graph embeddings treat all entities as flat. This project builds a heterogeneous knowledge graph of ~20K entities from Yahoo Finance data and applies **HAKE** (Hierarchy-Aware Knowledge Graph Embeddings) to encode these taxonomic relationships in polar coordinates, then uses a **GraphSAGE** encoder for link prediction across three tasks: Fund→Stock, Institution→Stock, and Combined.

## Key Results

| Task | Best AUC |
|------|----------|
| Fund → Stock | 0.969 |
| Institution → Stock | 0.943 |
| Combined | 0.957 |

fANOVA analysis over 1,200 Optuna trials revealed that the GNN operator choice alone accounts for 26% of performance variance, and uncovered a "deep vs. wide" architectural split: specialist mutual funds are best modeled by deeper networks (3 layers), while generalist institutions favor wider networks (120 channels).

## Knowledge Graph Schema

7 node types (Company, StockSymbol, Industry, Sector, Institution, Fund, FundSymbol) connected by 4 edge types (`hasSymbol`, `holds`, `isPartOf`, `belongsTo`).

## Project Structure

```
00_data_collection.ipynb                         # Scrape holdings and metadata from Yahoo Finance
01_create_network.ipynb                          # Build the heterogeneous knowledge graph
02_hake_embeddings.ipynb                         # Train HAKE embeddings in polar coordinates
03_link_prediction_with_hyperparameter_tuning.ipynb  # Link prediction + Optuna hyperparameter search
data/                                            # Raw and processed graph data
hake/                                            # HAKE model implementation
utils/                                           # Helper functions
```

## Technologies

PyTorch, PyTorch Geometric, Optuna, yfinance, NetworkX
