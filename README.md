# Community Fetch

A hierarchical retrieval system combining Paris clustering and k-core community detection with MedCPT embeddings for efficient semantic search over citation networks.

## Method Overview

The system uses a two-stage approach for semantic search:

1. **Hierarchical Search**: A Paris clustering tree organizes papers hierarchically based on citation structure. Leaves are embedded with MedCPT and embeddings are averaged upward through the internal nodes. Given a query, we embed it using MedCPT and navigate down the tree by selecting the child with highest cosine similarity at each level until reaching a leaf node.

2. **Community Detection**: From the retrieved leaf node, we extract its k-core community using IKC (Iterative K-Core) decomposition. This finds the maximal densely-connected subgraph containing the target paper, representing related work in the same research area.

The result is a focused community of papers semantically similar to the query, leveraging both embedding similarity and citation network structure.

## Fetching and Visualizing

### Fetch a Community

Run hierarchical search and extract k-core community:

```bash
python fetch.py "your query here"
```

This outputs:
- The retrieved paper's metadata (title, abstract, DOI)
- Community statistics (k-value, number of nodes)
- `fetched_community.json` containing the community structure

Optional flags:
- `--wcc`: Refine community using well-connected components
- `--quiet`: Suppress progress output
- `--data-dir`: Specify data directory (default: `data/`)

### Visualize the Community

Create an interactive force-directed network visualization:

```bash
python visualize_community.py "your query here"
```

This generates `community_network.html` showing:
- Nodes colored by cosine similarity rank to the query
- Central/query paper marked with a star symbol
- Hover labels with paper title, abstract, and similarity score
- Force-directed layout revealing community structure

Optional flags:
- `--community-file`: JSON file with community data (default: `fetched_community.json`)
- `--output`: Output HTML file (default: `community_network.html`)
- `--data-dir`: Data directory (default: `data/`)

## References

- **MedCPT**: Jin et al. (2023). "MedCPT: Contrastive Pre-trained Transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval." [ncbi/MedCPT-Query-Encoder](https://huggingface.co/ncbi/MedCPT-Query-Encoder)
- **Paris Clustering**: Bonald et al. (2018). "Hierarchical Graph Clustering using Node Pair Sampling."
- **IKC**: Wedell et al. (2022). "Center–periphery structure in research communities."
- **WCC**: Park et. al. (2024). "Improved community detection using stochastic block models."
