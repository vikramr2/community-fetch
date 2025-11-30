# Community Fetch

A hierarchical retrieval system combining Paris clustering (hierarchical) and IKC clustering (community detection) with MedCPT embeddings for efficient semantic search.

## Overview

This system enables efficient semantic search over large document collections by:
1. Building a hierarchical dendrogram for fast top-down navigation
2. Using community detection to identify relevant document clusters
3. Leveraging MedCPT embeddings for biomedical text similarity

## Pipeline

### Phase 1: Indexing

#### 1. Compute Paris Clustering
Paris algorithm creates a hierarchical dendrogram of the document network based on graph structure.

**Input**: Edge list (`data/network/oc_mini_edgelist.tsv`)
**Output**: Hierarchical tree (`data/clustering/oc_mini_paris.json`)

```bash
# Run Paris clustering
python indexing/paris.py
```

#### 2. Compute IKC Clustering
IKC (Iterative K-core) algorithm identifies tightly-knit communities at different k-core levels.

**Input**: Edge list (`data/network/oc_mini_edgelist.tsv`)
**Output**: Cluster assignments (`data/clustering/oc_mini_ikc.csv`)

```bash
# Run IKC clustering
python indexing/run_ikc.py
```

#### 3. Reroot Paris Clustering for Balance
Reroots the dendrogram to minimize tree height imbalance, improving retrieval efficiency.

**Input**: Original Paris tree (`data/clustering/oc_mini_paris.json`)
**Output**: Rebalanced tree (`data/clustering/oc_mini_paris_rebalanced.json`)

```bash
# Reroot the tree
python indexing/reroot_paris.py
```

#### 4. Compute Tree Embeddings
Computes MedCPT embeddings for leaf nodes (documents) and propagates them upward via averaging for internal nodes.

**Input**:
- Rerooted tree (`data/clustering/oc_mini_paris_rebalanced.json`)
- Node metadata (`data/metadata/oc_mini_node_metadata.csv`)

**Output**: Tree embeddings (`data/clustering/oc_mini_paris_embeddings.h5`)

```bash
# Install dependencies
pip install transformers torch h5py tqdm

# Compute embeddings
python indexing/compute_tree_embeddings.py
```

**Embedding details**:
- Model: `ncbi/MedCPT-Query-Encoder`
- Text: Concatenation of title + abstract
- Leaf embeddings: Direct encoding from MedCPT
- Internal node embeddings: Average of children embeddings

### Phase 2: Retrieval (Planned)

#### 1. Encode Query
Encode the user's input query using the same MedCPT model.

```python
query_embedding = encode_query(user_query)
```

#### 2. Layer-by-Layer Tree Traversal
Navigate down the rerooted Paris tree using cosine similarity at each layer.

```python
# Start at root
current_node = root

# At each layer, choose child with highest cosine similarity
while current_node is not a leaf:
    children_similarities = [
        cosine_similarity(query_embedding, child.embedding)
        for child in current_node.children
    ]
    current_node = current_node.children[argmax(children_similarities)]

# Found the most relevant leaf node
target_node = current_node
```

#### 3. Community Expansion
Use IKC clustering to find the community of the retrieved node.

```python
# Look up the node's cluster in IKC results
target_cluster = ikc_clusters[target_node.id]

# Retrieve all nodes in the same community
community_nodes = [
    node for node in all_nodes
    if ikc_clusters[node.id] == target_cluster
]

# Return community as final results
return community_nodes
```

## Data Structure

```
data/
├── network/
│   └── oc_mini_edgelist.tsv           # Graph structure
├── metadata/
│   └── oc_mini_node_metadata.csv      # Document metadata (id, doi, title, abstract)
└── clustering/
    ├── oc_mini_paris.json             # Original Paris dendrogram
    ├── oc_mini_paris_rebalanced.json  # Rebalanced dendrogram
    ├── oc_mini_paris_embeddings.h5    # Tree embeddings (HDF5)
    └── oc_mini_ikc.csv                # IKC cluster assignments
```

## File Formats

### Paris JSON Structure
```json
{
  "algorithm": "Paris",
  "num_nodes": 14384,
  "num_edges": 111873,
  "hierarchy": {
    "id": 28766,
    "type": "cluster",
    "distance": 2129.49,
    "count": 14384,
    "children": [...]
  }
}
```

### IKC CSV Structure
```csv
node_id,cluster_id,k_value,modularity
128,1,5,0.823
163,1,5,0.823
...
```

### Embeddings HDF5 Structure
```
/embeddings:    (num_nodes, embedding_dim) - Embedding matrix
/node_ids:      (num_nodes,) - Node IDs for each row
/attributes:    num_nodes, embedding_dim, algorithm, etc.
```

## Loading Embeddings

```python
import h5py

with h5py.File('data/clustering/oc_mini_paris_embeddings.h5', 'r') as f:
    embeddings = f['embeddings'][:]  # numpy array
    node_ids = f['node_ids'][:]

    # Create lookup dictionary
    embedding_dict = {nid: emb for nid, emb in zip(node_ids, embeddings)}
```

## Requirements

```bash
pip install transformers torch h5py tqdm numpy
```

## References

- **Paris**: Hierarchical graph clustering optimizing modularity
- **IKC**: Iterative k-core based community detection
- **MedCPT**: Medical domain-specific contrastive pre-trained transformer
