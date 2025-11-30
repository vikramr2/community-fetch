#!/usr/bin/env python3
"""
Hierarchical search through the tree using cosine similarity.
Given a query, embed it with MedCPT and navigate down the tree
layer by layer, selecting the child with highest cosine similarity
at each level until reaching a leaf node.
"""
import json
import csv
import os
import h5py
import numpy as np
import argparse
from typing import Dict, Any
import ikc


def load_metadata(metadata_file: str) -> Dict[str, Dict[str, str]]:
    """
    Load metadata CSV and return a dictionary mapping node IDs to metadata.

    Returns:
        Dict mapping node_id -> {title, abstract, doi}
    """
    metadata = {}
    with open(metadata_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row['id']
            metadata[node_id] = {
                'title': row.get('title', ''),
                'abstract': row.get('abstract', ''),
                'doi': row.get('doi', '')
            }
    return metadata


def load_embeddings(h5_file: str) -> Dict[int, np.ndarray]:
    """
    Load embeddings from HDF5 file.

    Returns:
        Dict mapping node_id -> embedding vector
    """
    embeddings = {}
    with h5py.File(h5_file, 'r') as f:
        embedding_matrix = f['embeddings'][:]
        node_ids = f['node_ids'][:]

        for node_id, embedding in zip(node_ids, embedding_matrix):
            embeddings[int(node_id)] = embedding

    return embeddings


def load_hierarchy(hierarchy_file: str) -> Dict[str, Any]:
    """Load the hierarchy JSON file."""
    with open(hierarchy_file, 'r') as f:
        hierarchy_data = json.load(f)
    return hierarchy_data['hierarchy']


def embed_query(query: str, model_name: str = "ncbi/MedCPT-Query-Encoder") -> np.ndarray:
    """
    Embed a query string using MedCPT.

    Args:
        query: The query text to embed
        model_name: HuggingFace model name for MedCPT

    Returns:
        Embedding vector as numpy array
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except ImportError:
        print("Error: transformers and torch are required.")
        print("Install with: pip install transformers torch")
        exit(1)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Tokenize and encode
    with torch.no_grad():
        inputs = tokenizer(query, return_tensors="pt", padding=True,
                         truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get embeddings (use CLS token)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

    return embedding


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def hierarchical_search(query_embedding: np.ndarray,
                       root: Dict[str, Any],
                       embeddings: Dict[int, np.ndarray],
                       verbose: bool = True) -> Dict[str, Any]:
    """
    Navigate down the tree selecting the child with highest cosine similarity
    at each level until reaching a leaf node.

    Args:
        query_embedding: The embedded query vector
        root: Root node of the hierarchy
        embeddings: Dict mapping node_id -> embedding vector
        verbose: Whether to print progress

    Returns:
        The final leaf node selected
    """
    current_node = root
    depth = 0

    if verbose:
        print("\n" + "="*60)
        print("Hierarchical Search")
        print("="*60)

    while current_node.get('type') != 'leaf':
        children = current_node.get('children', [])

        if not children:
            print(f"Warning: Internal node {current_node['id']} has no children")
            break

        # Compute cosine similarity for each child
        similarities = []
        for child in children:
            child_id = child['id']
            if child_id in embeddings:
                child_embedding = embeddings[child_id]
                similarity = cosine_similarity(query_embedding, child_embedding)
                similarities.append((similarity, child))
            else:
                print(f"Warning: No embedding found for node {child_id}")
                similarities.append((float('-inf'), child))

        # Select child with highest similarity
        best_similarity, best_child = max(similarities, key=lambda x: x[0])

        if verbose:
            print(f"\nDepth {depth}: Node {current_node['id']} ({current_node.get('type', 'unknown')})")
            print(f"  Children: {len(children)}")
            print(f"  Selected: Node {best_child['id']} (similarity: {best_similarity:.4f})")

            # Show top 3 children
            top_3 = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]
            for i, (sim, child) in enumerate(top_3, 1):
                marker = "→" if child == best_child else " "
                print(f"  {marker} {i}. Node {child['id']}: {sim:.4f}")

        current_node = best_child
        depth += 1

    if verbose:
        print(f"\nDepth {depth}: Node {current_node['id']} (LEAF)")
        print("="*60)

    return current_node


def main():
    parser = argparse.ArgumentParser(description='Hierarchical search through tree using MedCPT')
    parser.add_argument('query', type=str, help='Query string to search for')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    # File paths
    data_dir = args.data_dir
    metadata_file = os.path.join(data_dir, 'metadata/oc_mini_node_metadata.csv')
    hierarchy_file = os.path.join(data_dir, 'clustering/oc_mini_paris_rebalanced.json')
    embeddings_file = os.path.join(data_dir, 'clustering/oc_mini_paris_embeddings.h5')
    ikc_file = os.path.join(data_dir, 'clustering/oc_mini_ikc.csv')
    network_file = os.path.join(data_dir, 'network/oc_mini_edgelist.csv')
    network_file_tsv = os.path.join(data_dir, 'network/oc_mini_edgelist.tsv')

    verbose = not args.quiet

    if verbose:
        print("Loading data...")

    # Load all data
    metadata = load_metadata(metadata_file)
    if verbose:
        print(f"  Loaded metadata for {len(metadata)} nodes")

    hierarchy = load_hierarchy(hierarchy_file)
    if verbose:
        print(f"  Loaded hierarchy")

    embeddings = load_embeddings(embeddings_file)
    if verbose:
        print(f"  Loaded {len(embeddings)} embeddings")

    # Embed query
    if verbose:
        print(f"\nEmbedding query: '{args.query}'")
    query_embedding = embed_query(args.query)
    if verbose:
        print(f"  Query embedding shape: {query_embedding.shape}")

    # Perform hierarchical search
    final_node = hierarchical_search(query_embedding, hierarchy, embeddings, verbose=verbose)

    # Get metadata for final node
    node_name = final_node.get('name')

    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(f"Node ID: {final_node['id']}")
    print(f"Node Name: {node_name}")

    if node_name and node_name in metadata:
        meta = metadata[node_name]
        print(f"\nTitle: {meta['title']}")
        print(f"\nAbstract: {meta['abstract']}")
        print(f"\nDOI: {meta['doi']}")
    else:
        print(f"\nWarning: No metadata found for node name '{node_name}'")
    print("="*60)

    # Load IKC and network data
    if verbose:
        print("\nLoading IKC and network data...")
    g = ikc.load_graph(network_file_tsv)
    kcore = g.compute_kcore_decomposition()

    if verbose:
        print(f"  Loaded graph with {g.num_nodes} nodes, {g.num_edges} edges")
        print(f"  Computed IKC clusters: {len(kcore)} clusters found")
        print("  Finding maximal k-core for final node...")

    community = g.find_maximal_kcore(int(final_node['id']), core_numbers=kcore.core_numbers)
    
    # print(community)
    if community:
        print(f"\nCommunity for node {final_node['id']} (k={community['k']}):")
        print(f"  Number of nodes in community: {len(community['nodes'])}")
        print("  Saving community to 'fetched_community.json'...")
        with open('fetched_community.json', 'w') as f:
            json.dump({
                'k': community['k'],
                'nodes': community['nodes']
            }, f, indent=2)
    else: 
        print(f"\nNo community found for node {final_node['id']}.")

if __name__ == "__main__":
    main()
