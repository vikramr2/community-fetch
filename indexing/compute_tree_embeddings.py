#!/usr/bin/env python3
"""
Compute MedCPT embeddings for leaf nodes from metadata,
then propagate embeddings up the tree hierarchy.
"""
import json
import csv
import os
import h5py
import numpy as np
from typing import Dict, Any, List, Tuple
from tqdm import tqdm


def load_metadata(metadata_file: str) -> Dict[str, Dict[str, str]]:
    """
    Load metadata CSV and return a dictionary mapping node IDs to metadata.

    Returns:
        Dict mapping node_id -> {title, abstract, doi}
    """
    print(f"Loading metadata from: {metadata_file}")
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

    print(f"  Loaded metadata for {len(metadata)} nodes")
    return metadata


def compute_leaf_embeddings(metadata: Dict[str, Dict[str, str]],
                           model_name: str = "ncbi/MedCPT-Query-Encoder") -> Tuple[Dict[str, np.ndarray], int]:
    """
    Compute MedCPT embeddings for all leaf nodes.

    Args:
        metadata: Dictionary mapping node_id -> {title, abstract}
        model_name: HuggingFace model name for MedCPT

    Returns:
        (embeddings_dict, embedding_dim) where embeddings_dict maps node_id -> embedding vector
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except ImportError:
        print("Error: transformers and torch are required.")
        print("Install with: pip install transformers torch")
        exit(1)

    print(f"\nLoading MedCPT model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"  Using device: {device}")

    embeddings = {}
    embedding_dim = None

    print(f"\nComputing embeddings for {len(metadata)} nodes...")

    with torch.no_grad():
        for node_id, meta in tqdm(metadata.items(), desc="Encoding"):
            # Concatenate title and abstract
            text = f"{meta['title']} {meta['abstract']}".strip()

            if not text:
                print(f"Warning: Empty text for node {node_id}, skipping")
                continue

            # Tokenize and encode
            inputs = tokenizer(text, return_tensors="pt", padding=True,
                             truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get embeddings (use CLS token or mean pooling)
            outputs = model(**inputs)
            # Using CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

            embeddings[node_id] = embedding

            if embedding_dim is None:
                embedding_dim = len(embedding)

    print(f"  Computed {len(embeddings)} embeddings of dimension {embedding_dim}")
    return embeddings, embedding_dim


def collect_all_nodes(node: Dict[str, Any], all_nodes: List[Dict[str, Any]]) -> None:
    """
    Recursively collect all nodes in the tree (both leaves and internal).
    """
    all_nodes.append(node)
    if node.get("type") == "cluster":
        for child in node.get("children", []):
            collect_all_nodes(child, all_nodes)


def propagate_embeddings_upward(root: Dict[str, Any],
                                leaf_embeddings: Dict[str, np.ndarray],
                                embedding_dim: int) -> Dict[int, np.ndarray]:
    """
    Propagate embeddings from leaves up to internal nodes via averaging.

    Args:
        root: Root node of the hierarchy
        leaf_embeddings: Dict mapping node_name (str) -> embedding vector
        embedding_dim: Dimensionality of embeddings

    Returns:
        Dict mapping node_id (int) -> embedding vector for ALL nodes
    """
    all_embeddings = {}

    def propagate(node: Dict[str, Any]) -> np.ndarray:
        """
        Recursively compute embeddings for a node.
        For leaves: use the precomputed embedding
        For internal: average of children embeddings
        """
        node_id = node['id']

        if node.get('type') == 'leaf':
            # Leaf node: use the embedding from metadata
            node_name = node.get('name')
            if node_name in leaf_embeddings:
                embedding = leaf_embeddings[node_name]
                all_embeddings[node_id] = embedding
                return embedding
            else:
                print(f"Warning: No embedding found for leaf node {node_name} (id={node_id})")
                # Return zero embedding
                embedding = np.zeros(embedding_dim)
                all_embeddings[node_id] = embedding
                return embedding
        else:
            # Internal node: average children embeddings
            children = node.get('children', [])
            if not children:
                print(f"Warning: Internal node {node_id} has no children")
                embedding = np.zeros(embedding_dim)
                all_embeddings[node_id] = embedding
                return embedding

            child_embeddings = [propagate(child) for child in children]
            # Average the embeddings
            embedding = np.mean(child_embeddings, axis=0)
            all_embeddings[node_id] = embedding
            return embedding

    print("\nPropagating embeddings up the tree...")
    propagate(root)
    print(f"  Computed embeddings for {len(all_embeddings)} total nodes")

    return all_embeddings


def save_embeddings_hdf5(embeddings: Dict[int, np.ndarray],
                        output_file: str,
                        hierarchy: Dict[str, Any]) -> None:
    """
    Save embeddings to HDF5 format.

    Structure:
        /embeddings: (num_nodes, embedding_dim)
        /node_ids: (num_nodes,) - node IDs corresponding to each row
        /node_metadata: JSON string with hierarchy info
    """
    print(f"\nSaving embeddings to HDF5: {output_file}")

    # Sort by node ID for consistent ordering
    node_ids = sorted(embeddings.keys())
    embedding_matrix = np.array([embeddings[nid] for nid in node_ids])

    with h5py.File(output_file, 'w') as f:
        # Save embeddings
        f.create_dataset('embeddings', data=embedding_matrix, compression='gzip')

        # Save node IDs
        f.create_dataset('node_ids', data=np.array(node_ids), compression='gzip')

        # Save metadata
        f.attrs['num_nodes'] = len(node_ids)
        f.attrs['embedding_dim'] = embedding_matrix.shape[1]
        f.attrs['algorithm'] = hierarchy.get('algorithm', 'Unknown')
        f.attrs['num_original_nodes'] = hierarchy.get('num_nodes', 0)
        f.attrs['num_original_edges'] = hierarchy.get('num_edges', 0)

    print(f"  Saved {len(node_ids)} embeddings of dimension {embedding_matrix.shape[1]}")
    print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")


def save_embeddings_json(embeddings: Dict[int, np.ndarray],
                        output_file: str,
                        hierarchy: Dict[str, Any]) -> None:
    """
    Save embeddings in JSON format (with base64 encoding for compactness).
    Warning: This creates a large file!
    """
    import base64

    print(f"\nSaving embeddings to JSON: {output_file}")

    # Encode embeddings as base64 strings
    encoded_embeddings = {}
    for node_id, embedding in embeddings.items():
        # Convert to float32 for smaller size
        embedding_bytes = embedding.astype(np.float32).tobytes()
        encoded = base64.b64encode(embedding_bytes).decode('utf-8')
        encoded_embeddings[str(node_id)] = encoded

    output_data = {
        'algorithm': hierarchy.get('algorithm', 'Unknown'),
        'num_nodes': hierarchy.get('num_nodes', 0),
        'num_edges': hierarchy.get('num_edges', 0),
        'embedding_dim': len(next(iter(embeddings.values()))),
        'num_embedded_nodes': len(embeddings),
        'embeddings': encoded_embeddings
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Saved {len(embeddings)} embeddings")
    print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    # Input files
    metadata_file = os.path.join(data_dir, 'metadata/oc_mini_node_metadata.csv')
    hierarchy_file = os.path.join(data_dir, 'clustering/oc_mini_paris.json')

    # Output files
    output_h5 = os.path.join(data_dir, 'clustering/oc_mini_paris_embeddings.h5')
    output_json = os.path.join(data_dir, 'clustering/oc_mini_paris_embeddings.json')

    print("=" * 60)
    print("Tree Embedding Computation Pipeline")
    print("=" * 60)
    print()

    # Step 1: Load metadata
    metadata = load_metadata(metadata_file)

    # Step 2: Load hierarchy
    print(f"\nLoading hierarchy from: {hierarchy_file}")
    with open(hierarchy_file, 'r') as f:
        hierarchy_data = json.load(f)
    hierarchy = hierarchy_data['hierarchy']
    print(f"  Algorithm: {hierarchy_data.get('algorithm')}")
    print(f"  Original nodes: {hierarchy_data.get('num_nodes')}")
    print(f"  Original edges: {hierarchy_data.get('num_edges')}")

    # Step 3: Compute leaf embeddings with MedCPT
    leaf_embeddings, embedding_dim = compute_leaf_embeddings(metadata)

    # Step 4: Propagate embeddings up the tree
    all_embeddings = propagate_embeddings_upward(hierarchy, leaf_embeddings, embedding_dim)

    # Step 5: Save embeddings
    save_embeddings_hdf5(all_embeddings, output_h5, hierarchy_data)

    # Optionally save to JSON (can be slow for large datasets)
    save_json = input("\nSave embeddings to JSON format as well? (y/n): ").lower().strip() == 'y'
    if save_json:
        save_embeddings_json(all_embeddings, output_json, hierarchy_data)

    print("\n" + "=" * 60)
    print("Completed successfully!")
    print("=" * 60)
    print(f"\nEmbeddings saved to: {output_h5}")
    print("\nTo load embeddings in Python:")
    print("  import h5py")
    print(f"  with h5py.File('{output_h5}', 'r') as f:")
    print("      embeddings = f['embeddings'][:]")
    print("      node_ids = f['node_ids'][:]")


if __name__ == "__main__":
    main()
