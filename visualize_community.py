#!/usr/bin/env python3
"""
Visualize a k-core community as a force-directed network graph.
Nodes are colored by cosine similarity to the input query embedding.
Hovering over nodes displays title and abstract from metadata.
"""
import json
import csv
import os
import h5py
import numpy as np
import argparse
from typing import Dict, List, Any, Tuple
import plotly.graph_objects as go
import networkx as nx


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


def load_edgelist(edgelist_file: str) -> List[Tuple[int, int]]:
    """
    Load edge list from TSV file.

    Returns:
        List of (source, target) tuples
    """
    edges = []
    with open(edgelist_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                edges.append((int(parts[0]), int(parts[1])))
    return edges


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


def visualize_community(
    community: Dict[str, Any],
    query_embedding: np.ndarray,
    all_edges: List[Tuple[int, int]],
    embeddings: Dict[int, np.ndarray],
    metadata: Dict[str, Dict[str, str]],
    query_text: str = "",
    output_file: str = "community_network.html",
    verbose: bool = False
) -> go.Figure:
    """
    Create a force-directed network visualization of the community.

    Args:
        community: Dict with 'nodes' (List[int]), 'k' (int), 'size' (int)
        query_embedding: The query embedding vector for computing similarities
        all_edges: List of all edges in the network
        embeddings: Dict mapping node_id -> embedding vector
        metadata: Dict mapping node_id -> metadata dict
        query_text: The original query text to display in the visualization
        output_file: Path to save the HTML visualization

    Returns:
        Plotly figure object
    """
    community_nodes = set(community['nodes'])
    k_value = community['k']

    # Filter edges to only those within the community
    community_edges = [
        (u, v) for u, v in all_edges
        if u in community_nodes and v in community_nodes
    ]

    # Create NetworkX graph for layout computation
    G = nx.Graph()
    G.add_nodes_from(community_nodes)
    G.add_edges_from(community_edges)

    # Compute spring layout
    pos = nx.spring_layout(G, k=1/np.sqrt(len(community_nodes)), iterations=50)

    # Compute cosine similarities for all nodes in community
    similarities = {}
    scaled_similarities = {}
    for node_id in community_nodes:
        if node_id in embeddings:
            node_embedding = embeddings[node_id]
            sim = cosine_similarity(query_embedding, node_embedding)
            similarities[node_id] = sim
        else:
            similarities[node_id] = 0.0

    # Scale similarities for better visualization
    sim_values = list(similarities.values())
    min_sim = min(sim_values)
    max_sim = max(sim_values)

    if verbose:
        print(f"\n  Similarity range: [{min_sim:.4f}, {max_sim:.4f}]")
        print(f"  Mean similarity: {np.mean(sim_values):.4f}")
        print(f"  Median similarity: {np.median(sim_values):.4f}")

    # Use rank-based coloring for better distribution across the color spectrum
    # Sort nodes by similarity and assign colors based on rank
    sorted_nodes = sorted(similarities.items(), key=lambda x: x[1])

    for rank, (node_id, sim) in enumerate(sorted_nodes):
        # Normalize rank to 0-1 range
        scaled_similarities[node_id] = rank / max(1, len(sorted_nodes) - 1)

    if verbose:
        print(f"  Using rank-based coloring")
        print(f"  Lowest similarity node: {sorted_nodes[0][1]:.4f}")
        print(f"  Median similarity node: {sorted_nodes[len(sorted_nodes)//2][1]:.4f}")
        print(f"  Highest similarity node: {sorted_nodes[-1][1]:.4f}")

    # Prepare edge trace
    edge_x = []
    edge_y = []
    for u, v in community_edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Get central node ID if available
    central_node_id = None
    if 'central_node' in community and 'name' in community['central_node']:
        try:
            central_node_id = int(community['central_node']['name'])
        except (ValueError, TypeError):
            pass

    # Prepare node traces - separate regular nodes from central node
    regular_node_x = []
    regular_node_y = []
    regular_node_text = []
    regular_node_color = []

    central_node_x = []
    central_node_y = []
    central_node_text = []
    central_node_color = []

    for node_id in community_nodes:
        x, y = pos[node_id]

        # Get metadata
        node_id_str = str(node_id)
        meta = metadata.get(node_id_str, {
            'title': f'Node {node_id}',
            'abstract': 'No metadata available',
            'doi': ''
        })

        # Create hover text with proper line wrapping
        title = meta['title']
        abstract = meta['abstract']
        doi = meta['doi']
        sim = similarities[node_id]

        # Wrap text to reasonable width (about 60 characters per line)
        def wrap_text(text, width=60):
            words = text.split()
            lines = []
            current_line = []
            current_length = 0

            for word in words:
                if current_length + len(word) + 1 <= width:
                    current_line.append(word)
                    current_length += len(word) + 1
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)

            if current_line:
                lines.append(' '.join(current_line))

            return '<br>'.join(lines)

        title_wrapped = wrap_text(title, 50)
        abstract_wrapped = wrap_text(abstract, 50)

        # Get rank from the scaled similarities (which is based on sorted position)
        rank = int(scaled_similarities[node_id] * (len(community_nodes) - 1)) + 1
        total = len(community_nodes)

        # Mark central node in hover text
        node_label = "QUERY PAPER" if node_id == central_node_id else ""
        hover_prefix = f"<b>{node_label}</b><br>" if node_label else ""

        hover_text = (
            f"{hover_prefix}"
            f"<b>Node ID:</b> {node_id}<br>"
            f"<b>Cosine Similarity:</b> {sim:.4f}<br>"
            f"<b>Rank:</b> {rank}/{total}<br>"
            f"<b>Title:</b><br>{title_wrapped}<br>"
            f"<b>Abstract:</b><br>{abstract_wrapped}<br>"
            f"<b>DOI:</b> {doi}"
        )

        # Separate central node from regular nodes
        if node_id == central_node_id:
            central_node_x.append(x)
            central_node_y.append(y)
            central_node_text.append(hover_text)
            central_node_color.append(scaled_similarities[node_id])
        else:
            regular_node_x.append(x)
            regular_node_y.append(y)
            regular_node_text.append(hover_text)
            regular_node_color.append(scaled_similarities[node_id])

    # Create regular nodes trace
    node_trace = go.Scatter(
        x=regular_node_x,
        y=regular_node_y,
        mode='markers',
        hoverinfo='text',
        text=regular_node_text,
        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="monospace",
            align="left",
            namelength=0
        ),
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=regular_node_color,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Similarity Rank',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )

    # Create central node trace (star shape)
    central_node_trace = go.Scatter(
        x=central_node_x,
        y=central_node_y,
        mode='markers',
        hoverinfo='text',
        text=central_node_text,
        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="monospace",
            align="left",
            namelength=0
        ),
        marker=dict(
            symbol='star',
            size=20,
            color=central_node_color,
            colorscale='Viridis',
            line=dict(color='red', width=3),
            showscale=False
        )
    )

    # Create annotations list
    annotations = []

    # Prepare query display (truncate if too long)
    query_display = query_text[:80] + '...' if len(query_text) > 80 else query_text

    # Add bottom annotation about colors
    annotations.append(
        dict(
            text=f"Node colors: similarity range [{min_sim:.3f}, {max_sim:.3f}] (rank-based)",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.04,
            xanchor='center',
            yanchor='top',
            font=dict(size=10, color='#666666', family='Arial')
        )
    )

    # Create figure - add central node trace if it exists
    traces = [edge_trace, node_trace]
    if central_node_x:
        traces.append(central_node_trace)

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(
                text=f'<i>Query: {query_display}</i><br><b>K-Core Community</b><br><sub>k = {k_value} | {len(community["nodes"])} nodes</sub>',
                x=0.5,
                xanchor='center',
                font=dict(size=16, family='Arial', color='#333333')
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=30, l=10, r=10, t=120),
            annotations=annotations,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#FAFAFA',
            paper_bgcolor='white'
        )
    )

    # Save to HTML
    fig.write_html(output_file)
    print(f"Visualization saved to {output_file}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize k-core community with force-directed layout'
    )
    parser.add_argument('query', type=str,
                       help='Query string to embed with MedCPT')
    parser.add_argument('--community-file', type=str, default='fetched_community.json',
                       help='JSON file containing community data (default: fetched_community.json)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--output', type=str, default='community_network.html',
                       help='Output HTML file (default: community_network.html)')

    args = parser.parse_args()

    # File paths
    data_dir = args.data_dir
    metadata_file = os.path.join(data_dir, 'metadata/oc_mini_node_metadata.csv')
    embeddings_file = os.path.join(data_dir, 'clustering/oc_mini_paris_embeddings.h5')
    network_file = os.path.join(data_dir, 'network/oc_mini_edgelist.tsv')

    print("Loading data...")

    # Load community data
    with open(args.community_file, 'r') as f:
        community = json.load(f)
    print(f"  Loaded community with {len(community['nodes'])} nodes (k={community['k']})")

    # Embed query
    print(f"\nEmbedding query: '{args.query}'")
    query_embedding = embed_query(args.query)
    print(f"  Query embedding shape: {query_embedding.shape}")

    # Load metadata
    metadata = load_metadata(metadata_file)
    print(f"  Loaded metadata for {len(metadata)} nodes")

    # Load embeddings
    embeddings = load_embeddings(embeddings_file)
    print(f"  Loaded {len(embeddings)} embeddings")

    # Load network edges
    edges = load_edgelist(network_file)
    print(f"  Loaded {len(edges)} edges")

    # Create visualization
    print("\nCreating visualization...")
    fig = visualize_community(
        community=community,
        query_embedding=query_embedding,
        all_edges=edges,
        embeddings=embeddings,
        metadata=metadata,
        query_text=args.query,
        output_file=args.output,
        verbose=True
    )

    print(f"\nDone! Open {args.output} in a browser to view the network.")


if __name__ == "__main__":
    main()
