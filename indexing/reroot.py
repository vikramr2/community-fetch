#!/usr/bin/env python3
"""
Script to reroot a Paris hierarchical clustering dendrogram for better balance.
"""
import json
import os
from typing import Dict, Any, Tuple, Optional


def compute_subtree_stats(node: Dict[str, Any]) -> Tuple[int, int]:
    """
    Compute statistics for a subtree.

    Returns:
        (height, count) where height is the maximum depth and count is the number of nodes
    """
    if node.get("type") == "leaf":
        return 0, 1

    children = node.get("children", [])
    if not children:
        return 0, 1

    child_stats = [compute_subtree_stats(child) for child in children]
    max_height = max(stat[0] for stat in child_stats) + 1
    total_count = sum(stat[1] for stat in child_stats)

    return max_height, total_count


def compute_balance_score(node: Dict[str, Any]) -> float:
    """
    Compute a balance score for a node as a potential root.
    Lower is better. Returns the maximum height difference between children.
    """
    if node.get("type") == "leaf":
        return float('inf')

    children = node.get("children", [])
    if len(children) < 2:
        return float('inf')

    heights = [compute_subtree_stats(child)[0] for child in children]

    # Return the range (max - min) of heights, plus a penalty for depth variance
    height_range = max(heights) - min(heights)
    height_variance = sum((h - sum(heights)/len(heights))**2 for h in heights) / len(heights)

    return height_range + 0.1 * height_variance


def find_all_nodes(node: Dict[str, Any], nodes_list: list) -> None:
    """
    Recursively collect all nodes in the tree.
    """
    nodes_list.append(node)
    if node.get("type") == "cluster":
        for child in node.get("children", []):
            find_all_nodes(child, nodes_list)


def find_best_root(root: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find the best node to use as root for balanced tree.
    """
    all_nodes = []
    find_all_nodes(root, all_nodes)

    print(f"Total nodes in tree: {len(all_nodes)}")
    print("Evaluating balance scores...")

    # Filter to only cluster nodes with at least 2 children
    cluster_nodes = [n for n in all_nodes if n.get("type") == "cluster" and len(n.get("children", [])) >= 2]
    print(f"Cluster nodes with 2+ children: {len(cluster_nodes)}")

    best_node = None
    best_score = float('inf')

    for i, node in enumerate(cluster_nodes):
        score = compute_balance_score(node)

        if i % 1000 == 0:
            print(f"  Evaluated {i}/{len(cluster_nodes)} nodes...")

        if score < best_score:
            best_score = score
            best_node = node
            height, count = compute_subtree_stats(node)
            print(f"  New best: node {node['id']} with score {score:.2f}, height {height}, count {count}")

    return best_node


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    input_file = os.path.join(data_dir, 'clustering/oc_mini_paris.json')
    output_file = os.path.join(data_dir, 'clustering/oc_mini_paris_rebalanced.json')

    print("=" * 60)
    print("Paris Dendrogram Rerooting Tool")
    print("=" * 60)
    print()

    # Load the JSON
    print(f"Loading hierarchy from: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    print()

    original_root = data["hierarchy"]
    orig_height, orig_count = compute_subtree_stats(original_root)
    orig_score = compute_balance_score(original_root)

    print("Original tree statistics:")
    print(f"  Root ID: {original_root['id']}")
    print(f"  Height: {orig_height}")
    print(f"  Total nodes: {orig_count}")
    print(f"  Balance score: {orig_score:.2f}")
    print()

    # Find best root
    print("Finding best root for balanced tree...")
    best_root = find_best_root(original_root)
    print()

    new_height, new_count = compute_subtree_stats(best_root)
    new_score = compute_balance_score(best_root)

    print("=" * 60)
    print("Rebalanced tree statistics:")
    print("=" * 60)
    print(f"  New root ID: {best_root['id']}")
    print(f"  Height: {new_height} (original: {orig_height})")
    print(f"  Total nodes: {new_count}")
    print(f"  Balance score: {new_score:.2f} (original: {orig_score:.2f})")
    print(f"  Improvement: {((orig_score - new_score) / orig_score * 100):.1f}%")
    print()

    # Save the rebalanced tree
    data["hierarchy"] = best_root

    print(f"Saving rebalanced hierarchy to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print()

    print("=" * 60)
    print("Completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
