#!/usr/bin/env python3
"""
Script to reroot a Paris hierarchical clustering dendrogram for better balance.
Rerooting rotates the tree to use a different node as root while preserving all nodes.
"""
import json
import os
import copy
from typing import Dict, Any, Tuple, Optional, List


def build_parent_map(node: Dict[str, Any], parent: Optional[Dict[str, Any]] = None,
                     parent_map: Optional[Dict[int, Dict[str, Any]]] = None) -> Dict[int, Dict[str, Any]]:
    """
    Build a map from node ID to parent node.

    Returns:
        Dict mapping node_id -> parent_node
    """
    if parent_map is None:
        parent_map = {}

    node_id = node['id']
    parent_map[node_id] = parent

    if node.get("type") == "cluster":
        for child in node.get("children", []):
            build_parent_map(child, node, parent_map)

    return parent_map


def build_node_map(node: Dict[str, Any], node_map: Optional[Dict[int, Dict[str, Any]]] = None) -> Dict[int, Dict[str, Any]]:
    """
    Build a map from node ID to node object.

    Returns:
        Dict mapping node_id -> node
    """
    if node_map is None:
        node_map = {}

    node_id = node['id']
    node_map[node_id] = node

    if node.get("type") == "cluster":
        for child in node.get("children", []):
            build_node_map(child, node_map)

    return node_map


def get_path_to_root(node_id: int, parent_map: Dict[int, Optional[Dict[str, Any]]]) -> List[int]:
    """
    Get the path from a node to the root.

    Returns:
        List of node IDs from the given node to the root
    """
    path = [node_id]
    current = parent_map.get(node_id)

    while current is not None:
        path.append(current['id'])
        current = parent_map.get(current['id'])

    return path


def reroot_tree(original_root: Dict[str, Any], new_root_id: int) -> Dict[str, Any]:
    """
    Reroot the tree at a different node while preserving all nodes.

    This works by:
    1. Finding the path from new_root to old_root
    2. Reversing parent-child relationships along this path
    3. Reconstructing the tree with new_root as root
    """
    # Build maps
    parent_map = build_parent_map(original_root)
    node_map = build_node_map(original_root)

    # Get the new root node
    if new_root_id not in node_map:
        raise ValueError(f"Node {new_root_id} not found in tree")

    # Deep copy all nodes to avoid modifying the original
    new_node_map = {}
    for node_id, node in node_map.items():
        new_node_map[node_id] = copy.deepcopy(node)

    # Get path from new root to old root
    path = get_path_to_root(new_root_id, parent_map)

    # Reverse edges along the path
    # For each edge in the path (except the new root), we need to:
    # - Remove the child from the parent's children list
    # - Add the parent to the child's children list

    for i in range(len(path) - 1):
        child_id = path[i]
        parent_id = path[i + 1]

        child_node = new_node_map[child_id]
        parent_node = new_node_map[parent_id]

        # Remove child from parent's children
        if 'children' in parent_node:
            parent_node['children'] = [c for c in parent_node['children'] if c['id'] != child_id]

        # Add parent as child of child (reversing the edge)
        if 'children' not in child_node:
            child_node['children'] = []
        child_node['children'].append(parent_node)

        # Update type if needed (a leaf might become internal after reversing)
        if child_node.get('type') == 'leaf' and len(child_node['children']) > 0:
            child_node['type'] = 'cluster'

        # Update type of parent if it has no more children
        if len(parent_node.get('children', [])) == 0 and parent_node.get('type') == 'cluster':
            # This shouldn't happen in a proper dendrogram, but handle it
            parent_node['type'] = 'leaf'

    return new_node_map[new_root_id]


def compute_tree_height(node: Dict[str, Any]) -> int:
    """
    Compute the height of a tree (maximum depth from root to any leaf).
    """
    if node.get("type") == "leaf":
        return 0

    children = node.get("children", [])
    if not children:
        return 0

    return max(compute_tree_height(child) for child in children) + 1


def count_nodes(node: Dict[str, Any]) -> int:
    """
    Count total number of nodes in the tree.
    """
    if node.get("type") == "leaf":
        return 1

    children = node.get("children", [])
    return 1 + sum(count_nodes(child) for child in children)


def precompute_heights(node: Dict[str, Any], height_map: Optional[Dict[int, int]] = None) -> int:
    """
    Precompute and cache heights for all nodes in the tree.

    Returns:
        The height of the given node, and fills height_map with all heights
    """
    if height_map is None:
        height_map = {}

    node_id = node['id']

    if node.get('type') == 'leaf':
        height_map[node_id] = 0
        return 0

    children = node.get('children', [])
    if not children:
        height_map[node_id] = 0
        return 0

    child_heights = [precompute_heights(child, height_map) for child in children]
    height = max(child_heights) + 1
    height_map[node_id] = height

    return height


def compute_height_if_rerooted(node_id: int, parent_map: Dict[int, Optional[Dict[str, Any]]],
                               node_map: Dict[int, Dict[str, Any]],
                               height_map: Dict[int, int]) -> int:
    """
    Efficiently compute the height if we rerooted at node_id without actually rerooting.

    Uses precomputed heights for efficiency.
    """
    # Get the path from this node to the current root
    path = get_path_to_root(node_id, parent_map)

    # Track maximum height
    max_height = 0

    # For each node in the path, compute the max height from branches not on the path
    for i, current_id in enumerate(path):
        current_node = node_map[current_id]

        if current_node.get('type') == 'leaf' and i > 0:
            continue

        # Get all children
        children = current_node.get('children', [])

        for child in children:
            child_id = child['id']

            # Skip the child that's in our path (leading back toward new_root)
            if i < len(path) - 1 and child_id == path[i + 1]:
                continue

            # Use precomputed height + distance from new root
            child_height = height_map.get(child_id, 0) + i + 1
            max_height = max(max_height, child_height)

    return max_height


def find_best_root(original_root: Dict[str, Any]) -> Tuple[int, int]:
    """
    Find the best node to use as root for minimal tree height.

    Returns:
        (best_node_id, best_height)
    """
    # Build maps once
    node_map = build_node_map(original_root)
    parent_map = build_parent_map(original_root)

    print(f"Total nodes in tree: {len(node_map)}")

    # Precompute all heights
    print("Precomputing subtree heights...")
    height_map = {}
    precompute_heights(original_root, height_map)
    print(f"  Computed heights for {len(height_map)} nodes")

    print("Evaluating all nodes as potential roots...")

    # Only consider internal (cluster) nodes
    cluster_nodes = [node_id for node_id, node in node_map.items()
                     if node.get("type") == "cluster"]
    print(f"Cluster nodes to evaluate: {len(cluster_nodes)}")

    best_node_id = None
    best_height = float('inf')

    for i, node_id in enumerate(cluster_nodes):
        if i % 1000 == 0 and i > 0:
            print(f"  Evaluated {i}/{len(cluster_nodes)} nodes...")

        # Compute height if rerooted (without actually rerooting)
        try:
            height = compute_height_if_rerooted(node_id, parent_map, node_map, height_map)

            if height < best_height:
                best_height = height
                best_node_id = node_id
                print(f"  New best: node {node_id} with height {height}")
        except Exception as e:
            print(f"  Warning: Failed to evaluate node {node_id}: {e}")
            continue

    return best_node_id, best_height


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
    orig_height = compute_tree_height(original_root)
    orig_count = count_nodes(original_root)

    print("Original tree statistics:")
    print(f"  Root ID: {original_root['id']}")
    print(f"  Height: {orig_height}")
    print(f"  Total nodes: {orig_count}")
    print()

    # Find best root
    print("Finding best root for minimal tree height...")
    print("(This will preserve all nodes while minimizing tree depth)")
    best_node_id, best_height = find_best_root(original_root)
    print()

    if best_node_id is None:
        print("Error: Could not find a suitable root node")
        return

    # Reroot at the best node
    print(f"Rerooting tree at node {best_node_id}...")
    rerooted_tree = reroot_tree(original_root, best_node_id)
    new_count = count_nodes(rerooted_tree)

    print("=" * 60)
    print("Rerooted tree statistics:")
    print("=" * 60)
    print(f"  New root ID: {best_node_id}")
    print(f"  Height: {best_height} (original: {orig_height})")
    print(f"  Total nodes: {new_count} (original: {orig_count})")
    print(f"  Height reduction: {orig_height - best_height} levels ({100 * (orig_height - best_height) / orig_height:.1f}%)")

    if new_count != orig_count:
        print(f"  WARNING: Node count changed! This should not happen.")
    else:
        print(f"  ✓ All nodes preserved")
    print()

    # Save the rerooted tree
    data["hierarchy"] = rerooted_tree

    print(f"Saving rerooted hierarchy to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print()

    print("=" * 60)
    print("Completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
