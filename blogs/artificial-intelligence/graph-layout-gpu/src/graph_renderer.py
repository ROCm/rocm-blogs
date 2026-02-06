"""
Graph rendering module for directed graphs with weighted nodes and edges.
"""

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch


def draw_graph(ax, nodes, positions, edges={}, node_labels=None):
    """
    Draw a directed graph with weighted nodes and edges.
    Positions must be within the unit disc (distance from origin <= 1.0).
    
    Args:
        ax: matplotlib axes
        nodes: dict {node: size} - node sizes (radius in plot coordinates)
        positions: dict {node: (x, y)} - node positions
        edges: dict {(from, to): width} - edge widths
        node_labels: dict {node: label_text}, optional (defaults to node names)
    """
    if node_labels is None:
        node_labels = {node: str(node) for node in nodes.keys()}

    # Check for bidirectional edges
    edge_list = list(edges.keys())
    edge_set = set(edge_list)
    bidirectional = {(u, v) for u, v in edge_list if (v, u) in edge_set}

    # Draw nodes
    for node, (x, y) in positions.items():
        radius = nodes[node]
        
        circle = Circle((x, y), radius, color='lightblue', ec='#999999', linewidth=1, zorder=2)
        ax.add_patch(circle)
        
        # Add label
        label = node_labels.get(node, str(node))
        ax.text(x, y, label, ha='center', va='center', fontsize=10, zorder=2)

    # Draw edges
    for (u, v), width in edges.items():
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        
        # Curve bidirectional edges (both arrows needed, curved to not overlap)
        connectionstyle = "arc3,rad=0.2" if (v, u) in bidirectional else "arc3,rad=0.0"
        
        # Calculate actual start/end points at node edges (not centers)
        # Direction vector from u to v
        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx**2 + dy**2)
        
        # Skip arrow if nodes are at same position
        if distance == 0:
            continue
        
        # Unit vector in direction of arrow
        ux = dx / distance
        uy = dy / distance
        
        # Node radii
        r1 = nodes[u]
        r2 = nodes[v]
        
        # Arrow should start at edge of u and end at edge of v
        start_x = x1 + r1 * ux
        start_y = y1 + r1 * uy
        end_x = x2 - r2 * ux
        end_y = y2 - r2 * uy

        if distance < r1 + r2:
            start_x, start_y, end_x, end_y = end_x, end_y, start_x, start_y
        
        arrow = FancyArrowPatch(
            (start_x, start_y), (end_x, end_y),
            arrowstyle='->', mutation_scale=20,
            linewidth=max(width, 2), color='#aaaaaa',  # Minimum width 2 for visibility
            connectionstyle=connectionstyle,
            joinstyle='miter',  # Sharp corners instead of rounded
            zorder=4 if distance < r1 + r2 else 1
        )
        ax.add_patch(arrow)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
