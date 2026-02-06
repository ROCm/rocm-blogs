import numpy as np
import torch


def check_position_invariants(positions):
    """
    Check that all positions are valid (within unit disc, no duplicates).
    Raises ValueError with details if any problems found.
    """
    problem_nodes = []
    for node, pos_tuple in positions.items():
        pos_arr = np.array(pos_tuple)
        if not np.isfinite(pos_arr).all():
            problem_nodes.append(f"{node} has non-finite position {pos_tuple}")
            continue
        if (dist := np.linalg.norm(pos_arr)) > 1.0:
            problem_nodes.append(f"{node} (distance {dist:.6f} > 1.0)")
    
    # Check for identical positions
    pos_list = list(positions.items())
    for i, (node1, pos1) in enumerate(pos_list):
        for node2, pos2 in pos_list[i+1:]:
            if np.allclose(pos1, pos2):
                problem_nodes.append(f"{node1} and {node2} have identical positions {pos1}")
    
    if problem_nodes:
        raise ValueError(f"Invalid positions: {', '.join(problem_nodes)}")


def update_node_position(active_node_id, nodes, positions, edges={},
                         node_repulsion=1, link_attraction=0.1, step_size=0.1, boundary_repulsion=0.1):
    """
    Update one node position using repulsion, attraction, and boundary forces.
    
    nodes: dict {node: size} - node sizes (radius in plot coordinates)
    positions: dict {node: (x, y)} - snapshot of all positions
    edges: dict {(from, to): weight} - edge weights
    active_node_id: str - id (dict key) of node to update
    Returns: (new_x, new_y) tuple
    """
    check_position_invariants(positions)
    
    pos = np.array(positions[active_node_id])
    r1 = nodes[active_node_id]
    force = np.array([0.0, 0.0])
    
    # Repulsion from other nodes (scaled by node sizes)
    for other_node, other_pos in positions.items():
        if other_node == active_node_id:
            continue
        r2 = nodes[other_node]
        delta = pos - np.array(other_pos)
        distance = np.linalg.norm(delta)
        size_factor = (r1 + r2) ** 2
        repulsion = (delta / distance) * (node_repulsion * size_factor / (distance ** 2))
        if not np.isfinite(repulsion).all():
            raise ValueError(f"Non-finite repulsion: repulsion={repulsion}, distance={distance:.10f}, delta={delta}, pos={pos}, other_pos={other_pos}")
        force += repulsion
    
    # Attraction along edges (proportional to edge weight and distance)
    for (u, v), weight in edges.items():
        if u == active_node_id:
            other_node = v
        elif v == active_node_id:
            other_node = u
        else:
            continue
        other_pos = np.array(positions[other_node])
        delta = other_pos - pos
        attraction = delta * link_attraction * weight ** 0.3
        # Cap attraction so proposed step toward target node never exceeds one third the distance to it
        movement_from_link = np.linalg.norm(attraction * step_size)
        if movement_from_link > np.linalg.norm(delta) / 3:
            attraction = attraction * (np.linalg.norm(delta) / (3 * movement_from_link))
        force += attraction
    
    # Boundary repulsion: force toward center, grows rapidly near boundary
    dist_from_center = np.linalg.norm(pos)
    dist_to_edge = 1.0 - dist_from_center
    if dist_to_edge <= 0:
        raise ValueError(f"Node {active_node_id} at distance {dist_from_center:.6f} >= 1.0, dist_to_edge={dist_to_edge:.6f}")
    boundary_force = boundary_repulsion * r1 * (-pos) / (dist_to_edge ** 2)
    force += boundary_force
    
    # Check for NaN/inf in force before updating position
    if not np.isfinite(force).all():
        raise ValueError(f"Non-finite force for node {active_node_id}: force={force}, pos={pos}, dist_from_center={dist_from_center:.6f}")
    
    # Update position
    velocity = force * step_size
    movement = np.linalg.norm(velocity)
    
    # Cap total movement to not exceed minimum distance to any other node
    # This prevents nodes from overshooting when repulsion/attraction combine
    min_distance_to_other = min(
        np.linalg.norm(pos - np.array(other_pos))
        for other_node, other_pos in positions.items()
        if other_node != active_node_id
    )
    if movement > min_distance_to_other:
        velocity = velocity * (min_distance_to_other / movement)
        movement = min_distance_to_other
    
    new_pos = pos + velocity
    
    # Limit outward movement: cap distance at (1 + current_dist) / 2
    max_dist = (1.0 + dist_from_center) / 2.0
    new_dist = np.linalg.norm(new_pos)
    if new_dist > max_dist:
        new_pos = new_pos * (max_dist / new_dist)
    
    if not np.linalg.norm(new_pos) <= 1.0 or not np.isfinite(new_pos).all():
        raise ValueError(f"New position for node {active_node_id} is invalid: {new_pos}")

    return tuple(new_pos)


def _prepare_torch_inputs(positions, nodes, edges, device=None):
    """Convert dict inputs to torch tensors. Returns tensors and mapping dicts."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    node_list = list(nodes.keys())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    positions_tensor = torch.tensor([[positions[node][0], positions[node][1]] for node in node_list],
                                    dtype=torch.float32, device=device)
    sizes_tensor = torch.tensor([nodes[node] for node in node_list], dtype=torch.float32, device=device)
    
    edge_indices_list = []
    edge_weights_list = []
    for (u, v) in edges.keys():
        if u in node_to_idx and v in node_to_idx:
            edge_indices_list.append([node_to_idx[u], node_to_idx[v]])
            edge_weights_list.append(edges[(u, v)])
    
    if edge_indices_list:
        edge_indices = torch.tensor(edge_indices_list, dtype=torch.long, device=device)
        edge_weights = torch.tensor(edge_weights_list, dtype=torch.float32, device=device)
    else:
        edge_indices = torch.zeros((0, 2), dtype=torch.long, device=device)
        edge_weights = torch.zeros(0, dtype=torch.float32, device=device)
    
    return positions_tensor, sizes_tensor, edge_indices, edge_weights, idx_to_node


def update_all_positions_torch(positions, sizes, edge_indices, edge_weights,
                               node_repulsion=1, link_attraction=0.1, step_size=0.1, boundary_repulsion=0.1,
                               device=None):
    """
    Update all node positions simultaneously using parallel matrix operations on GPU/CPU.
    GPU-optimized version using PyTorch tensors.

    Args:
        positions: torch.Tensor (N, 2) - node positions (x, y)
        sizes: torch.Tensor (N,) - node sizes (radius in plot coordinates)
        edge_indices: torch.Tensor (E, 2) - edge indices as (u_idx, v_idx) pairs, dtype=long
        edge_weights: torch.Tensor (E,) - edge weights
        node_repulsion: float - repulsion force strength
        link_attraction: float - attraction force strength
        step_size: float - step size for position updates
        boundary_repulsion: float - boundary repulsion strength
        device: torch.device or None - if None, uses device of positions tensor

    Returns:
        tuple (new_positions, movements) where:
        - new_positions: torch.Tensor (N, 2) - updated positions
        - movements: torch.Tensor (N,) - movement magnitude for each node
    """
    if device is None:
        device = positions.device

    # Ensure all tensors are on the same device
    positions = positions.to(device)
    sizes = sizes.to(device)
    edge_indices = edge_indices.to(device)
    edge_weights = edge_weights.to(device)

    N = positions.shape[0]
    E = edge_indices.shape[0]

    # Snapshot positions for this iteration (bulk synchronous)
    positions_snapshot = positions.clone()

    # === REPULSION: All pairs simultaneously ===
    # Broadcasting creates all N² difference vectors: (N, 1, 2) - (1, N, 2) = (N, N, 2)
    differences = positions_snapshot[:, None, :] - positions_snapshot[None, :, :]

    # Vectorized norm computation on N² vectors: (N, N)
    distances = torch.norm(differences, dim=2)
    # Avoid division by zero for self-interaction
    distances = distances.clone()
    distances.fill_diagonal_(1.0)

    # Size factors: (N, N) where size_factors[i,j] = (sizes[i] + sizes[j])^2
    size_factors = (sizes[:, None] + sizes[None, :]) ** 2

    # Repulsion magnitude: (N, N)
    repulsion_magnitudes = node_repulsion * size_factors / (distances ** 2)
    repulsion_magnitudes.fill_diagonal_(0.0)  # Zero out self-repulsion

    # Repulsion forces: (N, N, 2)
    repulsion_forces = (differences / distances[:, :, None]) * repulsion_magnitudes[:, :, None]

    # Sum across all other nodes: (N, 2)
    net_repulsion = repulsion_forces.sum(dim=1)

    # === ATTRACTION: Along edges ===
    net_attraction = torch.zeros(N, 2, device=device, dtype=positions.dtype)
    if E > 0:
        u_indices = edge_indices[:, 0]
        v_indices = edge_indices[:, 1]

        # For each edge u->v: u is pulled toward v
        # Delta vectors: v - u for each edge: (E, 2)
        edge_deltas_uv = positions_snapshot[v_indices] - positions_snapshot[u_indices]
        edge_distances_uv = torch.norm(edge_deltas_uv, dim=1, keepdim=True)  # (E, 1)

        # Attraction magnitude: delta * link_attraction * weight^0.3
        attractions_uv = edge_deltas_uv * link_attraction * (edge_weights ** 0.3)[:, None]  # (E, 2)

        # Cap attraction: movement_from_link should not exceed distance/3
        movement_from_link = torch.norm(attractions_uv * step_size, dim=1, keepdim=True)  # (E, 1)
        max_movement = edge_distances_uv / 3.0  # (E, 1)
        cap_mask = movement_from_link > max_movement  # (E, 1)
        cap_ratios = torch.where(cap_mask, max_movement / movement_from_link, torch.tensor(1.0, device=device))
        attractions_uv = attractions_uv * cap_ratios  # (E, 2)

        # Accumulate forces: u pulled toward v, v pulled toward u
        # Use index_add_ for scatter-add (GPU-friendly atomic operations)
        net_attraction.index_add_(0, u_indices, attractions_uv)
        net_attraction.index_add_(0, v_indices, -attractions_uv)  # Opposite direction (subtract)

    # === BOUNDARY REPULSION ===
    # Distance from center for all nodes: (N,)
    dist_from_center = torch.norm(positions_snapshot, dim=1)
    dist_to_edge = 1.0 - dist_from_center  # (N,)

    # Boundary force: boundary_repulsion * size * (-pos) / (dist_to_edge^2)
    # (-pos) gives direction toward center
    boundary_forces = boundary_repulsion * sizes[:, None] * (-positions_snapshot) / (dist_to_edge ** 2)[:, None]  # (N, 2)

    # === TOTAL FORCE ===
    net_forces = net_repulsion + net_attraction + boundary_forces  # (N, 2)

    # === UPDATE POSITIONS ===
    # Velocity = force * step_size
    velocities = net_forces * step_size  # (N, 2)
    movements = torch.norm(velocities, dim=1)  # (N,)

    # Cap total movement to not exceed minimum distance to any other node
    # Reuse distances matrix from repulsion calculation
    # Set diagonal to inf to exclude self, then take min along each row
    distances_for_min = distances.clone()
    distances_for_min.fill_diagonal_(float('inf'))
    min_distances = torch.min(distances_for_min, dim=1)[0]

    # Cap movements
    cap_mask = movements > min_distances
    cap_ratios = torch.where(cap_mask, min_distances / movements, torch.tensor(1.0, device=device))
    velocities = velocities * cap_ratios[:, None]
    movements = movements * cap_ratios

    # Update positions
    new_positions = positions_snapshot + velocities  # (N, 2)

    # Limit outward movement: cap distance at (1 + current_dist) / 2
    current_dists = dist_from_center  # (N,)
    max_dists = (1.0 + current_dists) / 2.0  # (N,)
    new_dists = torch.norm(new_positions, dim=1)  # (N,)
    dist_cap_mask = new_dists > max_dists
    dist_cap_ratios = torch.where(dist_cap_mask, max_dists / new_dists, torch.tensor(1.0, device=device))
    new_positions = new_positions * dist_cap_ratios[:, None]

    # Ensure all positions are within unit disc
    # GPU-optimized: use unconditional where instead of any() check to avoid CPU sync
    new_dists = torch.norm(new_positions, dim=1)
    outside_mask = new_dists > 1.0
    # Normalize positions outside unit disc (safe: dividing by dist when dist <= 1.0 is idempotent)
    new_positions = torch.where(outside_mask[:, None],
                               new_positions / new_dists[:, None],
                               new_positions)

    return new_positions, movements


def update_all_positions_torch_from_dicts(positions, nodes, edges={},
                                          node_repulsion=1, link_attraction=0.1, step_size=0.1, boundary_repulsion=0.1,
                                          device=None):
    """
    Wrapper that accepts dict inputs (matching numpy interface) and calls tensor-based update.
    
    Args:
        positions: dict {node: (x, y)} - snapshot of all positions
        nodes: dict {node: size} - node sizes (radius in plot coordinates)
        edges: dict {(from, to): weight} - edge weights
        node_repulsion: float - repulsion force strength
        link_attraction: float - attraction force strength
        step_size: float - step size for position updates
        boundary_repulsion: float - boundary repulsion strength
        device: torch.device or None - if None, auto-detects GPU if available, else CPU
    
    Returns:
        tuple (new_positions_dict, movements_array) where:
        - new_positions_dict: dict {node: (x, y)} - updated positions
        - movements_array: numpy array (N,) - movement magnitude for each node
    """
    positions_tensor, sizes_tensor, edge_indices, edge_weights, idx_to_node = _prepare_torch_inputs(
        positions, nodes, edges, device
    )
    
    new_positions, movements = update_all_positions_torch(
        positions_tensor, sizes_tensor, edge_indices, edge_weights,
        node_repulsion, link_attraction, step_size, boundary_repulsion, device
    )
    
    # Convert back to dict and numpy array
    new_positions_cpu = new_positions.cpu().numpy()
    movements_cpu = movements.cpu().numpy()
    new_positions_dict = {idx_to_node[i]: tuple(new_positions_cpu[i]) for i in range(len(idx_to_node))}
    
    return new_positions_dict, movements_cpu
