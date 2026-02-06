import numpy as np
from graph_layout import update_node_position, update_all_positions_numpy


def test_update_node_position_two_nodes_strong_link():
    """Test that update_node_position works correctly with two nodes."""
    nodes = {'A': 0.3, 'B': 0.2}
    edges = {('A', 'B'): 10000.0}
    positions = {'A': (0.5, 0.5), 'B': (0.5, -0.5)}
    
    # Update node B
    new_pos_b = update_node_position(
        'B', nodes, positions, edges,
        node_repulsion=0.3,
        link_attraction=0.2,
        step_size=0.3,
        boundary_repulsion=0.1
    )
    
    # Verify B' is between B and A
    pos_b_old = np.array(positions['B'])
    pos_b_new = np.array(new_pos_b)
    pos_a = np.array(positions['A'])
    
    assert np.linalg.norm(pos_b_new - pos_b_old) > 0, "B should have moved"
    assert np.linalg.norm(pos_b_new - pos_a) < np.linalg.norm(pos_b_old - pos_a), "B' should be closer to A than B was"
    assert np.linalg.norm(pos_b_new - pos_b_old) < np.linalg.norm(pos_b_old - pos_a), "B' should not overshoot past A"


def test_update_node_positions_parallel_two_nodes_strong_link():
    """Test that update_all_positions_parallel works correctly with two nodes."""
    nodes = {'A': 0.3, 'B': 0.2}
    edges = {('A', 'B'): 10000.0}
    positions = {'A': (0.5, 0.5), 'B': (0.5, -0.5)}
    
    # Store old positions
    pos_a_old = np.array(positions['A'])
    pos_b_old = np.array(positions['B'])
    
    # Update all positions (parallel version updates all nodes simultaneously)
    new_positions, movements = update_all_positions_numpy(
        positions, nodes, edges,
        node_repulsion=0.3,
        link_attraction=0.2,
        step_size=0.3,
        boundary_repulsion=0.1
    )
    
    # Extract new positions
    pos_a_new = np.array(new_positions['A'])
    pos_b_new = np.array(new_positions['B'])
    
    # Verify both nodes moved
    assert np.linalg.norm(pos_b_new - pos_b_old) > 0, "B should have moved"
    assert np.linalg.norm(pos_a_new - pos_a_old) > 0, "A should have moved"
    
    # Verify B moved closer to A (strong link pulls them together)
    old_distance = np.linalg.norm(pos_b_old - pos_a_old)
    new_distance = np.linalg.norm(pos_b_new - pos_a_new)
    assert new_distance < old_distance, "B should be closer to A after update"
    
    # Verify movements array has correct shape and values
    assert len(movements) == 2, "movements array should have 2 elements"
    assert movements[0] > 0, "A should have non-zero movement"
    assert movements[1] > 0, "B should have non-zero movement"
