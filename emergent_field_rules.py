"""
Emergent Field Rules - Pure field-level dynamics in the Collapse Geometry framework

Implements sophisticated field-level behaviors and rules that emerge from configuration space,
with full tensor fields, vectorial circulation, and spatialized gradients.

This module contains ONLY rules for how fields evolve, not the state itself.
All functions accept configuration/polarity space inputs and return computed values.

Enhanced with integrated curvature feedback for proper physical behavior.
"""

import math
import numpy as np
from typing import Dict, List, Set, Optional, Any, Tuple, Union

# Type checking for proper imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from axiom8.collapse_rules.config_space import ConfigurationSpace, ConfigurationPoint
    from axiom8.collapse_rules.polarity_space import PolarityField, EpistemologyRelation

def calculate_vectorial_circulation(
    config_space: 'ConfigurationSpace',
    point_id: str,
    neighbor_ids: List[str],
    curvature_field: Dict[str, Dict[str, Any]] = None  # Added curvature field parameter
) -> np.ndarray:
    """
    Calculate vectorial circulation for a point in configuration space.
    Pure function with no side effects.
    
    Args:
        config_space: The configuration space
        point_id: ID of the central point
        neighbor_ids: IDs of neighboring points
        curvature_field: Optional dict mapping point_id -> curvature properties
        
    Returns:
        3D circulation vector
    """
    # Skip if insufficient neighbors
    if len(neighbor_ids) < 3:
        return np.zeros(3)
    
    # Get center point
    center_point = config_space.points.get(point_id)
    if not center_point:
        return np.zeros(3)
    
    # Calculate circulation for each orthogonal plane
    circulation = np.zeros(3)
    
    # Get relational vectors between points
    relational_vectors = {}
    for neighbor_id in neighbor_ids:
        neighbor_point = config_space.points.get(neighbor_id)
        if not neighbor_point:
            continue
            
        # Use config space to calculate relational vector
        # This represents the purely relational nature of the framework
        rel_vector = calculate_relational_vector(config_space, point_id, neighbor_id)
        
        if np.linalg.norm(rel_vector) > 0.001:
            relational_vectors[neighbor_id] = rel_vector
    
    # Need sufficient neighbors with valid vectors
    if len(relational_vectors) < 3:
        return np.zeros(3)
        
    # For each orthogonal plane (xy, yz, xz), calculate circulation component
    for axis_idx in range(3):  # 0=x, 1=y, 2=z
        # Define orthogonal plane indices
        plane_indices = [(axis_idx+1)%3, (axis_idx+2)%3]
        
        # Sort neighbors by angle in this plane
        plane_neighbors = sorted(
            relational_vectors.items(),
            key=lambda n: math.atan2(n[1][plane_indices[1]], n[1][plane_indices[0]])
        )
        
        # Calculate circulation by line integral around loop
        plane_circulation = 0.0
        
        for i in range(len(plane_neighbors)):
            current_id, current_vec = plane_neighbors[i]
            next_id, next_vec = plane_neighbors[(i+1) % len(plane_neighbors)]
            
            # Get flow vectors from awareness gradients
            current_point = config_space.points.get(current_id)
            next_point = config_space.points.get(next_id)
            
            if not current_point or not next_point:
                continue
                
            # Calculate flow from awareness gradients
            current_flow = calculate_flow_vector(config_space, current_id)
            next_flow = calculate_flow_vector(config_space, next_id)
            
            # Calculate segment vector (midpoint to midpoint)
            segment_vec = (next_vec - current_vec) / 2
            
            # Project onto orthogonal plane
            segment_vec_plane = np.array([segment_vec[j] for j in plane_indices])
            
            # Calculate average flow in orthogonal directions
            flow1_plane = np.array([current_flow[j] for j in plane_indices])
            flow2_plane = np.array([next_flow[j] for j in plane_indices])
            avg_flow_plane = (flow1_plane + flow2_plane) / 2
            
            # NEW: Apply curvature effect to flow, if available
            if curvature_field and current_id in curvature_field and next_id in curvature_field:
                # Get curvature values
                current_curvature = curvature_field[current_id].get('local_curvature', 0.0)
                next_curvature = curvature_field[next_id].get('local_curvature', 0.0)
                
                # Get curvature gradients if available
                current_gradient = curvature_field[current_id].get('curvature_gradient')
                next_gradient = curvature_field[next_id].get('curvature_gradient')
                
                # Apply curvature enhancement to flow
                # Flow is enhanced in regions of high curvature gradient
                if current_gradient is not None and isinstance(current_gradient, np.ndarray):
                    # Project gradient onto orthogonal plane
                    gradient_plane = np.array([current_gradient[j] for j in plane_indices])
                    # Scale by curvature
                    gradient_contribution = gradient_plane * current_curvature * 0.3
                    # Add to flow (preserving direction)
                    flow_mag = np.linalg.norm(flow1_plane)
                    if flow_mag > 0.001:
                        flow1_plane += gradient_contribution
                
                if next_gradient is not None and isinstance(next_gradient, np.ndarray):
                    # Project gradient onto orthogonal plane
                    gradient_plane = np.array([next_gradient[j] for j in plane_indices])
                    # Scale by curvature
                    gradient_contribution = gradient_plane * next_curvature * 0.3
                    # Add to flow (preserving direction)
                    flow_mag = np.linalg.norm(flow2_plane)
                    if flow_mag > 0.001:
                        flow2_plane += gradient_contribution
                
                # Recalculate average flow with curvature influence
                avg_flow_plane = (flow1_plane + flow2_plane) / 2
            
            # Calculate dot product (flow along segment)
            contribution = np.dot(avg_flow_plane, segment_vec_plane)
            
            # Add to circulation
            plane_circulation += contribution
        
        # Store circulation component for this axis
        circulation[axis_idx] = plane_circulation
    
    return circulation

def calculate_relational_vector(
    config_space: 'ConfigurationSpace',
    point1_id: str,
    point2_id: str
) -> np.ndarray:
    """
    Calculate the relational vector between two points in configuration space.
    
    Args:
        config_space: The configuration space
        point1_id: First point ID
        point2_id: Second point ID
        
    Returns:
        Relational vector (3D)
    """
    # Get points
    point1 = config_space.points.get(point1_id)
    point2 = config_space.points.get(point2_id)
    
    if not point1 or not point2:
        return np.zeros(3)
    
    # Check if there's a phase relation
    if hasattr(point1, 'phase_relations') and point2_id in point1.phase_relations:
        # Use phase relation to construct vector
        theta_diff, phi_diff = point1.phase_relations[point2_id]
        
        # Convert to 3D vector
        # x,y components from theta (major circle)
        # z component from phi (minor circle)
        rel_vector = np.array([
            math.cos(theta_diff),
            math.sin(theta_diff),
            phi_diff / math.pi
        ])
        
        # Apply relation strength if available
        if hasattr(point1, 'relations') and point2_id in point1.relations:
            rel_strength = point1.relations[point2_id]
            rel_vector *= rel_strength
            
        return rel_vector
    
    # Fallback to using emergent position difference
    if hasattr(config_space, 'get_emergent_position'):
        try:
            pos1 = config_space.get_emergent_position(point1_id)
            pos2 = config_space.get_emergent_position(point2_id)
            
            # Calculate difference on toroidal manifold
            theta_diff = (pos2[0] - pos1[0] + math.pi) % (2 * math.pi) - math.pi
            phi_diff = (pos2[1] - pos1[1] + math.pi) % (2 * math.pi) - math.pi
            
            # Convert to 3D vector
            rel_vector = np.array([
                math.cos(theta_diff),
                math.sin(theta_diff),
                phi_diff / math.pi
            ])
            
            # Apply relation strength if available
            if hasattr(point1, 'relations') and point2_id in point1.relations:
                rel_strength = point1.relations[point2_id]
                rel_vector *= rel_strength
                
            return rel_vector
        except Exception:
            pass
    
    # Return zero vector if no relation can be established
    return np.zeros(3)

def calculate_flow_vector(
    config_space: 'ConfigurationSpace',
    point_id: str,
    curvature_field: Dict[str, Dict[str, Any]] = None  # Added curvature field parameter
) -> np.ndarray:
    """
    Calculate flow vector for a point based on its awareness gradients and curvature.
    
    Args:
        config_space: The configuration space
        point_id: Point ID
        curvature_field: Optional dict mapping point_id -> curvature properties
        
    Returns:
        Flow vector (3D)
    """
    # Get point
    point = config_space.points.get(point_id)
    if not point:
        return np.zeros(3)
    
    # Initialize flow vector
    flow = np.zeros(3)
    
    # Get neighbors from configuration space
    if hasattr(config_space, 'neighborhoods') and point_id in config_space.neighborhoods:
        neighbors = config_space.neighborhoods[point_id]
        
        # Calculate flow from gradients
        for neighbor_id in neighbors:
            neighbor = config_space.points.get(neighbor_id)
            if not neighbor:
                continue
                
            # Calculate gradient contribution
            rel_vector = calculate_relational_vector(config_space, point_id, neighbor_id)
            
            # Get awareness gradient
            if hasattr(point, 'gradients') and neighbor_id in point.gradients:
                gradient = point.gradients[neighbor_id]
                
                # Scale relational vector by gradient
                flow_contribution = rel_vector * gradient
                
                # Add to flow
                flow += flow_contribution
    
    # NEW: Apply curvature influence to flow vector
    if curvature_field and point_id in curvature_field:
        # Get curvature properties
        local_curvature = curvature_field[point_id].get('local_curvature', 0.0)
        curvature_gradient = curvature_field[point_id].get('curvature_gradient')
        
        if local_curvature > 0.1 and curvature_gradient is not None:
            if isinstance(curvature_gradient, np.ndarray) and np.linalg.norm(curvature_gradient) > 0.001:
                # Flow follows curvature gradient (like objects falling in gravity)
                curvature_contribution = curvature_gradient * local_curvature * 0.4
                flow += curvature_contribution
    
    # Normalize if needed
    flow_magnitude = np.linalg.norm(flow)
    if flow_magnitude > 0.001:
        return flow / flow_magnitude * math.tanh(flow_magnitude)  # Nonlinear scaling
    
    return flow

def identify_vortex_lines(
    config_space: 'ConfigurationSpace',
    circulation_vectors: Dict[str, np.ndarray],
    threshold: float = 0.3,
    curvature_field: Dict[str, Dict[str, Any]] = None  # Added curvature field parameter
) -> List[Dict[str, Any]]:
    """
    Identify vortex lines in the field based on vectorial circulation.
    Pure function that accepts computed circulation vectors.
    
    Args:
        config_space: The configuration space
        circulation_vectors: Dict mapping point_id -> circulation vector
        threshold: Circulation magnitude threshold
        curvature_field: Optional dict mapping point_id -> curvature properties
        
    Returns:
        List of vortex line dictionaries
    """
    # Find points with significant circulation
    vortex_points = {}
    for point_id, circ_vec in circulation_vectors.items():
        circ_mag = np.linalg.norm(circ_vec)
        
        # NEW: Adjust threshold based on curvature
        effective_threshold = threshold
        if curvature_field and point_id in curvature_field:
            # Get local curvature
            local_curvature = curvature_field[point_id].get('local_curvature', 0.0)
            # Higher curvature decreases the threshold (easier to form vortices)
            curvature_factor = local_curvature * 0.3
            effective_threshold = max(0.1, threshold - curvature_factor)
        
        if circ_mag > effective_threshold:
            # Store with circulation info
            point = config_space.points.get(point_id)
            if point:
                vortex_points[point_id] = {
                    'circulation': circ_vec,
                    'magnitude': circ_mag,
                    'direction': circ_vec / circ_mag if circ_mag > 0 else np.zeros(3)
                }
    
    # Group into vortex lines (aligned circulation)
    vortex_lines = []
    visited = set()
    
    for start_id, start_info in vortex_points.items():
        if start_id in visited:
            continue
            
        # Get circulation direction
        start_direction = start_info['direction']
        
        # Start new vortex line
        line_points = [start_id]
        line_circulation = [start_info['circulation']]
        visited.add(start_id)
        
        # Find connected points with aligned circulation
        current_id = start_id
        
        # Forward tracing
        while True:
            # Get neighbors
            if hasattr(config_space, 'neighborhoods') and current_id in config_space.neighborhoods:
                neighbors = config_space.neighborhoods[current_id]
            else:
                break
                
            # Find best aligned neighbor
            best_neighbor = None
            best_alignment = threshold  # Minimum threshold
            
            for neighbor_id in neighbors:
                if neighbor_id in visited or neighbor_id not in vortex_points:
                    continue
                    
                # Calculate alignment
                neighbor_direction = vortex_points[neighbor_id]['direction']
                alignment = np.dot(start_direction, neighbor_direction)
                
                # NEW: Enhance alignment if curvature is high in both points
                if curvature_field and current_id in curvature_field and neighbor_id in curvature_field:
                    # Get curvature values
                    current_curvature = curvature_field[current_id].get('local_curvature', 0.0)
                    neighbor_curvature = curvature_field[neighbor_id].get('local_curvature', 0.0)
                    
                    # Average curvature
                    avg_curvature = (current_curvature + neighbor_curvature) / 2
                    
                    # Enhance alignment in regions of high curvature
                    curvature_boost = avg_curvature * 0.2
                    alignment += curvature_boost
                
                if alignment > best_alignment:
                    best_alignment = alignment
                    best_neighbor = neighbor_id
            
            if best_neighbor:
                line_points.append(best_neighbor)
                line_circulation.append(vortex_points[best_neighbor]['circulation'])
                visited.add(best_neighbor)
                current_id = best_neighbor
            else:
                break
        
        # Backward tracing
        current_id = start_id
        backward_direction = -start_direction
        
        while True:
            # Get neighbors
            if hasattr(config_space, 'neighborhoods') and current_id in config_space.neighborhoods:
                neighbors = config_space.neighborhoods[current_id]
            else:
                break
                
            # Find best aligned neighbor
            best_neighbor = None
            best_alignment = threshold  # Minimum threshold
            
            for neighbor_id in neighbors:
                if neighbor_id in visited or neighbor_id not in vortex_points:
                    continue
                    
                # Calculate alignment with backward direction
                neighbor_direction = vortex_points[neighbor_id]['direction']
                alignment = np.dot(backward_direction, neighbor_direction)
                
                # NEW: Enhance alignment if curvature is high in both points
                if curvature_field and current_id in curvature_field and neighbor_id in curvature_field:
                    # Get curvature values
                    current_curvature = curvature_field[current_id].get('local_curvature', 0.0)
                    neighbor_curvature = curvature_field[neighbor_id].get('local_curvature', 0.0)
                    
                    # Average curvature
                    avg_curvature = (current_curvature + neighbor_curvature) / 2
                    
                    # Enhance alignment in regions of high curvature
                    curvature_boost = avg_curvature * 0.2
                    alignment += curvature_boost
                
                if alignment > best_alignment:
                    best_alignment = alignment
                    best_neighbor = neighbor_id
            
            if best_neighbor:
                # Insert at beginning
                line_points.insert(0, best_neighbor)
                line_circulation.insert(0, vortex_points[best_neighbor]['circulation'])
                visited.add(best_neighbor)
                current_id = best_neighbor
            else:
                break
        
        # Add vortex line if significant
        if len(line_points) >= 3:
            # Calculate average properties
            avg_circulation = np.mean(np.array(line_circulation), axis=0)
            avg_magnitude = np.linalg.norm(avg_circulation)
            
            # Calculate coherence (directional consistency)
            circulation_directions = np.array([c/np.linalg.norm(c) for c in line_circulation 
                                               if np.linalg.norm(c) > 0.001])
            
            coherence = 1.0
            if len(circulation_directions) > 1:
                # Average angular deviation from mean direction
                mean_direction = np.mean(circulation_directions, axis=0)
                norm = np.linalg.norm(mean_direction)
                if norm > 0.001:
                    mean_direction = mean_direction / norm
                    
                    # Calculate consistency (dot products with mean)
                    coherence = np.mean([np.dot(d, mean_direction) for d in circulation_directions])
            
            # NEW: Calculate average curvature along the vortex line
            avg_curvature = 0.0
            if curvature_field:
                curvature_values = [curvature_field.get(p_id, {}).get('local_curvature', 0.0) for p_id in line_points]
                avg_curvature = sum(curvature_values) / len(curvature_values) if curvature_values else 0.0
            
            # Create vortex line entry
            vortex_line = {
                'points': line_points,
                'length': len(line_points),
                'avg_circulation': avg_circulation.tolist(),
                'magnitude': avg_magnitude,
                'direction': (avg_circulation / avg_magnitude).tolist() if avg_magnitude > 0 else [0,0,0],
                'coherence': coherence,
                'avg_curvature': avg_curvature  # NEW: Average curvature
            }
            
            vortex_lines.append(vortex_line)
    
    return vortex_lines

def calculate_polarity_gradient(
    config_space: 'ConfigurationSpace',
    polarity_field: Dict[str, float],
    point_id: str,
    curvature_field: Dict[str, Dict[str, Any]] = None  # Added curvature field parameter
) -> np.ndarray:
    """
    Calculate polarity gradient vector for a point in configuration space.
    
    Args:
        config_space: The configuration space
        polarity_field: Dict mapping point_id -> polarity value
        point_id: Point ID
        curvature_field: Optional dict mapping point_id -> curvature properties
        
    Returns:
        Gradient vector (3D)
    """
    # Skip if point not in configuration space
    if point_id not in config_space.points:
        return np.zeros(3)
    
    # Get point polarity
    point_polarity = polarity_field.get(point_id, 0.0)
    
    # Get neighbors from configuration space
    if hasattr(config_space, 'neighborhoods') and point_id in config_space.neighborhoods:
        neighbors = config_space.neighborhoods[point_id]
    else:
        return np.zeros(3)
    
    # Initialize weighted gradients
    weighted_gradients = []
    weights = []
    
    for neighbor_id in neighbors:
        if neighbor_id not in config_space.points or neighbor_id not in polarity_field:
            continue
            
        # Get neighbor polarity
        neighbor_polarity = polarity_field[neighbor_id]
        
        # Calculate polarity difference
        polarity_diff = neighbor_polarity - point_polarity
        
        # Get relational vector
        rel_vector = calculate_relational_vector(config_space, point_id, neighbor_id)
        
        # Skip if no valid relation
        if np.linalg.norm(rel_vector) < 0.001:
            continue
            
        # Normalize direction
        rel_norm = np.linalg.norm(rel_vector)
        rel_direction = rel_vector / rel_norm
        
        # Calculate distance
        if hasattr(config_space, 'calculate_emergent_distance'):
            distance = config_space.calculate_emergent_distance(point_id, neighbor_id)
        else:
            distance = rel_norm
            
        # Avoid division by zero
        if distance < 0.001:
            distance = 0.001
        
        # NEW: Apply curvature influence on distance
        # Higher curvature effectively reduces distance (enhances gradient)
        effective_distance = distance
        if curvature_field:
            # Get curvature values for both points
            point_curvature = curvature_field.get(point_id, {}).get('local_curvature', 0.0)
            neighbor_curvature = curvature_field.get(neighbor_id, {}).get('local_curvature', 0.0)
            
            # Average curvature
            avg_curvature = (point_curvature + neighbor_curvature) / 2
            
            # Reduce effective distance in high curvature regions
            if avg_curvature > 0.2:
                curvature_contraction = avg_curvature * 0.3
                effective_distance = distance * (1.0 - curvature_contraction)
                effective_distance = max(0.001, effective_distance)  # Ensure positive
        
        # Calculate gradient contribution (polarity change per unit distance)
        contribution = rel_direction * (polarity_diff / effective_distance)
        
        # Weight by inverse distance (closer neighbors have more influence)
        weight = 1.0 / effective_distance
        
        weighted_gradients.append(contribution * weight)
        weights.append(weight)
    
    # Calculate weighted average
    if weighted_gradients:
        total_weights = sum(weights)
        if total_weights > 0:
            gradient = sum(weighted_gradients) / total_weights
        else:
            gradient = np.zeros(3)
    else:
        gradient = np.zeros(3)
    
    # NEW: Add direct curvature influence on polarity gradient
    if curvature_field and point_id in curvature_field:
        # Get curvature gradient
        curvature_gradient = curvature_field[point_id].get('curvature_gradient')
        local_curvature = curvature_field[point_id].get('local_curvature', 0.0)
        
        if curvature_gradient is not None and isinstance(curvature_gradient, np.ndarray):
            if np.linalg.norm(curvature_gradient) > 0.001 and local_curvature > 0.2:
                # Add curvature contribution to gradient
                # This creates a "gravitational" bias for polarity flow
                curvature_contribution = curvature_gradient * local_curvature * 0.3
                gradient += curvature_contribution
    
    return gradient

def detect_polarity_domains(
    config_space: 'ConfigurationSpace',
    polarity_field: Dict[str, float],
    threshold: float = 0.3,
    curvature_field: Dict[str, Dict[str, Any]] = None  # Added curvature field parameter
) -> List[Dict[str, Any]]:
    """
    Detect coherent polarity domains in configuration space.
    
    Args:
        config_space: The configuration space
        polarity_field: Dict mapping point_id -> polarity value
        threshold: Similarity threshold for domain inclusion
        curvature_field: Optional dict mapping point_id -> curvature properties
        
    Returns:
        List of domain dictionaries
    """
    # Group points into domains based on polarity and spatial proximity
    domains = []
    visited = set()
    
    # Filter points with superposition state
    active_points = {}
    for point_id, point in config_space.points.items():
        # Skip superposition points
        if point_id in polarity_field and is_in_superposition(point) == False:
            active_points[point_id] = polarity_field[point_id]
    
    # NEW: If curvature field is provided, adjust sorting to favor high curvature points as starting seeds
    if curvature_field:
        # Sort points by a combination of polarity strength and curvature
        sorted_points = []
        for point_id, polarity in active_points.items():
            # Get curvature value
            curvature = curvature_field.get(point_id, {}).get('local_curvature', 0.0)
            # Combined score = polarity strength + curvature contribution
            score = abs(polarity) + curvature * 0.5
            sorted_points.append((point_id, polarity, score))
        
        # Sort by score
        sorted_points.sort(key=lambda x: x[2], reverse=True)
        
        # Convert back to (point_id, polarity) format
        sorted_points = [(p[0], p[1]) for p in sorted_points]
    else:
        # Sort points by polarity strength (original behavior)
        sorted_points = sorted(
            active_points.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
    
    # Start from points with strongest polarity or highest combined score
    for point_id, polarity in sorted_points:
        if point_id in visited:
            continue
            
        # Skip very weak polarity
        if abs(polarity) < threshold:
            continue
            
        # Start new domain
        domain_points = [point_id]
        to_visit = [point_id]
        visited.add(point_id)
        
        # Polarity sign determines domain type
        domain_type = 'positive' if polarity > 0 else 'negative'
        domain_sign = 1 if polarity > 0 else -1
        
        # NEW: Get initial curvature value if available
        initial_curvature = 0.0
        if curvature_field and point_id in curvature_field:
            initial_curvature = curvature_field[point_id].get('local_curvature', 0.0)
        
        # BFS to find connected similar polarity
        while to_visit:
            current_id = to_visit.pop(0)
            
            # Get neighbors
            if hasattr(config_space, 'neighborhoods') and current_id in config_space.neighborhoods:
                neighbors = config_space.neighborhoods[current_id]
            else:
                neighbors = []
            
            # NEW: Get current point's curvature if available
            current_curvature = 0.0
            if curvature_field and current_id in curvature_field:
                current_curvature = curvature_field[current_id].get('local_curvature', 0.0)
            
            for neighbor_id in neighbors:
                if neighbor_id in visited or neighbor_id not in active_points:
                    continue
                    
                neighbor_polarity = active_points[neighbor_id]
                
                # Check polarity similarity
                neighbor_sign = 1 if neighbor_polarity > 0 else -1
                
                # Get neighbor's curvature if available
                neighbor_curvature = 0.0
                if curvature_field and neighbor_id in curvature_field:
                    neighbor_curvature = curvature_field[neighbor_id].get('local_curvature', 0.0)
                
                # NEW: Calculate effective threshold based on curvature
                # Higher curvature means easier domain inclusion (like gravity wells)
                effective_threshold = threshold
                if curvature_field:
                    # Use average curvature between points
                    avg_curvature = (current_curvature + neighbor_curvature) / 2
                    
                    # Reduce threshold in high curvature regions
                    curvature_factor = avg_curvature * 0.3
                    effective_threshold = max(0.1, threshold - curvature_factor)
                
                # Domain inclusion requires same polarity sign and sufficient strength
                if (neighbor_sign == domain_sign and 
                    abs(neighbor_polarity) >= effective_threshold):
                    
                    # Add to domain
                    domain_points.append(neighbor_id)
                    to_visit.append(neighbor_id)
                    visited.add(neighbor_id)
        
        # Add domain if significant
        if len(domain_points) >= 3:
            # Calculate average properties
            domain_polarities = [active_points[p_id] for p_id in domain_points]
            avg_polarity = sum(domain_polarities) / len(domain_polarities)
            
            # Calculate polarity coherence
            variance = sum((p - avg_polarity)**2 for p in domain_polarities) / len(domain_polarities)
            coherence = 1.0 - min(1.0, variance * 5.0)
            
            # Calculate spatial properties if available
            spatial_coherence = 1.0
            avg_position = None
            
            if hasattr(config_space, 'get_emergent_position'):
                try:
                    # Get positions
                    positions = []
                    for p_id in domain_points:
                        pos = config_space.get_emergent_position(p_id)
                        positions.append(pos)
                        
                    if positions:
                        # Calculate average position (needs special handling for torus)
                        theta_coords = [p[0] for p in positions]
                        phi_coords = [p[1] for p in positions]
                        
                        # Use circular mean for toroidal coordinates
                        avg_theta = circular_mean(theta_coords)
                        avg_phi = circular_mean(phi_coords)
                        avg_position = (avg_theta, avg_phi)
                        
                        # Calculate spatial dispersion
                        total_distance = 0.0
                        for p_id in domain_points:
                            for other_id in domain_points:
                                if p_id != other_id:
                                    dist = config_space.calculate_emergent_distance(p_id, other_id)
                                    total_distance += dist
                        
                        # Average distance between points
                        if len(domain_points) > 1:
                            pair_count = len(domain_points) * (len(domain_points) - 1)
                            avg_distance = total_distance / pair_count
                            
                            # Convert to coherence (lower distance = higher coherence)
                            spatial_coherence = max(0.1, 1.0 - avg_distance)
                except Exception:
                    # Fallback if spatial calculation fails
                    pass
            
            # NEW: Calculate curvature properties of the domain
            domain_curvature = 0.0
            max_domain_curvature = 0.0
            curvature_coherence = 1.0
            
            if curvature_field:
                # Collect curvature values
                curvature_values = []
                for p_id in domain_points:
                    if p_id in curvature_field:
                        curvature = curvature_field[p_id].get('local_curvature', 0.0)
                        curvature_values.append(curvature)
                
                if curvature_values:
                    # Calculate domain curvature statistics
                    domain_curvature = sum(curvature_values) / len(curvature_values)
                    max_domain_curvature = max(curvature_values)
                    
                    # Calculate curvature coherence (variance-based)
                    if len(curvature_values) > 1:
                        curvature_variance = sum((c - domain_curvature)**2 for c in curvature_values) / len(curvature_values)
                        curvature_coherence = 1.0 - min(1.0, curvature_variance * 10.0)
            
            # Create domain entry
            domain = {
                'points': domain_points,
                'size': len(domain_points),
                'type': domain_type,
                'avg_polarity': avg_polarity,
                'polarity_coherence': coherence,
                'spatial_coherence': spatial_coherence,
                'overall_coherence': (coherence + spatial_coherence) / 2,
                'avg_curvature': domain_curvature,            # NEW: Average curvature
                'max_curvature': max_domain_curvature,        # NEW: Maximum curvature
                'curvature_coherence': curvature_coherence    # NEW: Curvature coherence
            }
            
            # Add spatial info if available
            if avg_position:
                domain['avg_position'] = avg_position
            
            domains.append(domain)
    
    return domains

def identify_domain_interfaces(
    config_space: 'ConfigurationSpace',
    domains: List[Dict[str, Any]],
    gradient_field: Dict[str, np.ndarray],
    curvature_field: Dict[str, Dict[str, Any]] = None  # Added curvature field parameter
) -> List[Dict[str, Any]]:
    """
    Identify interfaces between opposing polarity domains.
    
    Args:
        config_space: The configuration space
        domains: List of domain dictionaries
        gradient_field: Dict mapping point_id -> gradient vector
        curvature_field: Optional dict mapping point_id -> curvature properties
        
    Returns:
        List of interface dictionaries
    """
    # Find pairs of opposing domains
    interfaces = []
    processed_pairs = set()
    
    for i, domain1 in enumerate(domains):
        for j, domain2 in enumerate(domains[i+1:], i+1):
            # Skip if already processed
            pair_key = frozenset([i, j])
            if pair_key in processed_pairs:
                continue
            
            processed_pairs.add(pair_key)
            
            # Check if domains have opposite polarity
            if (domain1['type'] == 'positive' and domain2['type'] == 'negative') or \
               (domain1['type'] == 'negative' and domain2['type'] == 'positive'):
                
                # Find points at the interface
                interface_pairs = find_interface_points(config_space, domain1, domain2)
                
                if interface_pairs:
                    # Calculate interface properties
                    # Get the average gradient magnitude at interface
                    gradient_magnitudes = []
                    
                    # NEW: Track curvature at interface
                    interface_curvatures = []
                    
                    for p1, p2 in interface_pairs:
                        if p1 in gradient_field and p2 in gradient_field:
                            grad1 = gradient_field[p1]
                            grad2 = gradient_field[p2]
                            
                            # Average magnitude
                            mag1 = np.linalg.norm(grad1)
                            mag2 = np.linalg.norm(grad2)
                            avg_mag = (mag1 + mag2) / 2
                            
                            gradient_magnitudes.append(avg_mag)
                            
                            # NEW: Get curvature values if available
                            if curvature_field:
                                curvature1 = curvature_field.get(p1, {}).get('local_curvature', 0.0)
                                curvature2 = curvature_field.get(p2, {}).get('local_curvature', 0.0)
                                avg_curvature = (curvature1 + curvature2) / 2
                                interface_curvatures.append(avg_curvature)
                    
                    # Calculate average gradient magnitude
                    if gradient_magnitudes:
                        avg_gradient = sum(gradient_magnitudes) / len(gradient_magnitudes)
                    else:
                        avg_gradient = 0.0
                    
                    # NEW: Calculate average interface curvature
                    avg_interface_curvature = 0.0
                    if interface_curvatures:
                        avg_interface_curvature = sum(interface_curvatures) / len(interface_curvatures)
                    
                    # NEW: Calculate curvature gradient across interface
                    curvature_gradient = 0.0
                    if curvature_field and domain1.get('avg_curvature') is not None and domain2.get('avg_curvature') is not None:
                        # Calculate difference between domain curvatures
                        curvature_gradient = domain1['avg_curvature'] - domain2['avg_curvature']
                    
                    # NEW: Calculate interface tension with curvature influence
                    # Curvature increases tension (higher curvature = stronger interface)
                    base_tension = avg_gradient * len(interface_pairs)
                    curvature_factor = 1.0 + avg_interface_curvature * 0.4
                    interface_tension = base_tension * curvature_factor
                    
                    # Create interface entry
                    interface = {
                        'domain1_idx': i,
                        'domain2_idx': j,
                        'domain1_type': domain1['type'],
                        'domain2_type': domain2['type'],
                        'interface_pairs': interface_pairs,
                        'size': len(interface_pairs),
                        'avg_gradient': avg_gradient,
                        'avg_curvature': avg_interface_curvature,         # NEW: Average curvature
                        'curvature_gradient': curvature_gradient,         # NEW: Curvature gradient
                        'tension': interface_tension                     # NEW: Enhanced tension
                    }
                    
                    interfaces.append(interface)
    
    return interfaces

def find_interface_points(
    config_space: 'ConfigurationSpace',
    domain1: Dict[str, Any],
    domain2: Dict[str, Any]
) -> List[Tuple[str, str]]:
    """
    Find pairs of points that form the interface between two domains.
    
    Args:
        config_space: The configuration space
        domain1: First domain dictionary
        domain2: Second domain dictionary
        
    Returns:
        List of (point1_id, point2_id) pairs
    """
    interface_pairs = []
    
    # Get domain point sets
    domain1_points = set(domain1['points'])
    domain2_points = set(domain2['points'])
    
    # Check for connections between domains
    for point1_id in domain1['points']:
        # Get neighbors
        if hasattr(config_space, 'neighborhoods') and point1_id in config_space.neighborhoods:
            neighbors = config_space.neighborhoods[point1_id]
        else:
            neighbors = []
        
        # Find neighbors in domain2
        for neighbor_id in neighbors:
            if neighbor_id in domain2_points:
                # This is an interface pair
                interface_pairs.append((point1_id, neighbor_id))
    
    return interface_pairs

def calculate_collapse_direction_field(
    config_space: 'ConfigurationSpace',
    polarity_field: Dict[str, float],
    flow_field: Dict[str, np.ndarray] = None,
    curvature_field: Dict[str, Dict[str, Any]] = None  # Added curvature field parameter
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate the collapse direction vector field across all points.
    
    Args:
        config_space: The configuration space
        polarity_field: Dict mapping point_id -> polarity value
        flow_field: Optional dict mapping point_id -> flow vector
        curvature_field: Optional dict mapping point_id -> curvature gradient
        
    Returns:
        Dictionary mapping point_id -> field vector data
    """
    direction_field = {}
    
    # Generate flow field if not provided
    if flow_field is None:
        flow_field = {}
        for point_id in config_space.points:
            flow_field[point_id] = calculate_flow_vector(config_space, point_id, curvature_field)
    
    # Calculate for each point
    for point_id, point in config_space.points.items():
        # Get point properties
        point_polarity = polarity_field.get(point_id, 0.0)
        is_superposition = is_in_superposition(point)
        
        # Calculate saturation
        if hasattr(point, 'get_saturation'):
            saturation = point.get_saturation()
        elif hasattr(point, 'grain_saturation'):
            saturation = point.grain_saturation
        else:
            # Calculate from basic properties
            saturation = calculate_point_saturation(point)
        
        if is_superposition:
            # Superposition collapses based on neighborhood influence
            collapse_dir = calculate_superposition_direction(config_space, point_id, polarity_field, curvature_field)
            collapse_mag = 0.8  # High potential for collapse
            field_type = 'superposition'
            
        else:
            # Calculate direction from multiple field influences
            flow_vector = flow_field.get(point_id, np.zeros(3))
            
            # Create polarity vector from scalar polarity
            polarity_vector = np.array([0.0, 0.0, point_polarity])
            
            # Get curvature vector if available
            if curvature_field and point_id in curvature_field:
                curvature_vector = curvature_field[point_id].get('curvature_gradient', np.zeros(3))
                local_curvature = curvature_field[point_id].get('local_curvature', 0.0)
            else:
                curvature_vector = np.zeros(3)
                local_curvature = 0.0
            
            # Get neighbor influence
            relation_vector = calculate_relation_vector(config_space, point_id)
            
            # Weight factors - dependent on saturation
            # Higher saturation = more influence from curvature and polarization
            # Lower saturation = more influence from flow and relations
            flow_weight = max(0.1, 0.5 - saturation * 0.4)
            polarity_weight = 0.3 + saturation * 0.2
            curvature_weight = max(0.1, local_curvature * 0.5)  # NEW: Scale by actual curvature
            relation_weight = max(0.1, 0.4 - saturation * 0.3)
            
            # Normalize weights to sum to 1.0
            total_weight = flow_weight + polarity_weight + curvature_weight + relation_weight
            if total_weight > 0:
                flow_weight /= total_weight
                polarity_weight /= total_weight
                curvature_weight /= total_weight
                relation_weight /= total_weight
            
            # Scale vectors by weights
            flow_contri = flow_vector * flow_weight if np.any(flow_vector) else np.zeros(3)
            polarity_contri = polarity_vector * polarity_weight if np.any(polarity_vector) else np.zeros(3)
            curvature_contri = curvature_vector * curvature_weight if np.any(curvature_vector) else np.zeros(3)
            relation_contri = relation_vector * relation_weight if np.any(relation_vector) else np.zeros(3)
            
            # Sum the contributions
            collapse_vector = flow_contri + polarity_contri + curvature_contri + relation_contri
            
            # Calculate magnitude
            collapse_mag = np.linalg.norm(collapse_vector)
            
            # Determine field type based on dominant contribution
            contributions = [
                ('flow', np.linalg.norm(flow_contri)),
                ('polarity', np.linalg.norm(polarity_contri)),
                ('curvature', np.linalg.norm(curvature_contri)),
                ('relation', np.linalg.norm(relation_contri))
            ]
            
            # Get dominant contribution
            field_type = max(contributions, key=lambda x: x[1])[0]
            
            # Normalize direction
            if collapse_mag > 0.001:
                collapse_dir = collapse_vector / collapse_mag
            else:
                # No clear direction
                collapse_dir = np.zeros(3)
                field_type = 'neutral'
        
        # Determine if this is a singular point (where field converges or diverges)
        is_singular = check_singularity(config_space, point_id, collapse_dir, collapse_mag, curvature_field)
        
        # Store field data
        direction_field[point_id] = {
            'direction': collapse_dir,
            'magnitude': collapse_mag,
            'type': field_type,
            'is_singular': is_singular,
            'local_curvature': local_curvature if curvature_field and point_id in curvature_field else 0.0
        }
    
    return direction_field

def calculate_superposition_direction(
    config_space: 'ConfigurationSpace',
    point_id: str, 
    polarity_field: Dict[str, float],
    curvature_field: Dict[str, Dict[str, Any]] = None  # Added curvature field parameter
) -> np.ndarray:
    """
    Calculate collapse direction for a superposition point.
    
    Args:
        config_space: The configuration space
        point_id: Point ID
        polarity_field: Dict mapping point_id -> polarity value
        curvature_field: Optional dict mapping point_id -> curvature properties
        
    Returns:
        Collapse direction vector (3D)
    """
    # Get point
    point = config_space.points.get(point_id)
    if not point:
        return np.zeros(3)
        
    # Get point polarity
    point_polarity = polarity_field.get(point_id, 0.0)
    
    # Get neighbors
    if hasattr(config_space, 'neighborhoods') and point_id in config_space.neighborhoods:
        neighbors = config_space.neighborhoods[point_id]
    else:
        # Default direction aligns with inherent polarity
        return np.array([0.0, 0.0, point_polarity])
    
    # Calculate influences from neighbors
    neighbor_influences = []
    
    for neighbor_id in neighbors:
        neighbor = config_space.points.get(neighbor_id)
        if not neighbor:
            continue
            
        # Get relational vector
        rel_vector = calculate_relational_vector(config_space, point_id, neighbor_id)
        
        # Get influence strength
        if not is_in_superposition(neighbor):
            # Non-superposition neighbor has stronger influence
            
            # Get neighbor awareness
            if hasattr(neighbor, 'awareness'):
                awareness_factor = neighbor.awareness
            else:
                awareness_factor = 0.0
            
            # Get neighbor saturation
            if hasattr(neighbor, 'get_saturation'):
                saturation_factor = neighbor.get_saturation()
            elif hasattr(neighbor, 'grain_saturation'):
                saturation_factor = neighbor.grain_saturation
            else:
                saturation_factor = calculate_point_saturation(neighbor)
            
            # NEW: Get neighbor curvature
            curvature_factor = 0.0
            if curvature_field and neighbor_id in curvature_field:
                curvature_factor = curvature_field[neighbor_id].get('local_curvature', 0.0)
            
            # Calculate direction
            if np.linalg.norm(rel_vector) > 0.001:
                direction = rel_vector
            else:
                # Use polarity direction
                neighbor_polarity = polarity_field.get(neighbor_id, 0.0)
                direction = np.array([0.0, 0.0, neighbor_polarity])
            
            # Calculate influence strength with curvature enhancement
            # Higher curvature = stronger pull on superposition
            base_strength = (abs(awareness_factor) + saturation_factor) / 2
            curvature_enhancement = 1.0 + curvature_factor * 0.6  # Significant boost
            strength = base_strength * curvature_enhancement
            
            # Add to influences
            if np.linalg.norm(direction) > 0.001:
                neighbor_influences.append((direction, strength))
        else:
            # Superposition-to-superposition influence is weaker
            if np.linalg.norm(rel_vector) > 0.001:
                # Add slight influence
                neighbor_influences.append((rel_vector, 0.1))
    
    # NEW: Add direct curvature influence
    if curvature_field and point_id in curvature_field:
        # Get curvature gradient
        curvature_gradient = curvature_field[point_id].get('curvature_gradient')
        local_curvature = curvature_field[point_id].get('local_curvature', 0.0)
        
        if curvature_gradient is not None and isinstance(curvature_gradient, np.ndarray):
            if np.linalg.norm(curvature_gradient) > 0.001 and local_curvature > 0.1:
                # Add curvature gradient as direction influence
                # This acts like gravitational pull on superposition state
                curvature_strength = local_curvature * 0.5  # Significant influence
                neighbor_influences.append((curvature_gradient, curvature_strength))
    
    # Calculate weighted sum of influences
    if neighbor_influences:
        weighted_sum = np.zeros(3)
        total_weight = 0.0
        
        for direction, weight in neighbor_influences:
            weighted_sum += direction * weight
            total_weight += weight
            
        if total_weight > 0:
            weighted_sum /= total_weight
            
            # Normalize
            norm = np.linalg.norm(weighted_sum)
            if norm > 0.001:
                return weighted_sum / norm
    
    # Default: align with polarity
    return np.array([0.0, 0.0, point_polarity])

def calculate_relation_vector(
    config_space: 'ConfigurationSpace',
    point_id: str,
    curvature_field: Dict[str, Dict[str, Any]] = None  # Added curvature field parameter
) -> np.ndarray:
    """
    Calculate net relation vector from all relations.
    
    Args:
        config_space: The configuration space
        point_id: Point ID
        curvature_field: Optional dict mapping point_id -> curvature properties
        
    Returns:
        Relation vector (3D)
    """
    # Get point
    point = config_space.points.get(point_id)
    if not point:
        return np.zeros(3)
    
    # Check for relations
    if not hasattr(point, 'relations'):
        return np.zeros(3)
    
    # Sum influence from all relations
    relation_sum = np.zeros(3)
    
    for rel_id, rel_strength in point.relations.items():
        # Calculate relational vector
        rel_vec = calculate_relational_vector(config_space, point_id, rel_id)
        
        # NEW: Apply curvature enhancement if available
        if curvature_field and point_id in curvature_field and rel_id in curvature_field:
            # Get curvature values
            point_curvature = curvature_field[point_id].get('local_curvature', 0.0)
            rel_curvature = curvature_field[rel_id].get('local_curvature', 0.0)
            
            # Calculate curvature differential
            curvature_diff = rel_curvature - point_curvature
            
            # Enhance relation vector based on curvature differential
            # This makes flow prefer downhill curvature gradients
            if curvature_diff > 0.1:
                # Flowing to higher curvature
                enhancement = 1.0 + curvature_diff * 0.5
                rel_vec *= enhancement
        
        # Add to sum
        relation_sum += rel_vec
    
    # Normalize if necessary
    rel_norm = np.linalg.norm(relation_sum)
    if rel_norm > 0.001:
        return relation_sum / rel_norm
    
    return np.zeros(3)

def check_singularity(
    config_space: 'ConfigurationSpace',
    point_id: str,
    direction: np.ndarray,
    magnitude: float,
    curvature_field: Dict[str, Dict[str, Any]] = None  # Added curvature field parameter
) -> bool:
    """
    Check if a point is a singular point in the field.
    
    Args:
        config_space: The configuration space
        point_id: Point ID
        direction: Collapse direction vector
        magnitude: Collapse magnitude
        curvature_field: Optional dict mapping point_id -> curvature properties
        
    Returns:
        True if the point is a singular point
    """
    # Special case for zero or near-zero magnitude
    if magnitude < 0.1:
        return True
        
    # NEW: Apply curvature-specific singularity detection
    # High curvature points are often field singularities
    if curvature_field and point_id in curvature_field:
        local_curvature = curvature_field[point_id].get('local_curvature', 0.0)
        
        # Extreme curvature peaks/valleys are automatic singularities
        if local_curvature > 0.7:
            return True
            
        # Check for curvature maximum relative to neighbors
        if hasattr(config_space, 'neighborhoods') and point_id in config_space.neighborhoods:
            neighbors = config_space.neighborhoods[point_id]
            
            if len(neighbors) >= 3:
                # Get neighbor curvatures
                neighbor_curvatures = []
                for neighbor_id in neighbors:
                    if neighbor_id in curvature_field:
                        neighbor_curvature = curvature_field[neighbor_id].get('local_curvature', 0.0)
                        neighbor_curvatures.append(neighbor_curvature)
                
                # Check if point is a local maximum of curvature
                if neighbor_curvatures and local_curvature > max(neighbor_curvatures):
                    return True
    
    # Get neighbors
    if hasattr(config_space, 'neighborhoods') and point_id in config_space.neighborhoods:
        neighbors = config_space.neighborhoods[point_id]
    else:
        return False
    
    if len(neighbors) < 3:
        return False
        
    # Get relational vectors to neighbors
    neighbor_vectors = []
    neighbor_directions = []
    
    for neighbor_id in neighbors:
        # Get relational vector
        rel_vector = calculate_relational_vector(config_space, point_id, neighbor_id)
        
        # Skip if no clear relation
        if np.linalg.norm(rel_vector) < 0.001:
            continue
            
        neighbor_vectors.append((neighbor_id, rel_vector))
    
    # Need sufficient neighbors with vectors
    if len(neighbor_vectors) < 3:
        return False
    
    # Calculate mean direction of neighbor vectors
    vectors = [v[1] for v in neighbor_vectors]
    mean_vec = np.mean(vectors, axis=0)
    mean_norm = np.linalg.norm(mean_vec)
    
    if mean_norm < 0.001:
        # Very inconsistent directions - likely a singular point
        return True
        
    mean_dir = mean_vec / mean_norm
    
    # Calculate average angular deviation
    total_deviation = 0.0
    for _, vec in neighbor_vectors:
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 0.001:
            vec_dir = vec / vec_norm
            
            # Calculate angle between directions
            dot_product = np.dot(mean_dir, vec_dir)
            # Ensure valid domain for arccos
            dot_product = max(-1.0, min(1.0, dot_product))
            angle = math.acos(dot_product)
            total_deviation += angle
    
    avg_deviation = total_deviation / len(neighbor_vectors)
    
    # High angular deviation indicates a singular point
    # Threshold at 0.5 radians (~30 degrees)
    return avg_deviation > 0.5

def trace_field_lines(
    config_space: 'ConfigurationSpace',
    direction_field: Dict[str, Dict[str, Any]],
    start_points: List[str] = None,
    max_lines: int = 5,
    curvature_field: Dict[str, Dict[str, Any]] = None  # Added curvature field parameter
) -> List[Dict[str, Any]]:
    """
    Trace field lines following the collapse direction field.
    
    Args:
        config_space: The configuration space
        direction_field: Calculated direction field
        start_points: Optional list of starting point IDs
        max_lines: Maximum number of lines to trace
        curvature_field: Optional dict mapping point_id -> curvature properties
        
    Returns:
        List of field line dictionaries
    """
    field_lines = []
    
    # Identify singular points to use as starting references if not provided
    if start_points is None:
        singular_points = []
        
        # NEW: Sort points by a combined score of singularity and curvature
        point_scores = []
        
        for point_id, field_data in direction_field.items():
            # Base score from field data
            if field_data['is_singular']:
                singular_type = determine_singularity_type(
                    config_space, point_id, direction_field, curvature_field)
                
                base_score = 1.0
                if singular_type == 'source':
                    base_score = 1.5  # Prefer sources as starting points
                
                # NEW: Add curvature contribution to score
                curvature_score = 0.0
                if curvature_field and point_id in curvature_field:
                    local_curvature = curvature_field[point_id].get('local_curvature', 0.0)
                    curvature_score = local_curvature * 0.8  # Significant boost
                
                # Combined score
                total_score = base_score + curvature_score
                
                point_scores.append({
                    'point_id': point_id,
                    'type': singular_type,
                    'magnitude': field_data['magnitude'],
                    'score': total_score
                })
        
        # Sort by score
        point_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Add to singular points
        singular_points = point_scores
        
        # Start from sources or high-score singular points
        sources = [p for p in singular_points if p['type'] == 'source']
        
        # If no sources, use high-score points
        if not sources:
            # Use points with highest combined score
            start_points = [p['point_id'] for p in singular_points[:min(max_lines, len(singular_points))]]
        else:
            start_points = [p['point_id'] for p in sources[:min(max_lines, len(sources))]]
    
    # Track visited points to avoid overlap
    visited = set()
    
    # Generate field lines from each start point
    for start_id in start_points:
        if start_id in visited or start_id not in direction_field:
            continue
        
        # Generate field line
        field_line = trace_single_field_line(config_space, start_id, direction_field, visited, curvature_field)
        
        if field_line and len(field_line['points']) >= 3:
            field_lines.append(field_line)
    
    return field_lines

def trace_single_field_line(
    config_space: 'ConfigurationSpace',
    start_id: str,
    direction_field: Dict[str, Dict[str, Any]],
    visited: Set[str],
    curvature_field: Dict[str, Dict[str, Any]] = None  # Added curvature field parameter
) -> Dict[str, Any]:
    """
    Trace a single field line starting from a specific point.
    
    Args:
        config_space: The configuration space
        start_id: Starting point ID
        direction_field: Calculated direction field
        visited: Set of already visited points
        curvature_field: Optional dict mapping point_id -> curvature properties
        
    Returns:
        Field line dictionary or None
    """
    # Skip if not in direction field
    if start_id not in direction_field:
        return None

def calculate_line_coherence(directions: List[np.ndarray]) -> float:
    """
    Calculate the directional coherence of a field line.
    
    Args:
        directions: List of direction vectors
        
    Returns:
        Coherence value (0.0 to 1.0)
    """
    if len(directions) < 2:
        return 1.0
        
    # Calculate average alignment between consecutive directions
    total_alignment = 0.0
    
    for i in range(len(directions) - 1):
        # Get consecutive vectors
        v1 = directions[i]
        v2 = directions[i+1]
        
        # Calculate alignment
        dot_product = np.dot(v1, v2)
        
        # Ensure valid domain
        dot_product = max(-1.0, min(1.0, dot_product))
        
        # Convert to angle
        angle = math.acos(dot_product)
        
        # Convert angle to alignment (1.0 = perfect alignment, 0.0 = orthogonal)
        alignment = 1.0 - angle / math.pi
        
        total_alignment += alignment
    
    # Calculate average
    avg_alignment = total_alignment / (len(directions) - 1)
    
    return avg_alignment

def determine_singularity_type(
    config_space: 'ConfigurationSpace', 
    point_id: str, 
    direction_field: Dict[str, Dict[str, Any]],
    curvature_field: Dict[str, Dict[str, Any]] = None  # Added curvature field parameter
) -> str:
    """
    Determine the type of singular point.
    
    Args:
        config_space: The configuration space
        point_id: Point ID
        direction_field: Calculated direction field
        curvature_field: Optional dict mapping point_id -> curvature properties
        
    Returns:
        Singular point type
    """
    # Get neighbors
    if hasattr(config_space, 'neighborhoods') and point_id in config_space.neighborhoods:
        neighbors = config_space.neighborhoods[point_id]
    else:
        return 'unknown'
    
    if len(neighbors) < 3:
        return 'unknown'
    
    # NEW: Check for curvature extrema
    if curvature_field and point_id in curvature_field:
        local_curvature = curvature_field[point_id].get('local_curvature', 0.0)
        
        # Get neighbor curvatures
        neighbor_curvatures = []
        for neighbor_id in neighbors:
            if neighbor_id in curvature_field:
                neighbor_curvature = curvature_field[neighbor_id].get('local_curvature', 0.0)
                neighbor_curvatures.append(neighbor_curvature)
        
        if neighbor_curvatures:
            # Local maximum of curvature is a potential sink (like gravity well)
            if local_curvature > 0.5 and all(local_curvature >= nc for nc in neighbor_curvatures):
                # Double-check with flow pattern before deciding
                pass
            
            # Local minimum of curvature is a potential source
            elif local_curvature < 0.1 and all(local_curvature <= nc for nc in neighbor_curvatures):
                # Double-check with flow pattern before deciding
                pass
        
    # Check for source or sink pattern
    outward_count = 0
    inward_count = 0
    
    for neighbor_id in neighbors:
        if neighbor_id not in direction_field:
            continue
            
        neighbor_data = direction_field[neighbor_id]
        neighbor_dir = neighbor_data['direction']
        
        # Skip if no clear direction
        if np.linalg.norm(neighbor_dir) < 0.001:
            continue
            
        # Get relational vector from point to neighbor
        rel_vector = calculate_relational_vector(config_space, point_id, neighbor_id)
        
        # If no relational vector, can't determine
        if np.linalg.norm(rel_vector) < 0.001:
            continue
            
        # Normalize
        rel_norm = np.linalg.norm(rel_vector)
        rel_dir = rel_vector / rel_norm
        
        # Check alignment with field direction
        dot_product = np.dot(rel_dir, neighbor_dir)
        
        # Positive dot product = pointing away (source)
        # Negative dot product = pointing toward (sink)
        if dot_product > 0.3:
            outward_count += 1
        elif dot_product < -0.3:
            inward_count += 1
    
    # NEW: Combine with curvature data for final determination
    if curvature_field and point_id in curvature_field:
        local_curvature = curvature_field[point_id].get('local_curvature', 0.0)
        
        # High curvature biases toward sink determination
        if local_curvature > 0.6:
            # High curvature regions are more likely to be sinks (gravitational analogy)
            inward_count += int(local_curvature * 2)
        # Very low curvature biases toward source
        elif local_curvature < 0.1:
            outward_count += 1
    
    # Determine type based on counts
    if outward_count > 2 and outward_count > inward_count * 2:
        return 'source'
    elif inward_count > 2 and inward_count > outward_count * 2:
        return 'sink'
    elif outward_count > 0 and inward_count > 0:
        return 'saddle'
    else:
        return 'unknown'

def detect_cascade_paths(
    config_space: 'ConfigurationSpace',
    polarity_field: Dict[str, float],
    direction_field: Dict[str, Dict[str, Any]],
    curvature_field: Dict[str, Dict[str, Any]] = None  # Added curvature field parameter
) -> List[Dict[str, Any]]:
    """
    Detect potential cascade collapse paths in the field.
    
    Args:
        config_space: The configuration space
        polarity_field: Dict mapping point_id -> polarity value
        direction_field: Calculated direction field
        curvature_field: Optional dict mapping point_id -> curvature properties
        
    Returns:
        List of cascade path dictionaries
    """
    # Use field lines as starting points for potential cascades
    field_lines = trace_field_lines(config_space, direction_field, curvature_field=curvature_field)
    
    # Convert field lines to cascade paths
    cascade_paths = []
    
    for field_line in field_lines:
        # Get points
        point_ids = field_line.get('points', [])
        
        # Skip short lines
        if len(point_ids) < 3:
            continue
        
        # Calculate average properties
        polarities = [polarity_field.get(pid, 0.0) for pid in point_ids]
        avg_polarity = sum(polarities) / len(polarities) if polarities else 0.0
        
        # Determine path type
        path_type = 'structure' if avg_polarity > 0 else 'decay'
        
        # Calculate lightlike ratio (points with low saturation)
        lightlike_count = 0
        freedoms = []
        
        # NEW: Track curvature properties
        curvatures = []
        
        for point_id in point_ids:
            point = config_space.points.get(point_id)
            if not point:
                continue
                
            # Calculate saturation
            if hasattr(point, 'get_saturation'):
                saturation = point.get_saturation()
            elif hasattr(point, 'grain_saturation'):
                saturation = point.grain_saturation
            else:
                saturation = calculate_point_saturation(point)
                
            # Check if lightlike
            if saturation < 0.2:
                lightlike_count += 1
                
            # Calculate freedom
            if hasattr(point, 'degrees_of_freedom'):
                freedom = point.degrees_of_freedom
            else:
                freedom = 1.0 - saturation
                
            freedoms.append(freedom)
            
            # NEW: Get curvature if available
            if curvature_field and point_id in curvature_field:
                curvature = curvature_field[point_id].get('local_curvature', 0.0)
                curvatures.append(curvature)
        
        # Calculate metrics
        point_count = len(point_ids)
        lightlike_ratio = lightlike_count / point_count if point_count > 0 else 0.0
        avg_freedom = sum(freedoms) / len(freedoms) if freedoms else 0.0
        
        # NEW: Calculate curvature metrics
        avg_curvature = sum(curvatures) / len(curvatures) if curvatures else 0.0
        max_curvature = max(curvatures) if curvatures else 0.0
        
        # NEW: Calculate cascade potential with curvature influence
        # Structure cascades are enhanced by high curvature gradients
        # Decay cascades are less affected by curvature
        base_potential = min(1.0, field_line.get('length', 0) / 10) * (
            0.5 + 0.3 * abs(avg_polarity) + 0.2 * lightlike_ratio
        )
        
        # Apply curvature influence based on cascade type
        curvature_factor = 0.0
        if path_type == 'structure':
            # Structure cascades enhanced by high curvature
            curvature_factor = avg_curvature * 0.3
        else:
            # Decay cascades less affected by curvature
            curvature_factor = avg_curvature * 0.1
            
        cascade_potential = min(1.0, base_potential + curvature_factor)
        
        # Create cascade path
        cascade_path = {
            'points': point_ids,
            'length': len(point_ids),
            'type': path_type,
            'coherence': field_line.get('coherence', 0.0),
            'avg_polarity': avg_polarity,
            'avg_freedom': avg_freedom,
            'lightlike_ratio': lightlike_ratio,
            'start_type': field_line.get('start_type'),
            'end_type': field_line.get('end_type'),
            'avg_curvature': avg_curvature,           # NEW: Average curvature
            'max_curvature': max_curvature,           # NEW: Maximum curvature
            'cascade_potential': cascade_potential    # NEW: Enhanced cascade potential
        }
        
        cascade_paths.append(cascade_path)
    
    return cascade_paths

def identify_field_boundaries(
    config_space: 'ConfigurationSpace',
    domains: List[Dict[str, Any]],
    gradient_field: Dict[str, np.ndarray],
    curvature_field: Dict[str, Dict[str, Any]] = None  # Added curvature field parameter
) -> List[Dict[str, Any]]:
    """
    Identify boundaries in the field where properties change significantly.
    
    Args:
        config_space: The configuration space
        domains: List of domain dictionaries
        gradient_field: Dict mapping point_id -> gradient vector
        curvature_field: Optional dict mapping point_id -> curvature properties
        
    Returns:
        List of boundary dictionaries
    """
    # Use domain interfaces to identify boundaries
    interfaces = identify_domain_interfaces(config_space, domains, gradient_field, curvature_field)
    
    # Convert to boundaries
    boundaries = []
    
    for interface in interfaces:
        boundary = {
            'type': 'polarity_interface',
            'pairs': interface.get('interface_pairs', []),
            'tension': interface.get('tension', 0.0),
            'size': interface.get('size', 0),
            'domain1_type': interface.get('domain1_type'),
            'domain2_type': interface.get('domain2_type'),
            'avg_curvature': interface.get('avg_curvature', 0.0),         # NEW: Average curvature
            'curvature_gradient': interface.get('curvature_gradient', 0.0)  # NEW: Curvature gradient
        }
        
        boundaries.append(boundary)
    
    
        
    # Get starting direction
    start_data = direction_field[start_id]
    
    # Skip if no clear direction
    if np.linalg.norm(start_data['direction']) < 0.001:
        return None
        
    # Start tracing
    line_points = [start_id]
    line_directions = [start_data['direction']]
    
    # NEW: Track curvature along the field line
    line_curvatures = []
    if curvature_field and start_id in curvature_field:
        line_curvatures.append(curvature_field[start_id].get('local_curvature', 0.0))
    else:
        line_curvatures.append(0.0)
    
    visited.add(start_id)
    current_id = start_id
    
    # Maximum line length
    max_length = 10
    
    # Follow field direction
    while len(line_points) < max_length:
        # Get neighbors
        if hasattr(config_space, 'neighborhoods') and current_id in config_space.neighborhoods:
            neighbors = config_space.neighborhoods[current_id]
        else:
            break
        
        # Get current direction
        current_dir = direction_field[current_id]['direction']
        
        # Find next point in field line
        best_neighbor = None
        best_alignment = 0.3  # Minimum alignment threshold
        
        for neighbor_id in neighbors:
            if neighbor_id in visited or neighbor_id not in direction_field:
                continue
                
            # Get relational vector
            rel_vector = calculate_relational_vector(config_space, current_id, neighbor_id)
            
            # Skip if no clear relation
            if np.linalg.norm(rel_vector) < 0.001:
                continue
                
            # Normalize
            rel_norm = np.linalg.norm(rel_vector)
            rel_dir = rel_vector / rel_norm
            
            # Check alignment with current direction
            alignment = np.dot(current_dir, rel_dir)
            
            # NEW: Apply curvature influence on alignment
            # Lower alignment threshold in regions of high curvature
            effective_alignment = alignment
            if curvature_field and current_id in curvature_field and neighbor_id in curvature_field:
                # Get curvature values
                current_curvature = curvature_field[current_id].get('local_curvature', 0.0)
                neighbor_curvature = curvature_field[neighbor_id].get('local_curvature', 0.0)
                
                # Favor paths following increasing curvature gradient
                if neighbor_curvature > current_curvature:
                    # Enhanced alignment toward higher curvature (like gravity)
                    curvature_diff = (neighbor_curvature - current_curvature)
                    curvature_bonus = curvature_diff * 0.3
                    effective_alignment = alignment + curvature_bonus
            
            if effective_alignment > best_alignment:
                best_alignment = effective_alignment
                best_neighbor = neighbor_id
        
        if best_neighbor:
            # Add to field line
            line_points.append(best_neighbor)
            line_directions.append(direction_field[best_neighbor]['direction'])
            
            # Add curvature
            if curvature_field and best_neighbor in curvature_field:
                line_curvatures.append(curvature_field[best_neighbor].get('local_curvature', 0.0))
            else:
                line_curvatures.append(0.0)
            
            visited.add(best_neighbor)
            current_id = best_neighbor
            
            # Check if hit a sink or another singular point
            if direction_field[best_neighbor]['is_singular']:
                break
        else:
            break
    
    # Create field line entry
    if len(line_points) >= 2:
        # Calculate coherence (directional consistency)
        line_coherence = calculate_line_coherence(line_directions)
        
        # NEW: Calculate average and maximum curvature
        avg_curvature = sum(line_curvatures) / len(line_curvatures) if line_curvatures else 0.0
        max_curvature = max(line_curvatures) if line_curvatures else 0.0
        
        # NEW: Calculate curvature coherence (consistency)
        curvature_coherence = 1.0
        if len(line_curvatures) > 1:
            # Calculate variance of curvature
            mean_curvature = avg_curvature
            variance = sum((c - mean_curvature)**2 for c in line_curvatures) / len(line_curvatures)
            # Convert to coherence (lower variance = higher coherence)
            curvature_coherence = 1.0 - min(1.0, variance * 10.0)
        
        field_line = {
            'points': line_points,
            'length': len(line_points),
            'coherence': line_coherence,
            'start_type': start_data['type'],
            'end_type': direction_field[line_points[-1]]['type'] if line_points else None,
            'avg_curvature': avg_curvature,            # NEW: Average curvature
            'max_curvature': max_curvature,            # NEW: Maximum curvature
            'curvature_coherence': curvature_coherence  # NEW: Curvature coherence
        }
        
        return field_line
    
    return None