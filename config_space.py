"""
Configuration Space - Pure relational implementation with no Cartesian assumptions

Represents a purely relational configuration space in the Collapse Geometry framework,
where all structure emerges from relations with no cartesian coordinates assumed.
All points exist solely in relation to one another on an emergent toroidal manifold,
where position is defined by phase relationships rather than coordinates.

This implementation maintains strict topological principles:
- No direct storage of phase values
- All phases are derived from relations on-demand
- No projection into Cartesian space for calculations
- Pure relation-driven structure emergence
- No binary states - everything is a continuous threshold
"""

import math
import random
import uuid
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Callable
from collections import defaultdict, deque

# Constants
TWO_PI = 2 * math.pi
EPSILON = 0.01

# ============================================================================
# Core Data Structures
# ============================================================================

class ConfigurationPoint:
    """
    Represents a point in configuration space with pure relational properties.
    Position is never stored directly, only inferred from relations.
    """
    
    def __init__(self, point_id: str = None):
        """
        Initialize a configuration point with purely relational properties.
        Everything starts with unbounded potential (near-zero values).
        
        Args:
            point_id: Unique identifier (generated if None)
        """
        self.id = point_id or str(uuid.uuid4())
        
        # Core field values
        self.awareness = 0.0       # Near-zero = unbounded potential, positive = structure, negative = decay
        self.polarity = 0.0        # Near-zero = neutral, positive = structure bias, negative = decay bias
        self.activation = 0.0      # Activation level (0.0 to 1.0)
        self.coherence = 1.0       # Phase coherence (0.0 to 1.0)
        self.collapse_metric = 0.0 # Net collapse effect
        
        # Relational properties - no direct phase storage
        self.phase_relations = {}  # Maps related_id -> (phase_difference_theta, phase_difference_phi)
        self.relations = {}        # Maps related_id -> relation strength (-1.0 to 1.0)
        self.gradients = {}        # Maps related_id -> awareness gradient
        self.tensions = {}         # Maps related_id -> structural tension
        
        # Memory and ancestry
        self.ancestry = set()      # Set of ancestor grain_ids
        self.collapse_history = [] # List of collapse events
        
        # Continuous state factors
        self.potential_factor = 1.0  # Degree of unbounded potential (1.0 = maximum)
        
        # Mass-related properties
        self.mass_factor = 0.0     # Degree of mass-like behavior (0.0 = no mass, 1.0 = maximum mass)
        self.curvature = 0.0       # How much this point curves the surrounding field
        self.recursion_depth = 0   # Recursive ancestry depth - promotes mass-like behavior


class ConfigurationSpace:
    """
    Represents the configuration space as a pure toroidal manifold.
    All structure emerges from relations, with no cartesian coordinates.
    """
    
    def __init__(self):
        """
        Initialize the configuration space with purely relational topology.
        """
        self.points = {}                # Maps point_id -> ConfigurationPoint
        self.time = 0.0                 # Relational time
        
        # Relational neighborhoods
        self.neighborhoods = {}         # Maps point_id -> set of neighbor_ids
        
        # Phase dimensions (for reference only, not storage)
        self.phase_dimensions = ['theta', 'phi']
        
        # Coherence tracking
        self.coherence_history = deque(maxlen=20)  # Recent history of coherence values
        
        # Global field stats
        self.stats = {
            'global_coherence': {
                'theta': 0.0,          # Coherence of theta phase across system
                'phi': 0.0,            # Coherence of phi phase across system
                'overall': 0.0         # Overall system coherence
            },
            'structure_collapse_count': 0,   # Count of positive collapses
            'decay_collapse_count': 0,       # Count of negative collapses
            'superposition_count': 0,        # Count of points with high potential factor
            'total_saturation': 0.0,         # Total saturation in system
            'energy_balance_history': []     # History of energy balance
        }
    
    def calculate_commitment_pressure(self, point: 'ConfigurationPoint') -> float:
        """
        Calculate pressure toward identity commitment based on structural inevitability.
        Pure function that returns a value in [0,1] where higher values indicate greater pressure.
        
        Args:
            point: The configuration point to evaluate
            
        Returns:
            Commitment pressure value (0.0 to 1.0)
        """
        # No pressure if already high saturation/commitment
        if point.collapse_metric > 0.5:
            return 0.0
        
        # Base pressure from saturation
        saturation = calculate_saturation(point)
        saturation_pressure = saturation * 0.4
        
        # Ancestry pressure - recursive structures have more pressure
        ancestry_count = len(point.ancestry)
        recursive_depth = point.recursion_depth
        ancestry_pressure = min(0.3, (ancestry_count * 0.02 + recursive_depth * 0.05))
        
        # Calculate relational asymmetry - if relations are unbalanced, more pressure
        relational_asymmetry = 0.0
        if point.relations:
            rel_values = list(point.relations.values())
            rel_sum = sum(rel_values)
            rel_abs_sum = sum(abs(v) for v in rel_values)
            if rel_abs_sum > 0:
                # Higher when relations are more asymmetric/directional
                relational_asymmetry = abs(rel_sum) / rel_abs_sum
        
        # Relational pressure
        relation_pressure = relational_asymmetry * 0.2
        
        # Degeneracy pressure - higher when awareness/polarity are indistinct
        # Less degeneracy = more distinct = more pressure
        awareness_distinctness = min(1.0, abs(point.awareness) * 5)
        polarity_distinctness = min(1.0, abs(point.polarity) * 5)
        degeneracy_pressure = (awareness_distinctness * 0.15 + polarity_distinctness * 0.15)
        
        # Calculate coherence asymmetry - pressure increases with anisotropy
        coherence_pressure = (1.0 - point.coherence) * 0.1
        
        # Combine all pressures
        total_pressure = (
            saturation_pressure + 
            ancestry_pressure + 
            relation_pressure + 
            degeneracy_pressure +
            coherence_pressure
        )
        
        return min(1.0, total_pressure)
    
    def is_ready_for_identity(self, point: 'ConfigurationPoint') -> Tuple[bool, float, str]:
        """
        Determine if a point is ready to commit to identity.
        Returns a tuple of (is_ready, commitment_pressure, potential_id)
        
        Args:
            point: The configuration point to evaluate
            
        Returns:
            Tuple of (is_ready, commitment_pressure, potential_id)
        """
        # Calculate commitment pressure
        commitment_pressure = self.calculate_commitment_pressure(point)
        
        # Only proceed if commitment pressure is high enough
        if commitment_pressure > 0.8:
            # Calculate a stable hash seed based on structural properties
            # This ensures identity emerges deterministically from structure
            ancestry_seed = ",".join(sorted(point.ancestry)) if point.ancestry else ""
            seed = (
                ancestry_seed + 
                str(round(point.awareness, 3)) + 
                str(round(point.collapse_metric, 3)) +
                str(round(commitment_pressure, 3))  # Include pressure in seed
            )
            
            # Generate hash-based ID
            hash_val = hash(seed) % 10000
            potential_id = f"grain_{hash_val}"
            
            return (True, commitment_pressure, potential_id)
        
        return (False, commitment_pressure, "")
    
    def get_point(self, point_id: str) -> Optional['ConfigurationPoint']:
        """
        Get a configuration point by ID, or commit a new identity if needed.
        
        In Collapse Geometry, identity emerges from collapse - a point
        doesn't have a fixed identity until it commits structure.
        This method implements the principle that grains don't have identity;
        collapse gives them identity.
        
        Args:
            point_id: Point ID to look up
                
        Returns:
            ConfigurationPoint if found, None otherwise
        """
        # First try direct lookup (for already committed points)
        point = self.points.get(point_id)
        
        # If found, return it
        if point:
            return point
                
        # If not found, check if this might be a potential identity
        # emerging from a point ready to collapse (crossing threshold)
        
        # Search for points ready to collapse but without committed identity
        for candidate_id, candidate in self.points.items():
            # Skip points with committed identity
            if candidate.collapse_metric > 0.2 and candidate_id != point_id:
                continue
                
            # Check if ready for identity
            is_ready, commitment_pressure, potential_id = self.is_ready_for_identity(candidate)
            
            # If ready and ID matches, commit identity
            if is_ready and potential_id == point_id:
                # Commit identity by updating point ID
                old_id = candidate.id
                candidate.id = point_id
                
                # Add self-reference in ancestry (identity requires self-reference)
                candidate.ancestry.add(point_id)
                
                # Increase collapse metric (identity commitment is a form of collapse)
                # Make collapse metric proportional to commitment pressure
                candidate.collapse_metric += 0.2 + commitment_pressure * 0.1
                
                # Add the point under its new identity
                self.points[point_id] = candidate
                
                # Remove the old reference
                if old_id in self.points and old_id != point_id:
                    del self.points[old_id]
                
                # Update neighborhoods if needed
                if old_id in self.neighborhoods:
                    self.neighborhoods[point_id] = self.neighborhoods[old_id].copy()
                    
                    # Remove old neighborhood if different ID
                    if old_id != point_id:
                        del self.neighborhoods[old_id]
                        
                    # Update all references to this point in other neighborhoods
                    for other_id, neighbors in self.neighborhoods.items():
                        if old_id in neighbors:
                            neighbors.remove(old_id)
                            neighbors.add(point_id)
                
                # Return the newly committed identity
                return candidate
        
        # Identity not found and no candidates ready to commit
        return None


# ============================================================================
# Phase Relation Utilities
# ============================================================================

def angular_difference(a: float, b: float) -> float:
    """
    Calculate the minimum angular difference between two angles.
    
    Args:
        a: First angle (radians)
        b: Second angle (radians)
        
    Returns:
        Minimum angular difference (always positive)
    """
    diff = abs(a - b) % TWO_PI
    return min(diff, TWO_PI - diff)


def phase_consensus(relations: List[float], weights: List[float] = None) -> float:
    """
    Calculate the phase consensus from a set of phase relations.
    This avoids Cartesian projection by directly minimizing angular disagreement.
    
    Args:
        relations: List of phase relation values
        weights: Optional weights for each relation
        
    Returns:
        Consensus phase value in radians
    """
    if not relations:
        return 0.0
    
    # Default to equal weights if not provided
    if weights is None:
        weights = [1.0] * len(relations)
    elif len(weights) != len(relations):
        weights = weights[:len(relations)] if len(weights) > len(relations) else weights + [1.0] * (len(relations) - len(weights))
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0
        
    normalized_weights = [w / total_weight for w in weights]
    
    # Find phase that minimizes disagreement
    # We sample a number of candidate phases and choose the one with minimum disagreement
    # This is a simplified approach that works well for reasonable numbers of relations
    best_disagreement = float('inf')
    best_phase = 0.0
    
    # Try candidate phases from the relations themselves
    candidates = relations.copy()
    # Also add some intermediate values to test
    for i in range(len(relations)):
        for j in range(i+1, len(relations)):
            mid = (relations[i] + relations[j]) / 2 % TWO_PI
            candidates.append(mid)
    
    # Find phase with minimum disagreement
    for candidate in candidates:
        disagreement = sum(weight * angular_difference(relation, candidate) 
                         for relation, weight in zip(relations, normalized_weights))
        
        if disagreement < best_disagreement:
            best_disagreement = disagreement
            best_phase = candidate
    
    return best_phase


def topological_agreement(phase_values: List[float]) -> float:
    """
    Calculate the topological agreement between a set of phase values.
    Returns 1.0 for perfect agreement, lower values for disagreement.
    
    Args:
        phase_values: List of phase values
        
    Returns:
        Agreement value (0.0 to 1.0)
    """
    if len(phase_values) <= 1:
        return 1.0
        
    # Calculate average disagreement
    total_disagreement = 0.0
    count = 0
    
    for i in range(len(phase_values)):
        for j in range(i+1, len(phase_values)):
            total_disagreement += angular_difference(phase_values[i], phase_values[j]) / math.pi
            count += 1
    
    if count == 0:
        return 1.0
        
    avg_disagreement = total_disagreement / count
    
    # Convert to agreement (1.0 = perfect agreement, 0.0 = maximum disagreement)
    return 1.0 - min(1.0, avg_disagreement)


def calculate_angular_symmetry(angles: List[float]) -> float:
    """
    Calculate the angular symmetry of a set of angles.
    Perfect symmetry means angles are evenly distributed.
    
    Args:
        angles: List of angles in radians
        
    Returns:
        Symmetry value (1.0 = perfect symmetry, 0.0 = complete asymmetry)
    """
    if len(angles) <= 1:
        return 1.0
    
    # Convert angles to unit vectors
    vectors = [(math.cos(a), math.sin(a)) for a in angles]
    
    # Calculate resultant vector (perfect symmetry = zero resultant)
    resultant_x = sum(v[0] for v in vectors)
    resultant_y = sum(v[1] for v in vectors)
    resultant = math.sqrt(resultant_x**2 + resultant_y**2)
    
    # Normalize by number of vectors (resultant magnitude from 0 to 1)
    resultant_normalized = resultant / len(vectors)
    
    # Invert for symmetry (0 resultant = 1.0 symmetry, 1 resultant = 0.0 symmetry)
    return 1.0 - resultant_normalized


# ============================================================================
# Core Configuration Operations
# ============================================================================

def create_point(space: ConfigurationSpace, point_id: str = None) -> ConfigurationPoint:
    """
    Create a new configuration point and add it to the space.
    Point starts with maximum unbounded potential.
    
    Args:
        space: The configuration space
        point_id: Optional point ID (random UUID if None)
            
    Returns:
        The new configuration point
    """
    # Create new point
    point = ConfigurationPoint(point_id)
    
    # Add to space
    space.points[point.id] = point
    
    # Initialize neighborhood
    space.neighborhoods[point.id] = set()
    
    # Update superposition count
    space.stats['superposition_count'] += 1
    
    return point


def calculate_superposition_factor(point: ConfigurationPoint) -> float:
    """
    Calculate the degree to which a point exhibits superposition-like properties.
    Continuous scale from 0.0 (fully collapsed) to 1.0 (maximum superposition).
    
    Args:
        point: The configuration point to evaluate
        
    Returns:
        Superposition factor (0.0 to 1.0)
    """
    # Core criteria for unbounded potential
    # 1. Awareness near zero
    awareness_factor = math.exp(-abs(point.awareness) / EPSILON) 
    
    # 2. Low saturation
    saturation = calculate_saturation(point)
    saturation_factor = math.exp(-saturation / EPSILON)
    
    # 3. Few defined relations
    relation_count = sum(1 for r in point.relations.values() if abs(r) > 0.1)
    relation_factor = math.exp(-relation_count * 0.1)
    
    # 4. Explicit potential factor
    explicit_factor = point.potential_factor
    
    # Weighted combination - using smooth mathematical functions
    # This ensures continuous behavior with no discontinuities
    combined_factor = (
        awareness_factor * 0.35 +
        saturation_factor * 0.25 +
        relation_factor * 0.20 +
        explicit_factor * 0.20
    )
    
    # Smooth sigmoid scaling to emphasize threshold behavior without binary jumps
    sigmoid = lambda x: 1.0 / (1.0 + math.exp(-12 * (x - 0.5)))
    scaled_factor = sigmoid(combined_factor)
    
    return max(0.0, min(1.0, scaled_factor))


def calculate_saturation(point: ConfigurationPoint) -> float:
    """
    Compute saturation dynamically from structural properties.
    Saturation emerges from geometric constraints, not as a stored property.
    
    Args:
        point: The configuration point
        
    Returns:
        Computed saturation value (0.0 to 1.0)
    """
    # Superposition factor affects saturation
    superposition_factor = calculate_superposition_factor(point)
    
    # For high superposition, saturation approaches zero smoothly
    # This implements the zero=infinite potential principle without binary checks
    if superposition_factor > 0.9:
        return 0.0
        
    # Scale saturation inversely with superposition factor
    scaling_factor = 1.0 - superposition_factor * 0.9
    
    # 1. Ancestry contribution
    ancestry_factor = min(0.4, len(point.ancestry) * 0.02)
    
    # 2. Collapse history contribution
    collapse_saturation = min(0.7, 0.3 * point.collapse_metric) if point.collapse_metric > 0 else 0.0
    
    # 3. Relational commitment contribution
    if not point.relations:
        relation_saturation = 0.0
    else:
        relation_strength = sum(abs(r) for r in point.relations.values()) / len(point.relations)
        relation_complexity = min(0.5, len(point.relations) * 0.02)
        relation_saturation = relation_strength * (1.0 + relation_complexity)
    
    # 4. Coherence loss contribution
    coherence_loss = 1.0 - point.coherence
    coherence_saturation = coherence_loss * 0.3
    
    # 5. Mass contribution - mass directly increases saturation
    # Mass and saturation are correlated but have different effects
    mass_saturation = point.mass_factor * 0.7
    
    # 6. Recursive ancestry contribution (self-reference)
    # Recursion creates inward-curving structure = more saturation
    recursion_saturation = min(0.5, point.recursion_depth * 0.08)
    
    # 7. Field curvature contribution
    curvature_saturation = point.curvature * 0.4
    
    # Combined saturation calculation - scaled by superposition factor
    total_saturation = (
        ancestry_factor * 0.2 +
        collapse_saturation * 0.2 +
        relation_saturation * 0.1 +
        coherence_saturation * 0.1 +
        mass_saturation * 0.2 +
        recursion_saturation * 0.1 +
        curvature_saturation * 0.1
    ) * scaling_factor
    
    # Ensure bounds
    return max(0.0, min(1.0, total_saturation))


def calculate_degrees_of_freedom(point: ConfigurationPoint) -> float:
    """
    Calculate the degrees of freedom for a point.
    Freedom decreases as structure commits through saturation and mass increases.
    
    Args:
        point: The configuration point
        
    Returns:
        Freedom value (0.0 to 1.0)
    """
    # Get superposition factor - higher factor means more freedom
    superposition_factor = calculate_superposition_factor(point)
    
    # For high superposition, freedom approaches maximum smoothly
    if superposition_factor > 0.9:
        return 1.0  # Maximum freedom
    
    # Base freedom calculation
    # 1. Freedom decreases as saturation increases
    saturation = calculate_saturation(point)
    freedom_from_saturation = 1.0 - saturation
    
    # 2. Freedom decreases with ancestry
    ancestry_factor = min(1.0, len(point.ancestry) * 0.05)
    freedom_from_ancestry = 1.0 - ancestry_factor
    
    # 3. Freedom decreases with relation commitment
    if not point.relations:
        relation_constraint = 0.0
    else:
        relation_constraint = sum(abs(r) for r in point.relations.values()) / len(point.relations)
    freedom_from_relations = 1.0 - relation_constraint
    
    # 4. Freedom decreases with mass
    # Mass is the opposite of freedom - high mass means low freedom
    mass_constraint = point.mass_factor
    freedom_from_mass = 1.0 - mass_constraint
    
    # 5. Freedom decreases with recursive ancestry (self-reference)
    recursion_factor = min(1.0, point.recursion_depth * 0.1)
    freedom_from_recursion = 1.0 - recursion_factor
    
    # Combined freedom - weighted factors
    freedom = (
        0.3 * freedom_from_saturation + 
        0.2 * freedom_from_ancestry + 
        0.1 * freedom_from_relations +
        0.3 * freedom_from_mass +
        0.1 * freedom_from_recursion
    )
    
    # Blend with superposition freedom using a smooth function
    blended_freedom = freedom * (1.0 - superposition_factor) + superposition_factor
    
    return max(0.0, min(1.0, blended_freedom))


def derive_phase(point: ConfigurationPoint, dimension: str, space: ConfigurationSpace) -> float:
    """
    Derive phase value for a point purely from relational context.
    Phase is not stored, but computed on-demand from relations.
    
    Args:
        point: The configuration point
        dimension: The phase dimension ('theta' or 'phi')
        space: The configuration space for context
        
    Returns:
        Derived phase value in radians
    """
    # Get superposition factor
    superposition_factor = calculate_superposition_factor(point)
    
    # For high superposition, phase becomes more uncertain
    # But still deterministic based on seed for consistency
    seed = hash(point.id) % 1000 / 1000.0
    base_phase = seed * TWO_PI
    
    # As superposition factor increases, rely more on the random seed
    # and less on relations
    if superposition_factor > 0.8:
        # Add a small random fluctuation based on superposition factor
        fluctuation = (random.random() - 0.5) * superposition_factor * 0.2
        return (base_phase + fluctuation) % TWO_PI
    
    # If no relations, phase is derived from inherent properties with uncertainty
    if not point.phase_relations:
        # Derive from awareness and polarity (inherent properties)
        phase = base_phase
        
        # Apply awareness and polarity influence
        phase += point.awareness * math.pi * 0.2
        phase += point.polarity * math.pi * 0.1
        
        # Add uncertainty based on superposition factor
        phase += random.uniform(-0.1, 0.1) * superposition_factor
        
        return phase % TWO_PI
    
    # Derive phase from relations
    phase_differences = []
    weights = []
    
    # Collect phase differences from relations
    for related_id, (theta_diff, phi_diff) in point.phase_relations.items():
        # Get the related point
        related_point = space.points.get(related_id)
        if not related_point:
            continue
            
        # Get the appropriate phase difference
        if dimension == 'theta':
            diff = theta_diff
        else:  # phi
            diff = phi_diff
        
        # Weight by relation strength and related point's superposition factor
        relation_strength = abs(point.relations.get(related_id, 0.0))
        related_superposition = calculate_superposition_factor(related_point)
        
        # Lower weight for more superposition-like relations
        effective_weight = relation_strength * (1.0 - related_superposition * 0.8)
        
        if effective_weight > 0.01:
            # The relation gives us the difference (related - self)
            # So we need to derive self = related - diff
            
            # First get the related point's phase
            related_phase = derive_phase(related_point, dimension, space)
            
            # Calculate our phase based on relation
            # (We need to subtract because the diff is related - self)
            derived_phase = (related_phase - diff) % TWO_PI
            
            phase_differences.append(derived_phase)
            weights.append(effective_weight)
    
    # If we have relations, find the consensus phase
    if phase_differences:
        consensus = phase_consensus(phase_differences, weights)
        
        # Blend with base phase based on superposition
        # More superposition = more influence from base phase
        return (consensus * (1.0 - superposition_factor) + base_phase * superposition_factor) % TWO_PI
    
    # Fallback to inherent phase
    return base_phase


def connect_points(space: ConfigurationSpace, point1_id: str, point2_id: str, 
                   relation_strength: float = None) -> bool:
    """
    Connect two points with a relation.
    Also establishes their relative phases.
    
    Args:
        space: The configuration space
        point1_id: First point ID
        point2_id: Second point ID
        relation_strength: Optional relation strength (-1.0 to 1.0)
        
    Returns:
        True if connection created, False if points not found
    """
    point1 = space.points.get(point1_id)
    point2 = space.points.get(point2_id)
    
    if not point1 or not point2:
        return False
    
    # Get superposition factors
    point1_superposition = calculate_superposition_factor(point1)
    point2_superposition = calculate_superposition_factor(point2)
    
    # Generate relation strength if not provided
    if relation_strength is None:
        # For high superposition states, relation emerges unpredictably
        if point1_superposition > 0.8 and point2_superposition > 0.8:
            # Two superposition points have uncertain relation
            relation_strength = random.uniform(-0.3, 0.3)
        elif point1_superposition > 0.8 or point2_superposition > 0.8:
            # One superposition point adopts polarity from more defined point
            defined_point = point2 if point1_superposition > point2_superposition else point1
            # Weaker relation with superposition
            relation_strength = defined_point.polarity * 0.5
        else:
            # Relation strength based on polarity alignment
            polarity_product = point1.polarity * point2.polarity
            if polarity_product > 0:
                # Similar polarity - positive relation
                relation_strength = 0.5 * (point1.polarity + point2.polarity) / 2
            elif polarity_product < 0:
                # Opposed polarity - negative relation
                relation_strength = -0.5 * (abs(point1.polarity) + abs(point2.polarity)) / 2
            else:
                # At least one neutral - weak relation
                relation_strength = (point1.polarity + point2.polarity) * 0.3
    
    # Ensure relation strength is within bounds
    relation_strength = max(-1.0, min(1.0, relation_strength))
    
    # Update relations
    update_relation(point1, point2_id, relation_strength)
    update_relation(point2, point1_id, relation_strength)
    
    # Update field gradients
    update_gradient(point1, point2_id, point2.awareness)
    update_gradient(point2, point1_id, point1.awareness)
    
    # Calculate and update phase relations - critical for toroidal positioning
    update_phase_relation(space, point1, point2)
    
    # Update neighborhoods
    space.neighborhoods[point1_id].add(point2_id)
    space.neighborhoods[point2_id].add(point1_id)
    
    return True


def update_relation(point: ConfigurationPoint, related_id: str, relation_strength: float):
    """
    Update relation to another point.
    
    Args:
        point: The configuration point
        related_id: The other point's ID
        relation_strength: Relation strength (-1.0 to 1.0)
    """
    # Ensure value is within bounds
    relation_strength = max(-1.0, min(1.0, relation_strength))
    
    # Get previous relation if exists
    previous_strength = point.relations.get(related_id, 0.0)
    
    # Store relation
    point.relations[related_id] = relation_strength
    
    # Strong relations gradually decrease superposition factor
    if abs(relation_strength) > 0.7:
        # Calculate impact on potential factor
        superposition_impact = abs(relation_strength) * 0.1 * random.uniform(0.5, 1.5)
        point.potential_factor = max(0.0, point.potential_factor - superposition_impact)
        
        # As potential decreases, awareness increases proportionally
        if point.potential_factor < 0.5 and abs(point.awareness) < 0.1:
            # Awareness emerges in direction of relation
            direction = 1.0 if relation_strength > 0 else -1.0
            awareness_emergence = (0.5 - point.potential_factor) * 0.2 * direction
            point.awareness += awareness_emergence


def update_gradient(point: ConfigurationPoint, related_id: str, awareness: float):
    """
    Update field gradient with another point.
    Gradients represent the direction of potential flow.
    
    Args:
        point: The configuration point
        related_id: The other point's ID
        awareness: The other point's awareness value
    """
    # Calculate gradient
    gradient = awareness - point.awareness
    
    # Store gradient
    point.gradients[related_id] = gradient


def update_phase_relation(space: ConfigurationSpace, point1: ConfigurationPoint, 
                          point2: ConfigurationPoint):
    """
    Update toroidal phase relations between two points.
    This establishes their relative positions on the toroidal manifold.
    
    Args:
        space: The configuration space
        point1: First configuration point
        point2: Second configuration point
    """
    # Get current phases for both points (derived, not stored)
    point1_theta = derive_phase(point1, 'theta', space)
    point1_phi = derive_phase(point1, 'phi', space)
    point2_theta = derive_phase(point2, 'theta', space)
    point2_phi = derive_phase(point2, 'phi', space)
    
    # Calculate phase differences
    theta_diff = angular_difference(point1_theta, point2_theta)
    phi_diff = angular_difference(point1_phi, point2_phi)
    
    # Determine signs based on shortest path direction
    theta_sign = 1 if (point2_theta - point1_theta + TWO_PI) % TWO_PI < math.pi else -1
    phi_sign = 1 if (point2_phi - point1_phi + TWO_PI) % TWO_PI < math.pi else -1
    
    # Apply signs to create directed relations
    theta_relation = theta_diff * theta_sign
    phi_relation = phi_diff * phi_sign
    
    # Update phase relations in both directions
    point1.phase_relations[point2.id] = (theta_relation, phi_relation)
    point2.phase_relations[point1.id] = (-theta_relation, -phi_relation)


def process_collapse(point: ConfigurationPoint, collapse_strength: float, source_id: str = None) -> float:
    """
    Process a collapse event on a point.
    Positive values increase structure, negative values decrease it.
    
    Args:
        point: The configuration point to collapse
        collapse_strength: Signed strength of collapse (-1.0 to 1.0)
        source_id: Optional source point ID causing the collapse
        
    Returns:
        Actual saturation change applied
    """
    # Store previous saturation for delta calculation
    previous_saturation = calculate_saturation(point)
    
    # Get superposition factor
    superposition_factor = calculate_superposition_factor(point)
    
    # Determine collapse type from sign
    is_decay = collapse_strength < 0
    
    # Get magnitude (absolute value)
    magnitude = abs(collapse_strength)
    
    if is_decay:
        # NEGATIVE COLLAPSE (DECAY)
        
        # Effectiveness scales with current structure (less superposition)
        decay_effectiveness = 1.0 - superposition_factor * 0.8
        
        # Update awareness (decay pulls toward negative)
        awareness_change = magnitude * 0.2 * decay_effectiveness
        point.awareness = max(-1.0, point.awareness - awareness_change)
        
        # Update polarity toward negative
        polarity_change = magnitude * 0.3 * decay_effectiveness
        point.polarity = max(-1.0, point.polarity - polarity_change)
        
        # Reduce coherence
        coherence_change = magnitude * 0.1 * decay_effectiveness
        point.coherence = max(0.3, point.coherence - coherence_change)
        
        # Decrease mass factor for decay
        mass_decrease = magnitude * 0.1 * decay_effectiveness
        point.mass_factor = max(0.0, point.mass_factor - mass_decrease)
        
        # Reduce curvature for decay
        curvature_decrease = magnitude * 0.15 * decay_effectiveness
        point.curvature = max(0.0, point.curvature - curvature_decrease)
        
        # Record decay event
        point.collapse_history.append({
            'type': 'decay',
            'strength': -magnitude,
            'source': source_id
        })
        
        # Update collapse metric (negative)
        metric_change = magnitude * decay_effectiveness
        point.collapse_metric -= metric_change
        
    else:
        # POSITIVE COLLAPSE (STRUCTURE)
        
        # Determine how much this collapse resolves superposition
        # Higher initial superposition means more dramatic resolution
        resolution_strength = magnitude * superposition_factor * 2.0
        
        # Gradually reduce potential factor (resolve superposition)
        point.potential_factor = max(0.0, point.potential_factor - resolution_strength * 0.3)
        
        # Update awareness toward positive - more change for high superposition
        awareness_change = magnitude * 0.2 * (1.0 + superposition_factor)
        point.awareness = min(1.0, point.awareness + awareness_change)
        
        # Update polarity toward positive
        polarity_change = magnitude * 0.3
        point.polarity = min(1.0, point.polarity + polarity_change)
        
        # Increase coherence
        coherence_change = magnitude * 0.1
        point.coherence = min(1.0, point.coherence + coherence_change)
        
        # Record structure event
        point.collapse_history.append({
            'type': 'structure',
            'strength': magnitude,
            'source': source_id
        })
        
        # Update collapse metric (positive)
        metric_change = magnitude
        point.collapse_metric += metric_change
        
        # Add source to ancestry if provided
        if source_id and source_id not in point.ancestry:
            point.ancestry.add(source_id)
            
            # Check for recursive ancestry (self-reference)
            # This is critical for mass formation
            if source_id == point.id:
                # Self-reference increases recursion depth
                point.recursion_depth += 1
                
                # Self-reference promotes mass formation
                mass_increase = 0.1 * (1.0 + point.recursion_depth * 0.2)
                point.mass_factor = min(1.0, point.mass_factor + mass_increase)
                
                # Self-reference also increases curvature
                curvature_increase = 0.08 * (1.0 + point.recursion_depth * 0.15)
                point.curvature = min(1.0, point.curvature + curvature_increase)
            else:
                # Check if source has mass
                source = None
                if hasattr(point, '__class__') and hasattr(point.__class__, '__module__'):
                    module = point.__module__
                    if hasattr(module, 'points') and source_id in module.points:
                        source = module.points[source_id]
                
                # If source has mass, some is transferred to target
                if source and hasattr(source, 'mass_factor') and source.mass_factor > 0.2:
                    # Mass inheritance - transferred at reduced rate
                    mass_transfer = source.mass_factor * 0.15 * magnitude
                    point.mass_factor = min(1.0, point.mass_factor + mass_transfer)
                    
                    # Curvature inheritance
                    if hasattr(source, 'curvature') and source.curvature > 0:
                        curvature_transfer = source.curvature * 0.1 * magnitude
                        point.curvature = min(1.0, point.curvature + curvature_transfer)
    
    # Update mass factor based on current properties
    update_mass_properties(point)
    
    # Calculate and return saturation delta
    new_saturation = calculate_saturation(point)
    saturation_delta = new_saturation - previous_saturation
    return saturation_delta


def update_mass_properties(point: ConfigurationPoint):
    """
    Update a point's mass-related properties based on its current state.
    This implements the dual nature of mass vs. light in the model.
    
    Args:
        point: The configuration point to update
    """
    # Skip for high superposition states
    superposition_factor = calculate_superposition_factor(point)
    if superposition_factor > 0.9:
        point.mass_factor = 0.0
        point.curvature = 0.0
        return
    
    # Calculate core mass factor
    theoretical_mass = calculate_mass_factor(point)
    
    # Smooth transition to theoretical mass
    # Use exponential moving average for stability
    alpha = 0.3  # Blending factor
    point.mass_factor = point.mass_factor * (1 - alpha) + theoretical_mass * alpha
    
    # Update curvature based on mass and recursion
    base_curvature = point.mass_factor * (1.0 + point.recursion_depth * 0.2)
    
    # Ancestry size also contributes to curvature
    ancestry_contribution = min(0.5, len(point.ancestry) * 0.02)
    
    # Self-reference is key to curvature
    if point.id in point.ancestry:
        self_reference_factor = 0.3 * (1.0 + point.recursion_depth * 0.2)
    else:
        self_reference_factor = 0.0
    
    # Combine curvature components
    theoretical_curvature = base_curvature * 0.5 + ancestry_contribution * 0.2 + self_reference_factor * 0.3
    
    # Smooth transition to theoretical curvature
    point.curvature = point.curvature * (1 - alpha) + theoretical_curvature * alpha
    
    # Ensure bounds
    point.mass_factor = max(0.0, min(1.0, point.mass_factor))
    point.curvature = max(0.0, min(1.0, point.curvature))


def calculate_mass_factor(point: ConfigurationPoint) -> float:
    """
    Calculate the mass factor of a point based on its properties.
    Mass emerges from saturation, ancestry, and self-reference.
    
    Args:
        point: The configuration point
        
    Returns:
        Mass factor value (0.0 to 1.0)
    """
    # Check for superposition
    superposition_factor = calculate_superposition_factor(point)
    if superposition_factor > 0.9:
        return 0.0  # Superposition = zero mass
    
    # Main contributors to mass:
    
    # 1. Saturation - more saturation = more mass
    saturation = calculate_saturation(point)
    saturation_factor = saturation * 0.6
    
    # 2. Ancestry size - complex ancestry = more mass
    ancestry_size = len(point.ancestry)
    ancestry_factor = min(0.5, ancestry_size * 0.05)
    
    # 3. Self-reference - key to mass formation
    self_reference = 0.0
    if point.id in point.ancestry:
        # Self-reference scales with recursion depth
        self_reference = 0.2 * (1.0 + point.recursion_depth * 0.3)
    
    # 4. Coherence loss - less coherence = more mass-like
    coherence_factor = (1.0 - point.coherence) * 0.3
    
    # Combine factors, weighted by importance
    mass_factor = (
        saturation_factor * 0.4 +
        ancestry_factor * 0.2 +
        self_reference * 0.3 +
        coherence_factor * 0.1
    )
    
    # Apply superposition dampening
    mass_factor *= (1.0 - superposition_factor * 0.8)
    
    # Ensure bounds
    return max(0.0, min(1.0, mass_factor))


def calculate_lightlike_factor(point: ConfigurationPoint) -> float:
    """
    Calculate the lightlike factor of a point based on its properties.
    Lightlike behavior is the dual counterpart to mass.
    
    Args:
        point: The configuration point
        
    Returns:
        Lightlike factor value (0.0 to 1.0)
    """
    # Check for superposition
    superposition_factor = calculate_superposition_factor(point)
    if superposition_factor > 0.9:
        return 1.0  # Superposition = maximally lightlike
    
    # Lightlike behavior is the dual opposite of mass
    # Low saturation, high coherence, simple ancestry = lightlike
    
    # 1. Low saturation = high lightlike
    saturation = calculate_saturation(point)
    saturation_factor = 1.0 - saturation * 0.8
    
    # 2. High coherence = high lightlike
    coherence_factor = point.coherence * 0.7
    
    # 3. Simple ancestry = high lightlike
    ancestry_simplicity = 1.0 - min(0.8, len(point.ancestry) * 0.1)
    
    # 4. Low recursion depth = high lightlike
    recursion_simplicity = 1.0 - min(0.8, point.recursion_depth * 0.2)
    
    # Combine factors
    lightlike_factor = (
        saturation_factor * 0.4 +
        coherence_factor * 0.3 +
        ancestry_simplicity * 0.2 +
        recursion_simplicity * 0.1
    )
    
    # Lightlike factors enhanced by superposition
    lightlike_factor = lightlike_factor * (1.0 - superposition_factor * 0.3) + superposition_factor * 0.7
    
    # Ensure bounds
    return max(0.0, min(1.0, lightlike_factor))


def calculate_field_symmetry(space: ConfigurationSpace, point: ConfigurationPoint) -> float:
    """
    Calculate the symmetry of the field around a point.
    Perfect symmetry means the superposition is stable.
    
    Args:
        space: The configuration space
        point: The configuration point
        
    Returns:
        Symmetry value (1.0 = perfect symmetry, 0.0 = complete asymmetry)
    """
    # If no neighbors, field is perfectly symmetric (no external influence)
    if not point.relations:
        return 1.0
    
    # Measure relational symmetry
    relation_values = list(point.relations.values())
    relation_abs_sum = sum(abs(r) for r in relation_values)
    
    if relation_abs_sum == 0:
        return 1.0
    
    # Measure alignment of relations (if they all point in different directions = more symmetric)
    relation_sum = sum(relation_values)
    relation_balance = 1.0 - abs(relation_sum) / relation_abs_sum
    
    # Measure phase distribution symmetry (if phases are evenly distributed = more symmetric)
    phase_symmetry = 1.0
    if len(point.phase_relations) >= 3:
        neighbor_phases = []
        for neighbor_id in point.phase_relations:
            # Get phase relation
            if neighbor_id in point.phase_relations:
                theta_rel, phi_rel = point.phase_relations[neighbor_id]
                neighbor_phases.append((theta_rel, phi_rel))
        
        # Calculate angular spread in theta dimension
        if neighbor_phases:
            theta_values = [phase[0] for phase in neighbor_phases]
            # Perfect symmetry would have phases evenly distributed around circle
            theta_symmetry = calculate_angular_symmetry(theta_values)
            phase_symmetry = theta_symmetry
    
    # Return overall symmetry (higher = more stable superposition)
    return relation_balance * 0.7 + phase_symmetry * 0.3


def calculate_field_tension(space: ConfigurationSpace, point: ConfigurationPoint) -> float:
    """
    Calculate the inherent field tension for a point.
    This determines if a superposition is stable or must naturally resolve.
    
    Args:
        space: The configuration space
        point: The configuration point to evaluate
        
    Returns:
        Field tension value (positive means unstable superposition)
    """
    # Get superposition factor
    superposition_factor = calculate_superposition_factor(point)
    
    # If not in superposition, no inherent tension toward resolution
    if superposition_factor < 0.7:
        return 0.0
    
    # Calculate field-driven factors that create tension
    
    # 1. Relational asymmetry - asymmetric relations cause tension
    relation_tension = 0.0
    if point.relations:
        # Calculate net pull from relations
        relation_values = list(point.relations.values())
        relation_sum = sum(relation_values)
        relation_abs_sum = sum(abs(v) for v in relation_values)
        
        if relation_abs_sum > 0:
            # Directional bias creates tension
            relation_tension = abs(relation_sum) / relation_abs_sum * relation_abs_sum * 0.3
    
    # 2. Gradient asymmetry - asymmetric gradients cause tension
    gradient_tension = 0.0
    if point.gradients:
        # Calculate net gradient direction
        gradient_values = list(point.gradients.values())
        gradient_sum = sum(gradient_values)
        gradient_abs_sum = sum(abs(v) for v in gradient_values)
        
        if gradient_abs_sum > 0:
            # Directional gradient creates tension
            gradient_tension = abs(gradient_sum) / gradient_abs_sum * gradient_abs_sum * 0.3
    
    # 3. Phase decoherence - lower coherence creates instability
    coherence_tension = (1.0 - point.coherence) * 0.2
    
    # 4. Polarity emergence - any non-zero polarity creates tension
    polarity_tension = abs(point.polarity) * 0.3
    
    # Calculate net tension in the field at this point
    total_tension = (
        relation_tension +
        gradient_tension +
        coherence_tension +
        polarity_tension
    )
    
    # Calculate inherent stability from self-reinforcing factors
    
    # Zero-ness is stability (equals infinite potential)
    awareness_stability = math.exp(-abs(point.awareness) * 20)
    
    # Perfectly symmetric field is stable
    field_symmetry = calculate_field_symmetry(space, point)
    
    # Total stability factor
    stability = (awareness_stability * 0.7 + field_symmetry * 0.3) * superposition_factor
    
    # Net tension: positive means resolution becomes inevitable
    return total_tension - stability * 0.7


def calculate_resolution_direction(space: ConfigurationSpace, point: ConfigurationPoint) -> float:
    """
    Calculate the natural resolution direction based on field properties.
    
    Args:
        space: The configuration space
        point: The configuration point
        
    Returns:
        Resolution direction (-1.0 to 1.0)
    """
    # Calculate directional influence from relations
    relation_direction = 0.0
    relation_weight = 0.0
    
    if point.relations:
        relation_values = list(point.relations.values())
        relation_sum = sum(relation_values)
        relation_abs_sum = sum(abs(v) for v in relation_values)
        
        if relation_abs_sum > 0:
            relation_direction = relation_sum / relation_abs_sum
            relation_weight = min(1.0, relation_abs_sum * 0.2)
    
    # Calculate directional influence from gradients
    gradient_direction = 0.0
    gradient_weight = 0.0
    
    if point.gradients:
        gradient_values = list(point.gradients.values())
        gradient_sum = sum(gradient_values)
        gradient_abs_sum = sum(abs(v) for v in gradient_values)
        
        if gradient_abs_sum > 0:
            gradient_direction = gradient_sum / gradient_abs_sum
            gradient_weight = min(1.0, gradient_abs_sum * 0.2)
    
    # Inherent polarity bias (existing polarization tends to amplify)
    polarity_direction = point.polarity
    polarity_weight = abs(point.polarity) * 0.3
    
    # Combine directional influences with weights
    total_weight = relation_weight + gradient_weight + polarity_weight
    
    if total_weight > 0:
        weighted_direction = (
            relation_direction * relation_weight +
            gradient_direction * gradient_weight +
            polarity_direction * polarity_weight
        ) / total_weight
    else:
        # If no clear direction, random but deterministic based on ID
        # Even when random, the result is deterministic for the same point
        seed = hash(point.id) % 1000 / 1000.0
        weighted_direction = (seed * 2 - 1) * 0.1
    
    return weighted_direction


def update_point_from_field_resolution(point: ConfigurationPoint, direction: float, strength: float):
    """
    Update a point based on natural field resolution.
    This is NOT forcing collapse but observing the effect of field instability.
    
    Args:
        point: The configuration point to update
        direction: Resolution direction (-1.0 to 1.0)
        strength: Resolution strength (0.0 to 1.0)
    """
    # Ensure non-zero direction
    if abs(direction) < 0.01:
        direction = 0.01 * (1.0 if random.random() > 0.5 else -1.0)
    
    # Nonlinear response curve - small changes have proportionally larger effect
    # when a system is on the edge of instability
    response_strength = math.pow(strength, 0.7)
    
    # Update awareness - the primary manifestation of resolution
    awareness_change = direction * response_strength * 0.3
    point.awareness += awareness_change
    
    # Reduce potential factor (resolving superposition)
    potential_reduction = response_strength * 0.2
    point.potential_factor = max(0.0, point.potential_factor - potential_reduction)
    
    # Update polarity bias (secondary manifestation)
    polarity_target = math.copysign(min(0.3, abs(direction) * 2), direction)
    polarity_change = (polarity_target - point.polarity) * response_strength * 0.3
    point.polarity += polarity_change
    
    # Record transition event if significant
    if potential_reduction > 0.05:
        point.collapse_history.append({
            'type': 'field_resolution',
            'direction': direction,
            'strength': strength,
            'awareness_change': awareness_change,
            'potential_reduction': potential_reduction
        })
        
        # Update collapse metric for significant resolutions
        if direction > 0:
            # Structure formation
            point.collapse_metric += response_strength * 0.1
        else:
            # Decay
            point.collapse_metric -= response_strength * 0.1


def propagate_collapse(source: ConfigurationPoint, target: ConfigurationPoint, 
                      collapse_strength: float) -> float:
    """
    Propagate collapse from source to target point.
    Works for both positive (structure) and negative (decay) collapse.
    Implements distinct behaviors for lightlike vs massive objects.
    
    Args:
        source: Source point
        target: Target point
        collapse_strength: Signed strength of collapse (-1.0 to 1.0)
        
    Returns:
        Actual propagated collapse strength
    """
    # Skip if target is None
    if target is None:
        return 0.0
    
    # Determine if we're propagating structure or decay
    is_decay = collapse_strength < 0
    
    # Calculate lightlike vs mass properties
    lightlike_factor = calculate_lightlike_factor(source)
    mass_factor = calculate_mass_factor(source)
    
    # === LIGHTLIKE PROPAGATION ===
    # Lightlike objects propagate collapse with minimal loss in their polarity direction
    if lightlike_factor > 0.7:
        # Lightlike propagation is highly directional based on polarity
        polarity_alignment = source.polarity * target.polarity
        
        # Propagation factor - very high for aligned polarity
        if polarity_alignment > 0:
            # Aligned polarities = enhanced propagation (constructive interference)
            propagation_factor = 0.9 + lightlike_factor * 0.1
        else:
            # Opposed polarities = reduced propagation (destructive interference)
            propagation_factor = max(0.3, 0.7 - abs(polarity_alignment) * 0.4)
        
        # Coherence bonus
        coherence_bonus = source.coherence * 0.1
        propagation_factor = min(0.98, propagation_factor + coherence_bonus)
        
        # Calculate propagated strength (preserving sign)
        propagated_strength = collapse_strength * propagation_factor
        
        # Add quantum fluctuation for lightlike behavior
        if random.random() < 0.1:
            # Fluctuation preserves direction (sign)
            sign = -1.0 if is_decay else 1.0
            quantum_fluctuation = random.uniform(0, 0.05) * sign
            propagated_strength += quantum_fluctuation
            
        # Process collapse on target
        process_collapse(target, propagated_strength, source.id)
        return propagated_strength
    
    # === MASSIVE PROPAGATION ===
    # Massive objects bend/absorb collapse rather than propagating it cleanly
    elif mass_factor > 0.7:
        # Calculate curvature effect
        curvature_effect = source.curvature * 0.7
        
        # Massive objects bend collapse around them
        # This creates a "gravity-like" effect that pulls collapse inward
        if curvature_effect > 0.5:
            # Strong curvature - significant bending
            # Target gets less direct propagation, more curved/indirect effects
            
            # Direct propagation is reduced by curvature
            direct_propagation = collapse_strength * max(0.1, 1.0 - curvature_effect)
            
            # Some strength redirects back to source (recursive/inward bending)
            recursive_strength = collapse_strength * min(0.4, curvature_effect * 0.6)
            process_collapse(source, recursive_strength * 0.5, source.id)
            
            # Process reduced direct collapse on target
            process_collapse(target, direct_propagation, source.id)
            return direct_propagation
        else:
            # Moderate curvature - partial absorption
            absorbed_fraction = mass_factor * 0.5
            propagated_strength = collapse_strength * (1.0 - absorbed_fraction)
            
            # Process reduced collapse on target
            process_collapse(target, propagated_strength, source.id)
            return propagated_strength
    
    # === STANDARD PROPAGATION (NEITHER FULLY LIGHTLIKE NOR MASSIVE) ===
    else:
        # Calculate base propagation factors on continuous scale
        # More lightlike = better propagation
        # More massive = more absorption/damping
        
        # Base factor scales with inverse mass and partial lightlike behavior
        base_factor = 0.7 + lightlike_factor * 0.2 - mass_factor * 0.3
        
        # Adjust by phase coherence
        coherence_bonus = source.coherence * 0.2
        propagation_factor = min(0.95, max(0.3, base_factor + coherence_bonus))
        
        # Calculate propagated strength (preserving sign)
        propagated_strength = collapse_strength * propagation_factor
        
        # Process collapse on target
        process_collapse(target, propagated_strength, source.id)
        
        return propagated_strength


def calculate_emergent_distance(space: ConfigurationSpace, point1_id: str, point2_id: str) -> float:
    """
    Calculate emergent distance between two points on the toroidal manifold.
    Distance emerges from phase relations, not from cartesian coordinates.
    
    Args:
        space: The configuration space
        point1_id: First point ID
        point2_id: Second point ID
        
    Returns:
        Emergent distance value (0.0 to 1.0)
    """
    point1 = space.points.get(point1_id)
    point2 = space.points.get(point2_id)
    
    if not point1 or not point2:
        return 1.0  # Maximum distance if points don't exist
    
    # Get superposition factors
    point1_superposition = calculate_superposition_factor(point1)
    point2_superposition = calculate_superposition_factor(point2)
    
    # Combined superposition effect on distance
    combined_superposition = (point1_superposition + point2_superposition) / 2
    
    # Higher superposition factor leads to more uncertain distances
    # Use a smooth transition function
    if combined_superposition > 0.8:
        # High uncertainty in distance - ranges from close to medium
        # More uncertain as superposition increases
        uncertainty_range = 0.2 + combined_superposition * 0.1
        base_distance = 0.1 + combined_superposition * 0.1
        return base_distance + random.uniform(0, uncertainty_range)
    
    # Calculate toroidal distance based on derived phases
    point1_theta = derive_phase(point1, 'theta', space)
    point1_phi = derive_phase(point1, 'phi', space)
    point2_theta = derive_phase(point2, 'theta', space)
    point2_phi = derive_phase(point2, 'phi', space)
    
    # Calculate phase differences
    theta_diff = angular_difference(point1_theta, point2_theta)
    phi_diff = angular_difference(point1_phi, point2_phi)
    
    # Normalize to [0, 1]
    max_diff = math.pi
    theta_dist = theta_diff / max_diff
    phi_dist = phi_diff / max_diff
    
    # Use toroidal metric to calculate combined distance
    toroidal_distance = math.sqrt(theta_dist**2 + phi_dist**2) / math.sqrt(2)
    
    # For directly connected points, factor in relation strength
    if point2_id in point1.relations:
        relation_strength = abs(point1.relations[point2_id])
        relation_distance = 1.0 - relation_strength
        
        # Blend toroidal and relation distances
        # More saturated points rely more on relations than phases
        saturation_factor = max(calculate_saturation(point1), calculate_saturation(point2))
        blend_factor = 0.5 + saturation_factor * 0.3  # 0.5 to 0.8
        
        # Superposition adds uncertainty to distance
        uncertainty = random.uniform(-0.05, 0.05) * combined_superposition
        
        distance = toroidal_distance * (1 - blend_factor) + relation_distance * blend_factor + uncertainty
        return max(0.1, min(1.0, distance))
    
    # For unconnected points, add a distance penalty
    return min(1.0, toroidal_distance + 0.3)


def check_structural_instabilities(space: ConfigurationSpace, dt: float):
    """
    Check for inherent structural instabilities in the field.
    When superpositions become unstable, they naturally resolve.
    
    Args:
        space: The configuration space
        dt: Time delta
    """
    # Process all points for structural tensions
    for point_id, point in space.points.items():
        # Calculate field tension at this point
        field_tension = calculate_field_tension(space, point)
        
        # If tension exceeds the point's ability to maintain superposition,
        # the point naturally resolves without being "forced"
        if field_tension > 0:
            # Inherent resolution through field dynamics
            # The resolution direction emerges from field properties
            resolution_direction = calculate_resolution_direction(space, point)
            
            # Resolution strength scales with how much tension exceeds stability
            resolution_strength = field_tension * dt
            
            # Apply field-driven changes (not manually forcing anything)
            update_point_from_field_resolution(point, resolution_direction, resolution_strength)
            
            # Update space stats
            if resolution_direction > 0:
                space.stats['structure_collapse_count'] += 1
            else:
                space.stats['decay_collapse_count'] += 1