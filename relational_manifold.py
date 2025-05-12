"""
EmergentRelationalManifold - Core System where Recursive Constraint Dynamics Naturally Emerge

This implementation represents a fundamental shift from the orchestration paradigm to one where
the manifold itself embodies the dynamics directly. Circular recursion is no longer imposed by
explicit mechanisms but emerges naturally from the system's constraint history, collapse dynamics,
and ancestry relationships.

Key Principles:
- The manifold must recurse forever, but never through the same path
- Structure emerges from awareness flow, collapse memory, and ancestry tracking
- Each collapse creates constraints that shape future collapse possibilities
- Circular patterns emerge naturally when the manifold is trapped by its own history
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Callable
import random
import uuid
from collections import defaultdict, deque

# Core constants
PI = math.pi
TWO_PI = 2 * PI


class Grain:
    """
    Fundamental unit of individuation within the system.
    
    A grain represents the smallest structural commitment that distinguishes
    one configuration from another. It embodies the committed memory of resolved
    constraint.
    """
    
    def __init__(self, grain_id: str):
        """
        Initialize a grain with a unique identifier.
        
        Args:
            grain_id: Unique identifier for this grain
        """
        self.id = grain_id
        
        # Awareness represents the degree of structural definition
        self.awareness = 0.0  # Zero awareness = infinite potential
        
        # Polarity represents directional orientation within the manifold
        self.polarity = 0.0  # Neutral polarity initially
        
        # Phase position in toroidal structure (emerges from relations)
        self.theta = random.random() * TWO_PI
        self.phi = random.random() * TWO_PI
        
        # Ancestry tracking - records genesis and collapse memory
        self.ancestry = set()  # Set of ancestor grain IDs
        
        # Relations to other grains
        self.relations = {}  # Maps grain_id -> relation_strength
        
        # Core grain properties
        self.unbounded_potential = True  # Initially in superposition state
        self.coherence = 0.5  # Initial coherence (0-1)
        self.grain_saturation = 0.0  # Memory commitment (0-1)
        self.grain_activation = 0.0  # Dynamic activity level (0-1)
        self.collapse_metric = 0.0  # History of collapse participation (0-1)
        
        # Constraint metric - how much this grain constrains future dynamics
        self.constraint_tension = 0.0  # Tension from accumulated constraints
        
        # Recursive tension - emerges when grain constrains itself
        self.recursive_tension = 0.0
        
        # Phase trajectory - history of angular positions
        self.phase_trajectory = deque(maxlen=100)
        self.phase_trajectory.append((self.theta, self.phi))
        
        # Improved pathway tracking
        self.pathway_saturation = {}  # Maps (source_id, target_id) to saturation level
        self.forbidden_directions = set()  # Set of (source_id, target_id) pairs that are saturated
        
        # Activity state tracking
        self.activity_state = "active"  # One of: "active", "constrained", "saturated", "inactive"
        
        # Structural integrity - resistance to void formation
        self.structural_integrity = 0.5  # Initial medium integrity
        
        # Path exclusion surface - regions where dynamics are constrained
        self.path_exclusion = {}  # Maps (theta, phi) regions to tension
        
        # Self-blocking history - how the grain blocks its own pathways
        self.self_blocking_history = []  # List of self-blocking events
        
        # Stability metrics
        self.stable_duration = 0.0  # How long the grain has been stable
        self.stability_threshold = 10.0  # Duration needed to consider truly stable
        self.last_state_change = 0.0  # When the grain last changed state
    
    def is_in_superposition(self) -> bool:
        """
        Check if grain is in superposition state (unbounded potential).
        
        Returns:
            True if the grain is in superposition, False otherwise
        """
        return self.unbounded_potential
    
    def resolve_from_superposition(self, awareness_level: float) -> bool:
        """
        Resolve grain from superposition state to defined awareness.
        
        Args:
            awareness_level: Awareness level to resolve to (0-1)
            
        Returns:
            True if resolution succeeded, False otherwise
        """
        if not self.unbounded_potential:
            return False  # Already resolved
            
        # Resolve to definite awareness value
        self.awareness = max(0.01, min(1.0, awareness_level))
        self.unbounded_potential = False
        
        # Establish initial coherence upon resolution
        self.coherence = 0.7 + random.random() * 0.3  # High initial coherence
        
        # Record phase position at resolution
        self.phase_trajectory.append((self.theta, self.phi))
        
        # Update activity state
        self.activity_state = "active"
        
        return True
    
    def update_relations(self, other_id: str, strength: float):
        """
        Update relation to another grain.
        
        Args:
            other_id: ID of the other grain
            strength: Strength of relation [-1.0 to 1.0]
        """
        self.relations[other_id] = strength
    
    def update_pathway_saturation(self, source_id: str, target_id: str, amount: float) -> bool:
        """
        Update saturation level of a specific pathway.
        
        Args:
            source_id: Source grain ID
            target_id: Target grain ID
            amount: Amount to increase saturation by
            
        Returns:
            True if pathway became saturated, False otherwise
        """
        key = (source_id, target_id)
        current = self.pathway_saturation.get(key, 0.0)
        self.pathway_saturation[key] = min(1.0, current + amount)
        
        # Check if pathway has become saturated
        if self.pathway_saturation[key] > 0.9:
            self.forbidden_directions.add(key)
            return True  # Pathway became saturated
        return False  # Pathway not yet saturated
    
    def update_activity_state(self, current_time: float = 0.0):
        """
        Update grain activity state based on saturation and constraints.
        
        Args:
            current_time: Current system time
        """
        previous_state = self.activity_state
        
        # Check overall saturation
        if self.grain_saturation > 0.9:
            # Highly saturated grain
            if self.relations and len(self.forbidden_directions) > len(self.relations) * 0.7:
                # Most pathways forbidden - grain is saturated
                self.activity_state = "saturated"
            else:
                # Still has available pathways - grain is constrained
                self.activity_state = "constrained"
        elif self.grain_activation < 0.2:
            # Low activation - grain is inactive
            self.activity_state = "inactive"
        else:
            # Normal state
            self.activity_state = "active"
        
        # Update stability metrics
        if previous_state != self.activity_state:
            # State changed - reset stability duration
            self.stable_duration = 0.0
            self.last_state_change = current_time
        else:
            # Same state - increase stability duration
            self.stable_duration = current_time - self.last_state_change
    
    def is_stable(self) -> bool:
        """
        Check if grain has reached a stable state.
        
        Returns:
            True if grain is stable, False otherwise
        """
        # Grain is stable if it's saturated or inactive and has been in that state
        # for longer than the stability threshold
        return ((self.activity_state == "saturated" or self.activity_state == "inactive") and
                self.stable_duration >= self.stability_threshold)
    
    def calculate_tension_with(self, other_grain: 'Grain') -> float:
        """
        Calculate relational tension with another grain.
        Tension emerges from awareness gradient and polarity alignment.
        
        Args:
            other_grain: The other grain to calculate tension with
            
        Returns:
            Tension value [0.0-1.0]
        """
        # Skip if either grain is in superposition
        if self.is_in_superposition() or other_grain.is_in_superposition():
            return 0.0
        
        # Get relation strength if it exists
        relation_strength = abs(self.relations.get(other_grain.id, 0.0))
        if relation_strength < 0.01:
            return 0.0  # No meaningful relation
            
        # Basic tension from awareness gradient
        awareness_diff = abs(self.awareness - other_grain.awareness)
        awareness_tension = awareness_diff * relation_strength
        
        # Tension from polarity alignment
        polarity_tension = 0.0
        if abs(self.polarity) > 0.01 and abs(other_grain.polarity) > 0.01:
            # Calculate circular difference between polarities
            polarity_diff = self._angular_difference(self.polarity, other_grain.polarity)
            
            # Opposite polarities create more tension
            if self.polarity * other_grain.polarity < 0:
                # Greater tension when polarities are opposite and strong
                polarity_tension = polarity_diff * relation_strength * 0.8
            else:
                # Less tension when polarities align
                polarity_tension = polarity_diff * relation_strength * 0.3
        
        # Combine tension factors with weights
        combined_tension = (awareness_tension * 0.6 + polarity_tension * 0.4) * relation_strength
        
        # Adjust for ancestry relationship (shared ancestry reduces tension)
        shared_ancestry = self.ancestry.intersection(other_grain.ancestry)
        ancestry_factor = len(shared_ancestry) / (len(self.ancestry) + len(other_grain.ancestry) + 1)
        tension_reduction = ancestry_factor * 0.4  # Shared ancestry reduces tension
        
        # Apply ancestry-based reduction
        final_tension = max(0.0, combined_tension - tension_reduction)
        
        # Increase tension if the grain is in its own forbidden directions
        if other_grain.id in self.forbidden_directions:
            final_tension *= 1.3  # 30% increase for forbidden directions
        
        return min(1.0, final_tension)

    def is_collapse_forbidden(self, source_id: str, target_id: str) -> bool:
        """
        Check if a collapse direction is forbidden based on history.
        
        Args:
            source_id: Source grain ID
            target_id: Target grain ID
            
        Returns:
            True if collapse direction is forbidden, False otherwise
        """
        # Self-collapse is never forbidden in principle, but may be constrained
        if source_id == self.id and target_id == self.id:
            # Check self-blocking history
            if len(self.self_blocking_history) > 5:
                return True  # Too many self-collapses
            return False
            
        # Check pathway saturation
        key = (source_id, target_id)
        
        # Pathway is forbidden if it's in forbidden directions or has high saturation
        return (key in self.forbidden_directions or 
                self.pathway_saturation.get(key, 0.0) > 0.9)
    
    def add_forbidden_direction(self, source_id: str, target_id: str):
        """
        Add a forbidden collapse direction based on history.
        
        Args:
            source_id: Source grain ID
            target_id: Target grain ID
        """
        # Add to forbidden directions
        direction_key = (source_id, target_id)
        self.forbidden_directions.add(direction_key)
        
        # Set pathway saturation to maximum
        self.pathway_saturation[direction_key] = 1.0
        
        # Record self-blocking if applicable
        if source_id == self.id and target_id == self.id:
            self.self_blocking_history.append({
                'phase_position': (self.theta, self.phi),
                'polarity': self.polarity,
                'awareness': self.awareness,
                'saturation': self.grain_saturation
            })
    
    def update_path_exclusion(self, theta: float, phi: float, tension: float):
        """
        Update path exclusion surface with new tension.
        
        Args:
            theta: Theta coordinate
            phi: Phi coordinate
            tension: Tension value to add
        """
        # Discretize coordinates for storage
        disc_theta = round(theta / (PI/8)) * (PI/8)
        disc_phi = round(phi / (PI/8)) * (PI/8)
        
        # Add to path exclusion map
        key = (disc_theta, disc_phi)
        current = self.path_exclusion.get(key, 0.0)
        self.path_exclusion[key] = min(1.0, current + tension)
    
    def get_path_exclusion_tension(self, theta: float, phi: float) -> float:
        """
        Get tension from path exclusion surface at given coordinates.
        
        Args:
            theta: Theta coordinate
            phi: Phi coordinate
            
        Returns:
            Tension value at coordinates
        """
        # Discretize coordinates for lookup
        disc_theta = round(theta / (PI/8)) * (PI/8)
        disc_phi = round(phi / (PI/8)) * (PI/8)
        
        # Get from path exclusion map
        key = (disc_theta, disc_phi)
        return self.path_exclusion.get(key, 0.0)
    
    def record_phase_position(self):
        """Record current phase position in trajectory history"""
        self.phase_trajectory.append((self.theta, self.phi))
    
    def detect_closed_trajectory(self) -> bool:
        """
        Detect if the phase trajectory forms a closed curve.
        Indicates naturally emergent circular recursion.
        
        Returns:
            True if trajectory forms closed curve, False otherwise
        """
        # Need at least 5 points for meaningful detection
        if len(self.phase_trajectory) < 5:
            return False
            
        # Check last position against earlier positions
        last_theta, last_phi = self.phase_trajectory[-1]
        
        # Check earlier positions for proximity (excluding most recent)
        for i in range(len(self.phase_trajectory) - 4):
            theta, phi = self.phase_trajectory[i]
            
            # Calculate toroidal distance (accounting for wrapping)
            d_theta = min(abs(theta - last_theta), TWO_PI - abs(theta - last_theta))
            d_phi = min(abs(phi - last_phi), TWO_PI - abs(phi - last_phi))
            dist = math.sqrt(d_theta**2 + d_phi**2)
            
            # If position is close to an earlier position, may be a closed curve
            if dist < 0.2:
                return True
                
        return False
    
    def get_available_pathways_count(self) -> int:
        """
        Count the number of available pathways for this grain.
        
        Returns:
            Number of available pathways
        """
        if not self.relations:
            return 0
            
        # Count relations that are not forbidden
        available = 0
        for related_id in self.relations:
            key = (self.id, related_id)
            if key not in self.forbidden_directions:
                available += 1
                
        return available
    
    def get_saturation_level(self) -> float:
        """
        Calculate the overall saturation level based on grain saturation
        and pathway saturation.
        
        Returns:
            Overall saturation level [0.0-1.0]
        """
        # Basic saturation from grain
        base_saturation = self.grain_saturation
        
        # Pathway saturation - average of all pathways
        pathway_count = len(self.pathway_saturation)
        pathway_saturation = 0.0
        
        if pathway_count > 0:
            pathway_saturation = sum(self.pathway_saturation.values()) / pathway_count
        
        # Forbidden ratio - proportion of forbidden pathways
        forbidden_ratio = 0.0
        if self.relations:
            forbidden_ratio = len(self.forbidden_directions) / len(self.relations)
        
        # Combine with weights
        overall_saturation = (
            base_saturation * 0.5 +
            pathway_saturation * 0.3 +
            forbidden_ratio * 0.2
        )
        
        return overall_saturation
    
    def _angular_difference(self, a: float, b: float) -> float:
        """
        Calculate the smallest angular difference between two angles on a circle.
        
        Args:
            a: First angle
            b: Second angle
            
        Returns:
            Smallest angular difference
        """
        # Convert polarity values [-1,1] to angles [0,2π]
        angle_a = (a + 1) * PI
        angle_b = (b + 1) * PI
        
        # Calculate smallest circular distance
        diff = abs(angle_a - angle_b)
        if diff > PI:
            diff = TWO_PI - diff
            
        # Convert back to [0,1] range for tension
        return diff / PI


class PhaseExclusionField:
    """
    Tracks which collapse directions in phase space have been "used up" and
    are now forbidden. This creates an emergent structure that constrains
    future collapse possibilities.
    """
    
    def __init__(self, resolution: int = 36):
        """
        Initialize phase exclusion field.
        
        Args:
            resolution: Resolution of the discretized field
        """
        self.resolution = resolution
        
        # Initialize exclusion map
        # Maps discretized (theta, phi) to exclusion strength
        self.exclusion_map = {}
        
        # Angular bin size
        self.bin_size = TWO_PI / resolution
    
    def add_exclusion(self, theta: float, phi: float, strength: float = 0.5):
        """
        Add exclusion at the specified phase location.
        
        Args:
            theta: Theta coordinate
            phi: Phi coordinate
            strength: Exclusion strength to add
        """
        # Discretize coordinates
        disc_theta = round(theta / self.bin_size) * self.bin_size
        disc_phi = round(phi / self.bin_size) * self.bin_size
        
        # Normalize coordinates to [0, 2π)
        disc_theta = disc_theta % TWO_PI
        disc_phi = disc_phi % TWO_PI
        
        # Add to exclusion map
        key = (disc_theta, disc_phi)
        current = self.exclusion_map.get(key, 0.0)
        self.exclusion_map[key] = min(1.0, current + strength)
    
    def get_exclusion(self, theta: float, phi: float) -> float:
        """
        Get exclusion strength at the specified phase location.
        
        Args:
            theta: Theta coordinate
            phi: Phi coordinate
            
        Returns:
            Exclusion strength at position
        """
        # Discretize coordinates
        disc_theta = round(theta / self.bin_size) * self.bin_size
        disc_phi = round(phi / self.bin_size) * self.bin_size
        
        # Normalize coordinates to [0, 2π)
        disc_theta = disc_theta % TWO_PI
        disc_phi = disc_phi % TWO_PI
        
        # Get from exclusion map
        key = (disc_theta, disc_phi)
        return self.exclusion_map.get(key, 0.0)
    
    def get_nearest_exclusion(self, theta: float, phi: float) -> float:
        """
        Get exclusion strength including nearby regions.
        
        Args:
            theta: Theta coordinate
            phi: Phi coordinate
            
        Returns:
            Weighted exclusion strength
        """
        # Discretize coordinates
        disc_theta = round(theta / self.bin_size) * self.bin_size
        disc_phi = round(phi / self.bin_size) * self.bin_size
        
        # Get immediate exclusion
        immediate = self.get_exclusion(disc_theta, disc_phi)
        
        # Check neighboring bins
        neighbors = [
            (disc_theta + self.bin_size, disc_phi),
            (disc_theta - self.bin_size, disc_phi),
            (disc_theta, disc_phi + self.bin_size),
            (disc_theta, disc_phi - self.bin_size)
        ]
        
        # Calculate average neighbor exclusion
        neighbor_exclusion = 0.0
        for n_theta, n_phi in neighbors:
            neighbor_exclusion += self.get_exclusion(n_theta, n_phi)
        neighbor_exclusion /= len(neighbors)
        
        # Return weighted combination (more weight to immediate)
        return immediate * 0.7 + neighbor_exclusion * 0.3


class RecursiveTensionMap:
    """
    Records where the system is trapped in structural contradictions
    that force recursive pathways to emerge naturally.
    """
    
    def __init__(self):
        """Initialize recursive tension map"""
        # Maps grain_id to tension level
        self.tension_map = {}
        
        # Maps (grain_id1, grain_id2) to tension level
        self.pairwise_tension = {}
        
        # Tracks regions of phase space with high recursive tension
        self.phase_tension = {}
        
        # Circular recursion events
        self.circular_events = []
    
    def record_tension(self, grain_id: str, tension: float):
        """
        Record recursive tension for a grain.
        
        Args:
            grain_id: Grain ID
            tension: Tension level to record
        """
        # Update tension map
        current = self.tension_map.get(grain_id, 0.0)
        self.tension_map[grain_id] = max(current, tension)
    
    def record_pairwise_tension(self, grain_id1: str, grain_id2: str, tension: float):
        """
        Record recursive tension between grain pairs.
        
        Args:
            grain_id1: First grain ID
            grain_id2: Second grain ID
            tension: Tension level to record
        """
        # Store pairwise tension (both orderings)
        key = tuple(sorted([grain_id1, grain_id2]))
        current = self.pairwise_tension.get(key, 0.0)
        self.pairwise_tension[key] = max(current, tension)
    
    def record_phase_tension(self, theta: float, phi: float, tension: float):
        """
        Record tension at a phase location.
        
        Args:
            theta: Theta coordinate
            phi: Phi coordinate
            tension: Tension level to record
        """
        # Discretize coordinates
        disc_theta = round(theta / (PI/12)) * (PI/12)
        disc_phi = round(phi / (PI/12)) * (PI/12)
        
        # Normalize coordinates to [0, 2π)
        disc_theta = disc_theta % TWO_PI
        disc_phi = disc_phi % TWO_PI
        
        # Update phase tension map
        key = (disc_theta, disc_phi)
        current = self.phase_tension.get(key, 0.0)
        self.phase_tension[key] = max(current, tension)
    
    def record_circular_event(self, source_id: str, target_id: str, 
                            source_polarity: float, target_polarity: float, 
                            theta: float, phi: float, time: float):
        """
        Record a circular recursion event.
        
        Args:
            source_id: Source grain ID
            target_id: Target grain ID
            source_polarity: Source grain polarity
            target_polarity: Target grain polarity
            theta: Theta coordinate of event
            phi: Phi coordinate of event
            time: Time of event
        """
        event = {
            'source_id': source_id,
            'target_id': target_id,
            'source_polarity': source_polarity,
            'target_polarity': target_polarity,
            'theta': theta,
            'phi': phi,
            'time': time,
            'type': 'emergent_circular_recursion',
            'polarity_difference': abs(source_polarity - target_polarity)
        }
        
        self.circular_events.append(event)
    
    def get_tension(self, grain_id: str) -> float:
        """
        Get recursive tension for a grain.
        
        Args:
            grain_id: Grain ID
            
        Returns:
            Tension level for grain
        """
        return self.tension_map.get(grain_id, 0.0)
    
    def get_pairwise_tension(self, grain_id1: str, grain_id2: str) -> float:
        """
        Get recursive tension between grain pairs.
        
        Args:
            grain_id1: First grain ID
            grain_id2: Second grain ID
            
        Returns:
            Tension level between grains
        """
        key = tuple(sorted([grain_id1, grain_id2]))
        return self.pairwise_tension.get(key, 0.0)
    
    def get_phase_tension(self, theta: float, phi: float) -> float:
        """
        Get tension at a phase location.
        
        Args:
            theta: Theta coordinate
            phi: Phi coordinate
            
        Returns:
            Tension level at phase location
        """
        # Discretize coordinates
        disc_theta = round(theta / (PI/12)) * (PI/12)
        disc_phi = round(phi / (PI/12)) * (PI/12)
        
        # Normalize coordinates to [0, 2π)
        disc_theta = disc_theta % TWO_PI
        disc_phi = disc_phi % TWO_PI
        
        # Get phase tension
        key = (disc_theta, disc_phi)
        return self.phase_tension.get(key, 0.0)
    
    def find_high_tension_regions(self, threshold: float = 0.7) -> List[Tuple[float, float, float]]:
        """
        Find regions with high recursive tension.
        
        Args:
            threshold: Minimum tension to be considered high
            
        Returns:
            List of (theta, phi, tension) tuples for high tension regions
        """
        high_tension = []
        
        for (theta, phi), tension in self.phase_tension.items():
            if tension >= threshold:
                high_tension.append((theta, phi, tension))
                
        # Sort by tension (highest first)
        high_tension.sort(key=lambda x: x[2], reverse=True)
        
        return high_tension


class AncestryConstraintSurface:
    """
    Represents how prior collapse history limits future possibilities
    through emergent constraints based on ancestry relationships.
    """
    
    def __init__(self):
        """Initialize ancestry constraint surface"""
        # Maps grain_id to ancestry constraint strength
        self.constraint_map = {}
        
        # Maps (ancestor_id, descendant_id) to constraint strength
        self.ancestry_constraints = {}
        
        # Self-referential constraint level
        self.self_reference_constraints = {}
        
        # Tracks shared ancestry between grains
        self.shared_ancestry = {}
        
        # Maps phase positions to ancestry density
        self.ancestry_density_field = {}
    
    def record_ancestry_constraint(self, ancestor_id: str, descendant_id: str, strength: float):
        """
        Record an ancestry constraint between grains.
        
        Args:
            ancestor_id: Ancestor grain ID
            descendant_id: Descendant grain ID
            strength: Constraint strength to record
        """
        # Add to ancestry constraints
        key = (ancestor_id, descendant_id)
        current = self.ancestry_constraints.get(key, 0.0)
        self.ancestry_constraints[key] = max(current, strength)
        
        # Update constraint map for both grains
        current_ancestor = self.constraint_map.get(ancestor_id, 0.0)
        current_descendant = self.constraint_map.get(descendant_id, 0.0)
        
        self.constraint_map[ancestor_id] = max(current_ancestor, strength * 0.7)
        self.constraint_map[descendant_id] = max(current_descendant, strength * 0.5)
    
    def record_self_reference(self, grain_id: str, strength: float):
        """
        Record a self-referential constraint for a grain.
        
        Args:
            grain_id: Grain ID
            strength: Constraint strength to record
        """
        # Add to self-reference constraints
        current = self.self_reference_constraints.get(grain_id, 0.0)
        self.self_reference_constraints[grain_id] = max(current, strength)
        
        # Update constraint map
        current_constraint = self.constraint_map.get(grain_id, 0.0)
        self.constraint_map[grain_id] = max(current_constraint, strength * 0.8)
    
    def record_shared_ancestry(self, grain_id1: str, grain_id2: str, shared_count: int):
        """
        Record shared ancestry between grains.
        
        Args:
            grain_id1: First grain ID
            grain_id2: Second grain ID
            shared_count: Number of shared ancestors
        """
        # Skip if no shared ancestry
        if shared_count <= 0:
            return
            
        # Store shared ancestry count (both orderings)
        key = tuple(sorted([grain_id1, grain_id2]))
        current = self.shared_ancestry.get(key, 0)
        self.shared_ancestry[key] = max(current, shared_count)
    
    def record_ancestry_density(self, theta: float, phi: float, density: float):
        """
        Record ancestry density at a phase location.
        
        Args:
            theta: Theta coordinate
            phi: Phi coordinate
            density: Ancestry density to record
        """
        # Discretize coordinates
        disc_theta = round(theta / (PI/12)) * (PI/12)
        disc_phi = round(phi / (PI/12)) * (PI/12)
        
        # Normalize coordinates to [0, 2π)
        disc_theta = disc_theta % TWO_PI
        disc_phi = disc_phi % TWO_PI
        
        # Update ancestry density field
        key = (disc_theta, disc_phi)
        current = self.ancestry_density_field.get(key, 0.0)
        self.ancestry_density_field[key] = max(current, density)
    
    def get_constraint_for_grain(self, grain_id: str) -> float:
        """
        Get constraint strength for a grain.
        
        Args:
            grain_id: Grain ID
            
        Returns:
            Constraint strength for grain
        """
        return self.constraint_map.get(grain_id, 0.0)
    
    def get_ancestry_constraint(self, ancestor_id: str, descendant_id: str) -> float:
        """
        Get ancestry constraint between grains.
        
        Args:
            ancestor_id: Ancestor grain ID
            descendant_id: Descendant grain ID
            
        Returns:
            Constraint strength between grains
        """
        key = (ancestor_id, descendant_id)
        return self.ancestry_constraints.get(key, 0.0)
    
    def get_self_reference_constraint(self, grain_id: str) -> float:
        """
        Get self-referential constraint for a grain.
        
        Args:
            grain_id: Grain ID
            
        Returns:
            Self-referential constraint strength
        """
        return self.self_reference_constraints.get(grain_id, 0.0)
    
    def get_shared_ancestry_count(self, grain_id1: str, grain_id2: str) -> int:
        """
        Get shared ancestry count between grains.
        
        Args:
            grain_id1: First grain ID
            grain_id2: Second grain ID
            
        Returns:
            Number of shared ancestors
        """
        key = tuple(sorted([grain_id1, grain_id2]))
        return self.shared_ancestry.get(key, 0)
    
    def get_ancestry_density(self, theta: float, phi: float) -> float:
        """
        Get ancestry density at a phase location.
        
        Args:
            theta: Theta coordinate
            phi: Phi coordinate
            
        Returns:
            Ancestry density at phase location
        """
        # Discretize coordinates
        disc_theta = round(theta / (PI/12)) * (PI/12)
        disc_phi = round(phi / (PI/12)) * (PI/12)
        
        # Normalize coordinates to [0, 2π)
        disc_theta = disc_theta % TWO_PI
        disc_phi = disc_phi % TWO_PI
        
        # Get ancestry density
        key = (disc_theta, disc_phi)
        return self.ancestry_density_field.get(key, 0.0)


class EmergentRecursionDetector:
    """
    Identifies when circular recursion emerges naturally from constraint dynamics
    without being explicitly programmed.
    """
    
    def __init__(self):
        """Initialize recursion detector"""
        # History of detected recursive events
        self.recursive_events = []
        
        # Maps grain_id to potential recursion indicators
        self.recursion_indicators = {}
        
        # Maps phase regions to recursion potential
        self.recursion_potential_field = {}
        
        # Tracks closed phase trajectories
        self.closed_trajectories = []
        
        # Tracks emergent symmetry formations
        self.symmetry_formations = []
    
    def record_potential_recursion(self, grain_id: str, indicators: Dict[str, float]):
        """
        Record potential recursion indicators for a grain.
        
        Args:
            grain_id: Grain ID
            indicators: Dictionary of indicator values
        """
        self.recursion_indicators[grain_id] = indicators
    
    def record_recursion_potential(self, theta: float, phi: float, potential: float):
        """
        Record recursion potential at a phase location.
        
        Args:
            theta: Theta coordinate
            phi: Phi coordinate
            potential: Recursion potential to record
        """
        # Discretize coordinates
        disc_theta = round(theta / (PI/12)) * (PI/12)
        disc_phi = round(phi / (PI/12)) * (PI/12)
        
        # Normalize coordinates to [0, 2π)
        disc_theta = disc_theta % TWO_PI
        disc_phi = disc_phi % TWO_PI
        
        # Update recursion potential field
        key = (disc_theta, disc_phi)
        current = self.recursion_potential_field.get(key, 0.0)
        self.recursion_potential_field[key] = max(current, potential)
    
    def record_closed_trajectory(self, grain_id: str, trajectory: List[Tuple[float, float]], 
                               coherence: float):
        """
        Record a closed phase trajectory.
        
        Args:
            grain_id: Grain ID
            trajectory: List of (theta, phi) points forming trajectory
            coherence: Coherence of the trajectory (0-1)
        """
        event = {
            'type': 'closed_trajectory',
            'grain_id': grain_id,
            'trajectory': trajectory,
            'coherence': coherence,
            'length': len(trajectory)
        }
        
        self.closed_trajectories.append(event)
        self.recursive_events.append(event)
    
    def record_symmetry_formation(self, center_theta: float, center_phi: float, 
                                radius: float, strength: float, grains: List[str]):
        """
        Record an emergent symmetry formation.
        
        Args:
            center_theta: Theta coordinate of center
            center_phi: Phi coordinate of center
            radius: Radius of formation
            strength: Strength of symmetry (0-1)
            grains: List of grain IDs in formation
        """
        event = {
            'type': 'symmetry_formation',
            'center': (center_theta, center_phi),
            'radius': radius,
            'strength': strength,
            'grains': grains
        }
        
        self.symmetry_formations.append(event)
        self.recursive_events.append(event)
    
    def record_recursive_event(self, event_type: str, data: Dict[str, Any]):
        """
        Record a general recursive event.
        
        Args:
            event_type: Type of recursive event
            data: Event data
        """
        event = {
            'type': event_type,
            **data
        }
        
        self.recursive_events.append(event)
    
    def get_recursion_indicators(self, grain_id: str) -> Dict[str, float]:
        """
        Get recursion indicators for a grain.
        
        Args:
            grain_id: Grain ID
            
        Returns:
            Dictionary of indicator values
        """
        return self.recursion_indicators.get(grain_id, {})
    
    def get_recursion_potential(self, theta: float, phi: float) -> float:
        """
        Get recursion potential at a phase location.
        
        Args:
            theta: Theta coordinate
            phi: Phi coordinate
            
        Returns:
            Recursion potential at phase location
        """
        # Discretize coordinates
        disc_theta = round(theta / (PI/12)) * (PI/12)
        disc_phi = round(phi / (PI/12)) * (PI/12)
        
        # Normalize coordinates to [0, 2π)
        disc_theta = disc_theta % TWO_PI
        disc_phi = disc_phi % TWO_PI
        
        # Get recursion potential
        key = (disc_theta, disc_phi)
        return self.recursion_potential_field.get(key, 0.0)
    
    def detect_emergent_recursion(self, grain: Grain) -> bool:
        """
        Detect if a grain shows signs of emergent recursion.
        
        Args:
            grain: Grain to check for recursion
            
        Returns:
            True if recursion is detected, False otherwise
        """
        # Check for closed trajectories
        if grain.detect_closed_trajectory():
            self.record_closed_trajectory(
                grain.id, 
                list(grain.phase_trajectory),
                grain.coherence
            )
            return True
            
        # Check for self-reference in ancestry
        if grain.id in grain.ancestry:
            # Detected self-reference
            if len(grain.ancestry) >= 3:
                # Significant ancestry with self-reference
                self.record_recursive_event(
                    'self_reference',
                    {
                        'grain_id': grain.id,
                        'ancestry_size': len(grain.ancestry),
                        'phase_position': (grain.theta, grain.phi),
                        'polarity': grain.polarity,
                        'awareness': grain.awareness,
                        'recursive_tension': grain.recursive_tension
                    }
                )
                return True
        
        # Check for forbidden directions causing constraints
        if len(grain.forbidden_directions) > 5:
            # High constraints from history
            self.record_recursive_event(
                'constraint_recursion',
                {
                    'grain_id': grain.id,
                    'forbidden_count': len(grain.forbidden_directions),
                    'phase_position': (grain.theta, grain.phi),
                    'constraint_tension': grain.constraint_tension
                }
            )
            return True
            
        # No recursion detected
        return False


class RelationalManifold:
    """
    Core System where Recursive Constraint Dynamics Naturally Emerge
    
    This implementation represents a fundamental shift from the orchestration
    paradigm to one where the manifold itself embodies the dynamics directly.
    Circular recursion is no longer imposed by explicit mechanisms but emerges
    naturally from the system's constraint history, collapse dynamics, and
    ancestry relationships.
    """
    
    def __init__(self):
        """Initialize the emergent relational manifold"""
        # Core time property - emergent from collapse history
        self.time = 0.0
        
        # Track grains - fundamental units of individuation
        self.grains = {}  # Maps grain_id -> Grain
        
        # Track collapse history - the committed structural memory
        self.collapse_history = []
        
        # Track void formation events - where tension creates absence
        self.void_formation_events = []
        
        # Relation memory - maps (grain_id1, grain_id2) -> list of interactions
        self.relation_memory = defaultdict(list)
        
        # Emergent components that track different aspects of recursion
        self.phase_exclusion = PhaseExclusionField()
        self.recursive_tension = RecursiveTensionMap()
        self.ancestry_constraint = AncestryConstraintSurface()
        self.recursion_detector = EmergentRecursionDetector()
        
        # System-level metrics
        self.field_coherence = 0.5
        self.system_tension = 0.0
        self.total_collapses = 0
        self.superposition_count = 0
        
        # Phase field properties
        self.symmetry_level = 0.5
        self.circular_coherence = 0.5
        
        # System stability metrics
        self.system_stability = 0.0  # 0.0 (unstable) to 1.0 (stable)
        self.stable_duration = 0.0  # How long the system has been stable
        self.stable_threshold = 20.0  # Duration required to consider system stable
        self.last_major_event = 0.0  # Time of last major event
        self.stability_history = []  # Track stability over time
        
        # Tracking of emergent structures
        self.vortices = []
        self.lightlike_pathways = {
            'structure': [],  # Structure-building pathways
            'decay': []       # Structure-decaying pathways
        }
        
        # Tracking pairs of opposite grains
        self.opposite_pairs = []
        
        # Classification of grains by state
        self.grain_states = {
            'active': set(),
            'constrained': set(),
            'saturated': set(),
            'inactive': set(),
            'superposition': set()
        }
    
    def add_grain(self, grain_id: str = None) -> Grain:
        """
        Add a grain to the manifold environment.
        No explicit coordinates - position emerges from relations.
        Grain starts in superposition state by default.
        
        Args:
            grain_id: Unique ID for the grain, or None to generate one
            
        Returns:
            The grain instance
        """
        # Generate ID if not provided
        if grain_id is None:
            grain_id = str(uuid.uuid4())
            
        # Create grain in relational space, starting in superposition
        grain = Grain(grain_id)
        
        # Store the grain
        self.grains[grain_id] = grain
        
        # Update superposition count
        self.superposition_count += 1
        
        # Update grain state classification
        self.grain_states['superposition'].add(grain_id)
        
        return grain
    
    def get_grain(self, grain_id: str) -> Optional[Grain]:
        """
        Get a grain by ID.
        
        Args:
            grain_id: ID of the grain to get
            
        Returns:
            Grain object or None if not found
        """
        return self.grains.get(grain_id)
    
    def connect_grains(self, grain_id1: str, grain_id2: str, relation_strength: float = 0.5):
        """
        Establish a relational connection between grains.
        
        Args:
            grain_id1: First grain ID
            grain_id2: Second grain ID
            relation_strength: Strength of relation [-1.0 to 1.0]
        """
        # Ensure both grains exist
        if grain_id1 not in self.grains:
            self.add_grain(grain_id1)
        if grain_id2 not in self.grains:
            self.add_grain(grain_id2)
            
        grain1 = self.grains[grain_id1]
        grain2 = self.grains[grain_id2]
        
        # Create bidirectional relation
        grain1.update_relations(grain_id2, relation_strength)
        grain2.update_relations(grain_id1, relation_strength)
        
        # Strong relation can potentially trigger collapse from superposition
        if abs(relation_strength) > 0.8:
            # Check if either grain is in superposition and potentially collapse
            if grain1.is_in_superposition() and random.random() < 0.3:
                self._potentially_resolve_from_superposition(grain_id1)
            if grain2.is_in_superposition() and random.random() < 0.3:
                self._potentially_resolve_from_superposition(grain_id2)
        
        # Update stability metrics
        self._update_system_stability()
    
    def set_opposite_grains(self, grain_id1: str, grain_id2: str):
        """
        Set two grains as opposites of each other.
        Opposite grains create structural constraints that drive dynamics.
        
        Args:
            grain_id1: First grain ID
            grain_id2: Second grain ID
        """
        # Ensure both grains exist
        if grain_id1 not in self.grains:
            self.add_grain(grain_id1)
        if grain_id2 not in self.grains:
            self.add_grain(grain_id2)
            
        # Check if pair already exists (in either order)
        pair = (grain_id1, grain_id2)
        reverse_pair = (grain_id2, grain_id1)
        
        if pair not in self.opposite_pairs and reverse_pair not in self.opposite_pairs:
            self.opposite_pairs.append(pair)
        
        # Connect with a strong opposing relation
        self.connect_grains(grain_id1, grain_id2, -0.8)  # Strong negative relation
        
        # Set opposite polarities - creates tension that drives dynamics
        grain1 = self.grains[grain_id1]
        grain2 = self.grains[grain_id2]
        
        # Set polarities if not already set or very weak
        if abs(grain1.polarity) < 0.3:
            grain1.polarity = 0.8  # Strong positive
        if abs(grain2.polarity) < 0.3:
            grain2.polarity = -0.8  # Strong negative
            
        # Setting opposites likely causes collapse from superposition
        if grain1.is_in_superposition():
            self._potentially_resolve_from_superposition(grain_id1, probability=0.7)
        if grain2.is_in_superposition():
            self._potentially_resolve_from_superposition(grain_id2, probability=0.7)
        
        # Update stability metrics
        self._update_system_stability()
    
    def _potentially_resolve_from_superposition(self, grain_id: str, probability: float = 0.5) -> bool:
        """
        Potentially resolve a grain from superposition state based on probability.
        
        Args:
            grain_id: ID of the grain to potentially resolve
            probability: Probability of resolving [0.0 to 1.0]
            
        Returns:
            True if resolved, False otherwise
        """
        grain = self.grains.get(grain_id)
        if not grain or not grain.is_in_superposition():
            return False
            
        # Check probability
        if random.random() >= probability:
            return False
            
        # Resolve from superposition
        awareness_level = random.uniform(0.05, 0.2)
        resolved = grain.resolve_from_superposition(awareness_level)
        
        # Update superposition count if successfully resolved
        if resolved:
            self.superposition_count = max(0, self.superposition_count - 1)
            
            # Initialize ancestry to include self-reference
            grain.ancestry.add(grain_id)  # Self-reference - grain is its own ancestor
            
            # Update grain state tracking
            self.grain_states['superposition'].discard(grain_id)
            self.grain_states['active'].add(grain_id)
            
            # Initialize activity state
            grain.update_activity_state(self.time)
            
            # Record as collapse event
            event = {
                'type': 'superposition_collapse',
                'time': self.time,
                'grain_id': grain_id,
                'source': grain_id,  # Self-referential source
                'target': grain_id,  # Self as target
                'field_genesis': True,  # Marker for field-level causation
                'new_awareness': grain.awareness,
                'polarity': grain.polarity
            }
            
            self.collapse_history.append(event)
            self.total_collapses += 1
            
            # Reset system stability when a grain collapses
            self.stable_duration = 0.0
            self.last_major_event = self.time
            
            return True
            
        return False
    
    def is_system_stable(self) -> bool:
        """
        Check if the system has reached a stable configuration.
        
        Returns:
            True if system is stable, False otherwise
        """
        # System is stable if stability is high and has been maintained
        # for longer than the stability threshold
        return (self.system_stability > 0.8 and 
                self.stable_duration >= self.stable_threshold)
    
    def get_stability_metrics(self) -> Dict[str, Any]:
        """
        Get detailed metrics about system stability.
        
        Returns:
            Dictionary with stability metrics
        """
        # Count grains in different states
        active_count = len(self.grain_states['active'])
        constrained_count = len(self.grain_states['constrained'])
        saturated_count = len(self.grain_states['saturated'])
        inactive_count = len(self.grain_states['inactive'])
        superposition_count = len(self.grain_states['superposition'])
        
        total_grains = len(self.grains)
        
        # Calculate ratios
        metrics = {
            'system_stability': self.system_stability,
            'stable_duration': self.stable_duration,
            'is_stable': self.is_system_stable(),
            'time_since_last_event': self.time - self.last_major_event,
            'total_grains': total_grains,
            'states': {
                'active': active_count,
                'constrained': constrained_count,
                'saturated': saturated_count,
                'inactive': inactive_count,
                'superposition': superposition_count
            }
        }
        
        # Calculate ratios if there are grains
        if total_grains > 0:
            metrics['ratios'] = {
                'active_ratio': active_count / total_grains,
                'constrained_ratio': constrained_count / total_grains,
                'saturated_ratio': saturated_count / total_grains,
                'inactive_ratio': inactive_count / total_grains,
                'superposition_ratio': superposition_count / total_grains,
                'stable_ratio': (saturated_count + inactive_count) / total_grains
            }
        else:
            metrics['ratios'] = {
                'active_ratio': 0.0,
                'constrained_ratio': 0.0,
                'saturated_ratio': 0.0,
                'inactive_ratio': 0.0,
                'superposition_ratio': 0.0,
                'stable_ratio': 0.0
            }
            
        # Get counts of stable grains
        stable_grains = 0
        for grain_id in self.grain_states['saturated'].union(self.grain_states['inactive']):
            grain = self.grains.get(grain_id)
            if grain and grain.is_stable():
                stable_grains += 1
                
        metrics['stable_grains_count'] = stable_grains
        metrics['stable_grains_ratio'] = stable_grains / total_grains if total_grains > 0 else 0.0
        
        return metrics
    
    def _update_grain_states(self):
        """Update the classification of grains by state"""
        # Reset state sets
        for state in self.grain_states:
            self.grain_states[state] = set()
            
        # Classify each grain
        for grain_id, grain in self.grains.items():
            if grain.is_in_superposition():
                self.grain_states['superposition'].add(grain_id)
            else:
                # Update grain activity state
                grain.update_activity_state(self.time)
                
                # Add to appropriate state set
                self.grain_states[grain.activity_state].add(grain_id)
    
    def _update_system_stability(self):
        """Update system stability metrics"""
        # Update grain states
        self._update_grain_states()
        
        # Get total grain count
        total_grains = len(self.grains)
        if total_grains == 0:
            self.system_stability = 1.0  # Empty system is stable
            return
            
        # Calculate state ratios
        active_ratio = len(self.grain_states['active']) / total_grains
        constrained_ratio = len(self.grain_states['constrained']) / total_grains
        saturated_ratio = len(self.grain_states['saturated']) / total_grains
        inactive_ratio = len(self.grain_states['inactive']) / total_grains
        superposition_ratio = len(self.grain_states['superposition']) / total_grains
        
        # Calculate stability metric
        # Stable states: saturated and inactive
        # Unstable states: active, constrained, superposition
        stability_value = saturated_ratio + inactive_ratio - active_ratio - superposition_ratio
        
        # Normalize to [0, 1] range
        self.system_stability = max(0.0, min(1.0, (stability_value + 1.0) / 2.0))
        
        # Update stable duration
        if self.system_stability > 0.7:
            # System is stable - increase duration
            self.stable_duration = self.time - self.last_major_event
        else:
            # System not stable - reset duration
            self.stable_duration = 0.0
            self.last_major_event = self.time
        
        # Record stability history
        self.stability_history.append({
            'time': self.time,
            'stability': self.system_stability,
            'duration': self.stable_duration,
            'active_ratio': active_ratio,
            'saturated_ratio': saturated_ratio,
            'inactive_ratio': inactive_ratio
        })
    
    def propagate_saturation(self, grain_id: str):
        """
        Propagate saturation effects to connected grains.
        
        Args:
            grain_id: ID of the saturated grain
        """
        grain = self.grains.get(grain_id)
        if not grain or grain.activity_state != "saturated":
            return
        
        # Propagate to related grains
        for related_id in grain.relations:
            if related_id in self.grains:
                related_grain = self.grains[related_id]
                
                # Skip superposition grains
                if related_grain.is_in_superposition():
                    continue
                    
                # Increase saturation of related grain
                related_grain.grain_saturation = min(
                    1.0, 
                    related_grain.grain_saturation + 0.1
                )
                
                # Mark the connecting pathway as more saturated
                related_grain.update_pathway_saturation(grain_id, related_id, 0.3)
                
                # Update activity state
                related_grain.update_activity_state(self.time)
                
        # Update grain states
        self._update_grain_states()
        
        # Update system stability
        self._update_system_stability()
    
    def manifest_dynamics(self, time_step: float = 1.0) -> Dict[str, Any]:
        """
        Allow the inherent dynamics of the system to naturally manifest
        based on relational structure and constraint history.
        
        Args:
            time_step: Amount of time to evolve
            
        Returns:
            Dictionary with manifestation results
        """
        # Track significant events
        events = []
        
        # 1. FIELD PROPAGATION - Field naturally propagates based on gradients
        self._manifest_field_propagation(time_step)
        
        # 2. NATURAL COLLAPSE - Collapse naturally occurs where conditions align
        collapse_events = self._manifest_natural_collapses()
        events.extend(collapse_events)
        
        # 3. VOID FORMATION - Tension naturally forms voids where alignment fails
        void_events = self._manifest_void_formation()
        events.extend(void_events)
        
        # 4. CONSTRAINT EMERGENCE - Structural constraints emerge from history
        self._manifest_constraint_emergence()
        
        # 5. PROPAGATE SATURATION - Saturated grains influence connected grains
        self._propagate_saturations()
        
        # 6. RECURSION DETECTION - Detect naturally emergent circular recursion
        recursion_events = self._detect_emergent_recursion()
        events.extend(recursion_events)
        
        # 7. STRUCTURE IDENTIFICATION - Identify emergent structures
        structure_events = self._identify_emergent_structures()
        events.extend(structure_events)
        
        # 8. MEMORY EMERGENCE - Relational memory naturally emerges from interaction
        self._manifest_memory_emergence()
        
        # Update time
        self.time += time_step
        
        # Update system stability
        self._update_system_stability()
        
        # Observe system metrics
        self._observe_system_metrics()
        
        # Check if system has reached stability
        is_stable = self.is_system_stable()
        
        return {
            'time': self.time,
            'events': events,
            'collapses': len(collapse_events),
            'voids_formed': len(void_events),
            'recursions_detected': len(recursion_events),
            'structures_identified': len(structure_events),
            'coherence': self.field_coherence,
            'tension': self.system_tension,
            'superposition_count': self.superposition_count,
            'vortices': len(self.vortices),
            'structure_pathways': len(self.lightlike_pathways['structure']),
            'decay_pathways': len(self.lightlike_pathways['decay']),
            'system_stability': self.system_stability,
            'stable_duration': self.stable_duration,
            'is_stable': is_stable
        }
    
    def _propagate_saturations(self):
        """Propagate saturation effects from saturated grains"""
        # Find saturated grains
        for grain_id in self.grain_states['saturated']:
            # Propagate saturation from this grain
            self.propagate_saturation(grain_id)
    
    
    def _manifest_field_propagation(self, time_step: float):
        """
        Field naturally propagates based on gradients and relations.
        Emergence happens through local interactions, not global rules.
        
        Args:
            time_step: Amount of time to evolve
        """
        # Create temporary field values to avoid propagation bias
        awareness_updates = {}
        polarity_updates = {}
        phase_updates = {}
        
        # Process each grain
        for grain_id, grain in self.grains.items():
            # Skip grains with no relations
            if not grain.relations:
                continue
                
            # Check if grain is in superposition
            is_superposition = grain.is_in_superposition()
            
            # Calculate field updates based on relations
            total_awareness_flow = 0.0
            total_polarity_influence = 0.0
            total_phase_shift_theta = 0.0
            total_phase_shift_phi = 0.0
            relation_count = 0
            
            for related_id, relation_strength in grain.relations.items():
                if related_id not in self.grains:
                    continue
                    
                related_grain = self.grains[related_id]
                
                # Basic awareness flow based on gradient
                if not is_superposition and not related_grain.is_in_superposition():
                    awareness_diff = related_grain.awareness - grain.awareness
                    
                    # Basic flow scaled by relation
                    base_flow = awareness_diff * relation_strength
                    
                    # Adjust flow based on saturation levels
                    source_saturation = related_grain.grain_saturation
                    target_saturation = grain.grain_saturation
                    
                    # High saturation reduces inflow, low saturation increases it
                    saturation_factor = (1.0 - target_saturation) * (0.5 + source_saturation * 0.5)
                    
                    # Scale flow by saturation factor
                    awareness_flow = base_flow * saturation_factor
                    
                    # Accumulate total flow
                    total_awareness_flow += awareness_flow
                
                # Polarity influence - creates circular patterns naturally
                if not is_superposition:
                    # Get other grain's polarity
                    other_polarity = related_grain.polarity
                    
                    # Calculate angular difference (on circle)
                    polarity_diff = self._angular_difference(grain.polarity, other_polarity)
                    
                    # Determine natural direction of polarity shift
                    # Polarities naturally want to align with connected grains
                    shift_direction = 1.0 if self._shortest_path_direction(grain.polarity, other_polarity) > 0 else -1.0
                    
                    # Calculate polarity influence scaled by relation
                    # Strong relations create stronger alignment influences
                    polarity_influence = polarity_diff * abs(relation_strength) * shift_direction * 0.05
                    
                    # Accumulate total influence
                    total_polarity_influence += polarity_influence
                
                # Phase shifts - drift in toroidal coordinates creates flow patterns
                # Phase shifts are very small to allow natural emergence
                if not is_superposition:
                    # Get phase positions
                    grain_theta = grain.theta
                    grain_phi = grain.phi
                    other_theta = related_grain.theta
                    other_phi = related_grain.phi
                    
                    # Calculate angular differences
                    theta_diff = self._angular_difference_2pi(grain_theta, other_theta)
                    phi_diff = self._angular_difference_2pi(grain_phi, other_phi)
                    
                    # Determine direction of shift (shortest path)
                    theta_direction = 1.0 if self._shortest_path_direction_2pi(grain_theta, other_theta) > 0 else -1.0
                    phi_direction = 1.0 if self._shortest_path_direction_2pi(grain_phi, other_phi) > 0 else -1.0
                    
                    # Calculate shift magnitude
                    # Positive relations pull phases together, negative push apart
                    shift_factor = 0.01 * abs(relation_strength)
                    
                    if relation_strength > 0:
                        # Positive relation - pull together
                        theta_shift = theta_diff * theta_direction * shift_factor
                        phi_shift = phi_diff * phi_direction * shift_factor
                    else:
                        # Negative relation - push apart
                        theta_shift = -theta_diff * theta_direction * shift_factor
                        phi_shift = -phi_diff * phi_direction * shift_factor
                    
                    # Accumulate total shifts
                    total_phase_shift_theta += theta_shift
                    total_phase_shift_phi += phi_shift
                
                # Count valid relation
                relation_count += 1
            
            # Calculate average updates per relation
            if relation_count > 0:
                avg_awareness_flow = total_awareness_flow / relation_count
                avg_polarity_influence = total_polarity_influence / relation_count
                avg_phase_shift_theta = total_phase_shift_theta / relation_count
                avg_phase_shift_phi = total_phase_shift_phi / relation_count
                
                # Calculate updates scaled by time step
                awareness_update = avg_awareness_flow * time_step
                polarity_update = avg_polarity_influence * time_step
                phase_update_theta = avg_phase_shift_theta * time_step
                phase_update_phi = avg_phase_shift_phi * time_step
                
                # Special handling for superposition
                if is_superposition:
                    # Superposition doesn't directly update awareness but may resolve
                    if abs(avg_awareness_flow) > 0.3:
                        # Strong flow may cause collapse from superposition
                        collapse_probability = min(0.5, abs(avg_awareness_flow) * 0.3)
                        self._potentially_resolve_from_superposition(grain_id, collapse_probability)
                    
                    # Polarity can still be influenced even in superposition
                    # This creates predisposition for future collapse direction
                    polarity_updates[grain_id] = polarity_update
                else:
                    # Store updates for later application
                    awareness_updates[grain_id] = awareness_update
                    polarity_updates[grain_id] = polarity_update
                    
                    # Store phase updates
                    if grain_id not in phase_updates:
                        phase_updates[grain_id] = [0.0, 0.0]
                    phase_updates[grain_id][0] += phase_update_theta
                    phase_updates[grain_id][1] += phase_update_phi
        
        # Apply all updates
        for grain_id, update in awareness_updates.items():
            grain = self.grains[grain_id]
            
            # Skip if grain has resolved to superposition in the meantime
            if grain.is_in_superposition():
                continue
                
            # Apply awareness update with boundary checking
            grain.awareness = max(-1.0, min(1.0, grain.awareness + update))
                
            # Update activation based on awareness change
            if abs(update) > 0.01:
                activation_change = abs(update) * 2.0
                grain.grain_activation = min(1.0, grain.grain_activation + activation_change)
            else:
                # Slight decay for inactive grains
                grain.grain_activation = max(0.0, grain.grain_activation * 0.99)
        
        # Apply polarity updates
        for grain_id, update in polarity_updates.items():
            grain = self.grains[grain_id]
            
            # Apply polarity update using natural circular wrapping
            # This is how circular behavior emerges naturally without explicit mechanism
            current_polarity = grain.polarity
            
            # Convert to angle, apply update, and convert back
            polarity_angle = (current_polarity + 1.0) * PI  # Map [-1,1] to [0,2π]
            polarity_angle = (polarity_angle + update) % TWO_PI  # Apply update with wrapping
            new_polarity = (polarity_angle / PI) - 1.0  # Map back to [-1,1]
            
            grain.polarity = new_polarity
        
        # Apply phase updates
        for grain_id, (theta_update, phi_update) in phase_updates.items():
            grain = self.grains[grain_id]
            
            # Apply phase updates with wrapping
            grain.theta = (grain.theta + theta_update) % TWO_PI
            grain.phi = (grain.phi + phi_update) % TWO_PI
            
            # Record new phase position in trajectory
            grain.record_phase_position()
    
    def _manifest_natural_collapses(self) -> List[Dict[str, Any]]:
        """
        Collapse naturally occurs where readiness and structural alignment manifest.
        
        Returns:
            List of collapse event dictionaries
        """
        collapse_events = []
        
        # Check all grains for potential collapse readiness
        for source_id, source_grain in self.grains.items():
            # Special handling for superposition states
            if source_grain.is_in_superposition():
                # Superposition can spontaneously collapse with low probability
                if random.random() < 0.05:
                    if self._potentially_resolve_from_superposition(source_id, probability=0.3):
                        continue  # Already processed this grain
            
            # Skip grains with little activation unless in superposition
            if source_grain.grain_activation < 0.3 and not source_grain.is_in_superposition():
                continue
                
            # Check each relation for potential collapse
            for target_id in source_grain.relations:
                # Skip incompatible collapse pairs
                if not self._is_valid_collapse_pair(source_id, target_id):
                    continue
                
                # Check if this collapse direction is forbidden by history
                target_grain = self.grains[target_id]
                if target_grain.is_collapse_forbidden(source_id, target_id):
                    # This direction is forbidden by history
                    # Increase constraint tension instead
                    self._update_constraint_tension(source_id, target_id)
                    continue
                
                # Check natural collapse readiness
                readiness = self._calculate_collapse_readiness(source_id, target_id)
                
                # Adjust readiness based on constraint history
                readiness = self._adjust_collapse_readiness_constraints(source_id, target_id, readiness)
                
                # Check structural alignment
                alignment = self._attempt_structural_alignment(source_id, target_id)
                
                # If naturally ready and aligned, collapse occurs
                if readiness > 0.7 and alignment > 0.3:
                    # Allow natural collapse to occur
                    collapse_event = self._manifest_collapse(source_id, target_id, readiness)
                    if collapse_event:
                        collapse_events.append(collapse_event)
                # If ready but not aligned, tension builds
                elif readiness > 0.7 and alignment <= 0.3:
                    # Update structural tension
                    self._update_structural_tension(source_id, target_id, readiness)
                        
                    # Record in memory
                    relation_key = (source_id, target_id)
                    self.relation_memory[relation_key].append({
                        'time': self.time,
                        'type': 'tension',
                        'strength': readiness
                    })
        
        return collapse_events
    
    def _is_valid_collapse_pair(self, source_id: str, target_id: str) -> bool:
        """
        Check if a source-target pair is valid for potential collapse.
        
        Args:
            source_id: Source grain ID
            target_id: Target grain ID
            
        Returns:
            True if valid collapse pair, False otherwise
        """
        # Ensure both grains exist
        if source_id not in self.grains or target_id not in self.grains:
            return False
            
        source_grain = self.grains[source_id]
        target_grain = self.grains[target_id]
        
        # Special case: If target is in superposition, it can always receive collapse
        if target_grain.is_in_superposition():
            return True
            
        # Skip if target already highly saturated
        if target_grain.grain_saturation > 0.9:
            return False
            
        # Skip if target not in source relations
        if target_id not in source_grain.relations:
            return False
            
        # All checks passed
        return True
    
    def _calculate_collapse_readiness(self, source_id: str, target_id: str) -> float:
        """
        Calculate the readiness for collapse between source and target grains.
        
        Args:
            source_id: Source grain ID
            target_id: Target grain ID
            
        Returns:
            Readiness value [0.0-1.0]
        """
        # Get grains
        source_grain = self.grains.get(source_id)
        target_grain = self.grains.get(target_id)
        
        # Validate grains exist
        if not source_grain or not target_grain:
            return 0.0
        
        # Get relation strength
        relation_strength = abs(source_grain.relations.get(target_id, 0.0))
        
        # Base readiness is from source activation and relation
        base_readiness = source_grain.grain_activation * relation_strength
        
        # Enhanced by source saturation (higher saturation = more readiness)
        saturation_factor = source_grain.grain_saturation
        saturation_influence = saturation_factor * 0.5
        
        # Inverse target saturation effect (lower target saturation = more readiness)
        target_saturation_influence = (1.0 - target_grain.grain_saturation) * 0.3
        
        # Polarity alignment effect
        polarity_factor = 0.0
        source_polarity = source_grain.polarity
        target_polarity = target_grain.polarity
        
        # Calculate polarity effect based on alignment type
        if source_polarity * target_polarity > 0:
            # Same polarity direction - similar magnitudes increase readiness
            polarity_diff = abs(abs(source_polarity) - abs(target_polarity))
            polarity_factor = (1.0 - polarity_diff) * 0.3
        else:
            # Opposite polarity - greater combined strength increases readiness
            polarity_sum = abs(source_polarity) + abs(target_polarity)
            polarity_factor = polarity_sum * 0.15
        
        # Combine factors with weights
        readiness = (
            base_readiness * 0.4 +
            saturation_influence * 0.3 +
            target_saturation_influence * 0.2 +
            polarity_factor * 0.1
        )
        
        # Apply ancestry effect
        if source_id in target_grain.ancestry:
            # Source is already in target's ancestry - adds recursion benefit
            readiness *= 1.1  # 10% increase
        
        # Apply phase position effect
        position_factor = self._calculate_phase_position_factor(source_grain, target_grain)
        readiness *= (0.9 + position_factor * 0.2)  # ±10% adjustment
        
        # Normalize to [0.0, 1.0]
        return min(1.0, max(0.0, readiness))
    
    def _adjust_collapse_readiness_constraints(self, source_id: str, target_id: str, 
                                             readiness: float) -> float:
        """
        Adjust collapse readiness based on constraint history.
        
        Args:
            source_id: Source grain ID
            target_id: Target grain ID
            readiness: Base readiness value
            
        Returns:
            Adjusted readiness value
        """
        # Get constraint factors from emergent components
        exclusion_factor = self.phase_exclusion.get_exclusion(
            self.grains[source_id].theta, 
            self.grains[source_id].phi
        )
        
        recursion_factor = self.recursive_tension.get_pairwise_tension(source_id, target_id)
        
        ancestry_constraint = self.ancestry_constraint.get_ancestry_constraint(source_id, target_id)
        
        # Calculate constraint penalty
        constraint_penalty = (
            exclusion_factor * 0.4 +
            recursion_factor * 0.3 +
            ancestry_constraint * 0.3
        )
        
        # Apply penalty (stronger constraints reduce readiness)
        adjusted_readiness = readiness * (1.0 - constraint_penalty * 0.5)
        
        # Special case: unless they form an emergent circular pattern
        target_grain = self.grains[target_id]
        source_grain = self.grains[source_id]
        
        # Check for strong opposite polarities (circular potential)
        if (abs(source_grain.polarity) > 0.8 and abs(target_grain.polarity) > 0.8 and
            source_grain.polarity * target_grain.polarity < 0):
            # This is a potential circular boundary case
            # Check if this completes a circle (using recursion detector)
            recursion_potential = self.recursion_detector.get_recursion_potential(
                source_grain.theta, source_grain.phi
            )
            
            if recursion_potential > 0.5:
                # High potential for circular pattern - boost readiness
                circular_boost = recursion_potential * 0.4
                adjusted_readiness *= (1.0 + circular_boost)
        
        # Ensure within valid range
        return min(1.0, max(0.0, adjusted_readiness))
    
    def _calculate_phase_position_factor(self, source_grain: Grain, target_grain: Grain) -> float:
        """
        Calculate a factor based on relative phase positions of the grains.
        
        Args:
            source_grain: Source grain
            target_grain: Target grain
            
        Returns:
            Position factor [-0.5 to 0.5]
        """
        # Calculate phase distance
        d_theta = self._angular_difference_2pi(source_grain.theta, target_grain.theta)
        d_phi = self._angular_difference_2pi(source_grain.phi, target_grain.phi)
        
        # Normalize distances to [0,1]
        norm_d_theta = d_theta / PI
        norm_d_phi = d_phi / PI
        
        # Calculate average distance as a circular distance factor
        distance_factor = (norm_d_theta + norm_d_phi) / 2
        
        # Convert to position factor [-0.5, 0.5]
        # Closer positions get positive factor, farther get negative
        position_factor = 0.5 - distance_factor
        
        return position_factor
    
    def _attempt_structural_alignment(self, source_id: str, target_id: str) -> float:
        """
        Attempt to align source and target for collapse and return alignment quality.
        
        Args:
            source_id: Source grain ID
            target_id: Target grain ID
            
        Returns:
            Alignment quality [0.0-1.0]
        """
        # Get grains
        source_grain = self.grains.get(source_id)
        target_grain = self.grains.get(target_id)
        
        # Validate grains exist
        if not source_grain or not target_grain:
            return 0.0
            
        # Base alignment starts at moderate value
        alignment = 0.5
        
        # Check polarity alignment
        source_polarity = source_grain.polarity
        target_polarity = target_grain.polarity
        
        # Calculate polarity alignment
        if source_polarity * target_polarity > 0:
            # Same polarity direction - check for magnitude similarity
            polarity_diff = abs(abs(source_polarity) - abs(target_polarity))
            polarity_alignment = 1.0 - polarity_diff
            
            # Adjust alignment based on polarity match
            alignment = 0.7 * alignment + 0.3 * polarity_alignment
        else:
            # Opposite polarity can still work for certain cases
            polarity_sum = abs(source_polarity) + abs(target_polarity)
            
            # Check for extreme opposites (circular boundary case)
            if polarity_sum > 1.6:  # Both near +/-1
                # This is a special case that allows circular dynamics to emerge
                # We don't force this with an explicit mechanism, but allow it to happen
                alignment = 0.7 * alignment + 0.3 * 0.9  # High alignment for circular boundary
            else:
                alignment = 0.7 * alignment + 0.3 * 0.3  # Lower alignment for general opposites
        
        # Check ancestry relationship
        if source_id in target_grain.ancestry:
            # Source is in target's ancestry - recursive pattern
            # Allow recursive patterns to form naturally through alignment
            ancestry_factor = 0.7  # Good alignment for recursive patterns
            alignment = 0.6 * alignment + 0.4 * ancestry_factor
        
        # Check phase position alignment
        position_factor = self._calculate_phase_position_factor(source_grain, target_grain)
        position_alignment = position_factor + 0.5  # Convert to [0,1] range
        
        # Include position alignment in overall alignment
        alignment = 0.8 * alignment + 0.2 * position_alignment
        
        # Check path exclusion - reduce alignment if path is excluded
        exclusion = source_grain.get_path_exclusion_tension(target_grain.theta, target_grain.phi)
        if exclusion > 0.3:
            # Significant exclusion exists - reduce alignment
            alignment *= (1.0 - exclusion * 0.5)
        
        # Check constraint tension - reduce alignment but allow circular patterns
        constraint_tension = source_grain.constraint_tension + target_grain.constraint_tension
        if constraint_tension > 0.7:
            # High constraint tension - only allow alignment for circular patterns
            if polarity_sum > 1.6:  # Both near +/-1 (circular potential)
                # Maintain alignment for circular potential
                pass
            else:
                # Reduce alignment for non-circular patterns under constraint
                alignment *= (1.0 - constraint_tension * 0.3)
        
        # Normalize to [0.0, 1.0] range
        alignment = min(1.0, max(0.0, alignment))
        
        return alignment
    
    def _update_constraint_tension(self, source_id: str, target_id: str):
        """
        Update constraint tension when a collapse direction is forbidden.
        This records the system attempting to reuse pathways and being constrained.
        
        Args:
            source_id: Source grain ID
            target_id: Target grain ID
        """
        # Get grains
        source_grain = self.grains.get(source_id)
        target_grain = self.grains.get(target_id)
        
        if not source_grain or not target_grain:
            return
            
        # Increase constraint tension
        tension_increase = 0.2
        
        # Update source grain
        source_grain.constraint_tension = min(1.0, source_grain.constraint_tension + tension_increase)
        
        # Update target grain
        target_grain.constraint_tension = min(1.0, target_grain.constraint_tension + tension_increase)
        
        # Record in recursive tension map
        self.recursive_tension.record_pairwise_tension(source_id, target_id, tension_increase)
        
        # Record phase positions
        self.recursive_tension.record_phase_tension(
            source_grain.theta, source_grain.phi, 
            tension_increase
        )
        
        # Record in phase exclusion field
        self.phase_exclusion.add_exclusion(
            source_grain.theta,
            source_grain.phi,
            tension_increase * 0.7
        )
        
        # Update path exclusion in both grains
        source_grain.update_path_exclusion(target_grain.theta, target_grain.phi, tension_increase)
        target_grain.update_path_exclusion(source_grain.theta, source_grain.phi, tension_increase * 0.8)
        
        # Record in ancestry constraint surface if related through ancestry
        if source_id in target_grain.ancestry:
            self.ancestry_constraint.record_ancestry_constraint(
                source_id, target_id, 
                tension_increase * 1.2  # Stronger constraint for ancestry violations
            )
    
    def _update_structural_tension(self, source_id: str, target_id: str, readiness: float):
        """
        Update structural tension between source and target when alignment fails.
        
        Args:
            source_id: Source grain ID
            target_id: Target grain ID
            readiness: Collapse readiness value
        """
        # Skip if readiness is low
        if readiness < 0.3:
            return
            
        # Get grains
        source_grain = self.grains.get(source_id)
        target_grain = self.grains.get(target_id)
        
        if not source_grain or not target_grain:
            return
            
        # Calculate tension based on readiness
        tension_increase = readiness * 0.2
        
        # Update tension in source grain
        source_grain.constraint_tension = min(1.0, source_grain.constraint_tension + tension_increase)
        
        # Update tension in target grain (less impact)
        target_grain.constraint_tension = min(1.0, target_grain.constraint_tension + tension_increase * 0.5)
        
        # Record in recursive tension map
        self.recursive_tension.record_pairwise_tension(source_id, target_id, tension_increase * 0.5)
        
        # Record phase positions
        self.recursive_tension.record_phase_tension(
            source_grain.theta, source_grain.phi, 
            tension_increase * 0.5
        )
        
        # Record in phase exclusion field (weaker than constraint violation)
        self.phase_exclusion.add_exclusion(
            source_grain.theta,
            source_grain.phi,
            tension_increase * 0.3
        )
        
        # Record relation memory
        relation_key = (source_id, target_id)
        self.relation_memory[relation_key].append({
            'time': self.time,
            'type': 'tension',
            'strength': tension_increase
        })
        
        # Check for recursion potential
        if source_id in target_grain.ancestry:
            # Record recursive tension - creates driving force for circular patterns
            source_grain.recursive_tension = min(1.0, source_grain.recursive_tension + tension_increase * 0.7)
            target_grain.recursive_tension = min(1.0, target_grain.recursive_tension + tension_increase * 0.5)
            
            # Record in recursion detector
            self.recursion_detector.record_recursion_potential(
                source_grain.theta, source_grain.phi,
                tension_increase * 0.8
            )
    
    def _manifest_collapse(self, source_id: str, target_id: str, readiness: float) -> Optional[Dict[str, Any]]:
        """
        Manifest a collapse between source and target grains.
        
        Args:
            source_id: Source grain ID
            target_id: Target grain ID
            readiness: Collapse readiness value
            
        Returns:
            Collapse event dictionary or None if collapse failed
        """
        # Get grains
        source_grain = self.grains.get(source_id)
        target_grain = self.grains.get(target_id)
        
        if not source_grain or not target_grain:
            return None
            
        # Calculate collapse strength based on readiness and source activation
        collapse_strength = readiness * source_grain.grain_activation
        
        # Ensure minimum strength
        collapse_strength = max(0.1, collapse_strength)
        
        # Skip if collapse is too weak
        if collapse_strength < 0.1:
            return None
            
        # Resolve target from superposition if necessary
        if target_grain.is_in_superposition():
            resolved = target_grain.resolve_from_superposition(abs(collapse_strength) * 0.2)
            if resolved:
                # Update superposition count
                self.superposition_count = max(0, self.superposition_count - 1)
        
        # Calculate awareness transfer
        awareness_transfer = source_grain.awareness * collapse_strength * 0.2
        
        # Update target awareness
        new_awareness = target_grain.awareness + awareness_transfer
        
        # Ensure within range
        new_awareness = max(-1.0, min(1.0, new_awareness))
        
        # Update target awareness
        target_grain.awareness = new_awareness
        
        # Calculate polarity influence based on collapse strength
        source_polarity = source_grain.polarity
        target_polarity = target_grain.polarity
        
        # Natural polarity influence through collapse
        polarity_diff = self._angular_difference(source_polarity, target_polarity)
        polarity_shift_direction = 1.0 if self._shortest_path_direction(source_polarity, target_polarity) > 0 else -1.0
        
        # Calculate influence amount
        polarity_influence = polarity_diff * collapse_strength * 0.3 * polarity_shift_direction
        
        # Apply natural circular polarity update without forcing it
        # This is how circular patterns emerge naturally
        polarity_angle = (target_polarity + 1.0) * PI  # Map [-1,1] to [0,2π]
        polarity_angle = (polarity_angle + polarity_influence) % TWO_PI  # Apply with natural wrapping
        new_polarity = (polarity_angle / PI) - 1.0  # Map back to [-1,1]
        
        # Update target polarity
        target_grain.polarity = new_polarity
        
        # Increase grain activation
        activation_boost = collapse_strength * 0.3
        target_grain.grain_activation = min(1.0, target_grain.grain_activation + activation_boost)
        
        # Increase saturation (collapse memory)
        saturation_increase = collapse_strength * 0.2
        target_grain.grain_saturation = min(1.0, target_grain.grain_saturation + saturation_increase)
        
        # Update ancestry - target gains source's ancestry plus source itself
        target_grain.ancestry.add(source_id)
        target_grain.ancestry.update(source_grain.ancestry)
        
        # Update phase trajectories
        target_grain.record_phase_position()
        
        # Check for emergent circular patterns
        circular_pattern = False
        if (abs(source_polarity) > 0.8 and abs(target_polarity) > 0.8 and
            source_polarity * target_polarity < 0):
            # Detected potential circular pattern without imposing it
            # Just through natural phase shifts on a circle
            
            # Record in recursion detector for future detection
            self.recursion_detector.record_recursive_event(
                'polarity_wraparound',
                {
                    'source_id': source_id,
                    'target_id': target_id,
                    'source_polarity': source_polarity,
                    'target_polarity': target_polarity,
                    'new_polarity': new_polarity,
                    'phase_position': (target_grain.theta, target_grain.phi)
                }
            )
            
            # Flag as circular pattern
            circular_pattern = True
            
            # Record in recursive tension map
            self.recursive_tension.record_circular_event(
                source_id, target_id,
                source_polarity, target_polarity,
                target_grain.theta, target_grain.phi,
                self.time
            )
        
        # Check for recursion in ancestry
        if target_id in source_grain.ancestry:
            # Source has target in its ancestry - truly recursive pattern
            # Let the detector know about this
            self.recursion_detector.record_recursive_event(
                'ancestry_recursion',
                {
                    'source_id': source_id,
                    'target_id': target_id,
                    'source_ancestry_size': len(source_grain.ancestry),
                    'target_ancestry_size': len(target_grain.ancestry)
                }
            )
            
            # Record in ancestry constraint surface
            self.ancestry_constraint.record_self_reference(
                target_id, 
                min(1.0, len(target_grain.ancestry) / 10)  # Strength based on ancestry size
            )
        
        # Add forbidden directions if this pathway has been used too often
        # This ensures the manifold cannot keep using the same pathways
        relation_key = (source_id, target_id)
        if relation_key in self.relation_memory:
            collapse_count = sum(1 for interaction in self.relation_memory[relation_key]
                               if interaction.get('type') == 'collapse')
            
            if collapse_count > 5:
                # Too many collapses on this path - mark as forbidden
                source_grain.add_forbidden_direction(source_id, target_id)
                target_grain.add_forbidden_direction(source_id, target_id)
        
        # Record collapse event
        event = {
            'type': 'collapse',
            'time': self.time,
            'source': source_id,
            'target': target_id,
            'strength': collapse_strength,
            'source_polarity': source_polarity,
            'target_polarity': target_polarity,
            'new_polarity': new_polarity,
            'new_awareness': target_grain.awareness,
            'circular_pattern': circular_pattern
        }
        
        # Add to collapse history
        self.collapse_history.append(event)
        self.total_collapses += 1
        
        # Record in relation memory
        self.relation_memory[relation_key].append({
            'time': self.time,
            'type': 'collapse',
            'strength': collapse_strength
        })
        
        # Record ancestry relationship
        self.ancestry_constraint.record_ancestry_constraint(
            source_id, target_id,
            collapse_strength * 0.5
        )
        
        # Check for shared ancestry between source and target
        shared_ancestry = source_grain.ancestry.intersection(target_grain.ancestry)
        if shared_ancestry:
            self.ancestry_constraint.record_shared_ancestry(
                source_id, target_id,
                len(shared_ancestry)
            )
            
        # Update path exclusion and phase fields
        self.phase_exclusion.add_exclusion(
            source_grain.theta, source_grain.phi,
            collapse_strength * 0.4
        )
        
        # Return collapse event
        return event
    
    def _manifest_void_formation(self) -> List[Dict[str, Any]]:
        """
        Tension naturally forms voids where alignment fails.
        
        Voids emerge from regions of high constraint tension that cannot resolve
        through collapse, forcing the system to find new pathways.
        
        Returns:
            List of void formation event dictionaries
        """
        # Track void formation events
        void_events = []
        
        # Find regions of high tension
        high_tension_grains = []
        
        for grain_id, grain in self.grains.items():
            # Skip superposition grains
            if grain.is_in_superposition():
                continue
                
            # Check for high constraint tension
            if grain.constraint_tension > 0.7:
                high_tension_grains.append((grain_id, grain.constraint_tension))
        
        # Sort by tension (highest first)
        high_tension_grains.sort(key=lambda x: x[1], reverse=True)
        
        # Process highest tension grains
        for grain_id, tension in high_tension_grains[:min(5, len(high_tension_grains))]:
            # Skip if random threshold not met (probabilistic void formation)
            if random.random() > tension * 0.5:
                continue
                
            # Get grain
            grain = self.grains[grain_id]
            
            # Form void - decrease awareness and increase decay tendency
            awareness_loss = tension * 0.3
            grain.awareness = max(-1.0, grain.awareness - awareness_loss)
            
            # Decrease activation
            activation_loss = tension * 0.4
            grain.grain_activation = max(0.0, grain.grain_activation - activation_loss)
            
            # Shift polarity toward decay (negative)
            # This happens naturally through the circular mechanics
            polarity_shift = -tension * 0.3
            
            # Apply circular polarity shift toward decay
            polarity_angle = (grain.polarity + 1.0) * PI  # Map [-1,1] to [0,2π]
            polarity_angle = (polarity_angle + polarity_shift) % TWO_PI  # Apply shift with wrapping
            new_polarity = (polarity_angle / PI) - 1.0  # Map back to [-1,1]
            
            # Update polarity
            grain.polarity = new_polarity
            
            # Reset constraint tension
            grain.constraint_tension = 0.0
            
            # Update path exclusion
            # This reinforces that this region cannot support structure
            self.phase_exclusion.add_exclusion(
                grain.theta, grain.phi,
                tension * 0.6
            )
            
            # Create void event
            void_event = {
                'type': 'void_formation',
                'time': self.time,
                'grain_id': grain_id,
                'tension': tension,
                'awareness_loss': awareness_loss,
                'polarity_shift': polarity_shift,
                'new_polarity': new_polarity,
                'position': (grain.theta, grain.phi)
            }
            
            # Record event
            void_events.append(void_event)
            self.void_formation_events.append(void_event)
            
            # Update relation memory for all connections
            for related_id in grain.relations:
                relation_key = tuple(sorted([grain_id, related_id]))
                
                self.relation_memory[relation_key].append({
                    'time': self.time,
                    'type': 'void',
                    'tension': tension
                })
        
        # Update system metrics
        if void_events:
            self.system_tension = max(0.0, self.system_tension - 0.1 * len(void_events))
        
        return void_events
    
    def _manifest_constraint_emergence(self):
        """
        Constraints naturally emerge from the history of the system.
        
        This process allows the system to recursively constrain itself
        based on its own history, creating patterns without external rules.
        """
        # Process all grains to identify and record emerging constraints
        for grain_id, grain in self.grains.items():
            # Skip superposition grains
            if grain.is_in_superposition():
                continue
                
            # Check for self-reference in ancestry
            if grain_id in grain.ancestry:
                # Self-referential grain - creates recursion constraint
                self.ancestry_constraint.record_self_reference(
                    grain_id,
                    min(1.0, len(grain.ancestry) / 10)
                )
                
                # The longer the ancestry, the stronger the constraint
                constraint_strength = min(0.8, len(grain.ancestry) * 0.05)
                grain.constraint_tension = max(grain.constraint_tension, constraint_strength)
            
            # Update ancestry density field at grain's position
            self.ancestry_constraint.record_ancestry_density(
                grain.theta, grain.phi,
                len(grain.ancestry) * 0.05
            )
            
            # Record phase constraints
            if len(grain.phase_trajectory) > 3:
                # The more a grain has traversed phase space,
                # the more it constrains those regions
                constraint_strength = min(0.5, len(grain.phase_trajectory) * 0.02)
                
                # Record last position
                last_theta, last_phi = grain.phase_trajectory[-1]
                self.phase_exclusion.add_exclusion(
                    last_theta, last_phi,
                    constraint_strength * 0.3
                )
            
            # Record recursive tension if present
            if grain.recursive_tension > 0.3:
                self.recursive_tension.record_tension(
                    grain_id,
                    grain.recursive_tension
                )
                
                # Record phase position
                self.recursive_tension.record_phase_tension(
                    grain.theta, grain.phi,
                    grain.recursive_tension
                )
            
            # Check for potential recursion indications
            recursion_indicators = {}
            
            # Self-reference creates recursion
            if grain_id in grain.ancestry:
                recursion_indicators['self_reference'] = 1.0
            
            # Ancestry size indicates potential recursion
            ancestry_size = len(grain.ancestry)
            if ancestry_size > 3:
                recursion_indicators['ancestry_size'] = min(1.0, ancestry_size * 0.05)
            
            # Forbidden directions indicate pathway constraints
            if len(grain.forbidden_directions) > 0:
                recursion_indicators['forbidden_paths'] = min(1.0, len(grain.forbidden_directions) * 0.1)
            
            # Path exclusion indicates reused pathways
            exclusion_level = grain.get_path_exclusion_tension(grain.theta, grain.phi)
            if exclusion_level > 0.3:
                recursion_indicators['path_exclusion'] = exclusion_level
            
            # Record recursion indicators if significant
            if recursion_indicators and any(v > 0.3 for v in recursion_indicators.values()):
                self.recursion_detector.record_potential_recursion(
                    grain_id,
                    recursion_indicators
                )
                
                # Record recursion potential at this phase location
                potential = max(recursion_indicators.values())
                self.recursion_detector.record_recursion_potential(
                    grain.theta, grain.phi,
                    potential
                )
        
        # Analyze relation memory for emerging constraint patterns
        for relation_key, interactions in self.relation_memory.items():
            if len(interactions) < 3:
                continue
                
            # Get grain IDs
            grain_id1, grain_id2 = relation_key
            
            # Count different interaction types
            collapse_count = sum(1 for i in interactions if i.get('type') == 'collapse')
            tension_count = sum(1 for i in interactions if i.get('type') == 'tension')
            void_count = sum(1 for i in interactions if i.get('type') == 'void')
            
            # Calculate constraint implications
            if collapse_count > 5:
                # Heavily used collapse pathway - increase constraint
                constraint_strength = min(0.8, collapse_count * 0.1)
                
                # Mark pathway as constrained - this forces new pathways to emerge
                grain1 = self.grains.get(grain_id1)
                grain2 = self.grains.get(grain_id2)
                
                if grain1 and grain2:
                    grain1.add_forbidden_direction(grain_id1, grain_id2)
                    grain2.add_forbidden_direction(grain_id1, grain_id2)
            
            if tension_count > 3:
                # High tension indicates alignment failure
                constraint_strength = min(0.6, tension_count * 0.1)
                
                # Record in phase exclusion field
                grain1 = self.grains.get(grain_id1)
                grain2 = self.grains.get(grain_id2)
                
                if grain1 and grain2:
                    self.phase_exclusion.add_exclusion(
                        grain1.theta, grain1.phi,
                        constraint_strength * 0.5
                    )
                    
                    self.phase_exclusion.add_exclusion(
                        grain2.theta, grain2.phi,
                        constraint_strength * 0.5
                    )
            
            if void_count > 2:
                # Frequent void formation indicates fundamental incompatibility
                constraint_strength = min(0.7, void_count * 0.2)
                
                # This region should be especially avoided
                grain1 = self.grains.get(grain_id1)
                grain2 = self.grains.get(grain_id2)
                
                if grain1 and grain2:
                    self.phase_exclusion.add_exclusion(
                        grain1.theta, grain1.phi,
                        constraint_strength * 0.7
                    )
                    
                    self.phase_exclusion.add_exclusion(
                        grain2.theta, grain2.phi,
                        constraint_strength * 0.7
                    )
    
    def _detect_emergent_recursion(self) -> List[Dict[str, Any]]:
        """
        Detect emergent circular recursion patterns that arise naturally
        from constraint dynamics.
        
        Returns:
            List of recursion event dictionaries
        """
        recursion_events = []
        
        # Check all grains for signs of emergent recursion
        for grain_id, grain in self.grains.items():
            # Skip superposition grains
            if grain.is_in_superposition():
                continue
                
            # Use detector to check for emergent recursion
            if self.recursion_detector.detect_emergent_recursion(grain):
                # Record event
                event = {
                    'type': 'emergent_recursion_detected',
                    'time': self.time,
                    'grain_id': grain_id,
                    'position': (grain.theta, grain.phi),
                    'polarity': grain.polarity,
                    'ancestry_size': len(grain.ancestry),
                    'is_self_referential': grain_id in grain.ancestry,
                    'forbidden_directions_count': len(grain.forbidden_directions),
                    'path_exclusion_level': grain.get_path_exclusion_tension(grain.theta, grain.phi)
                }
                
                recursion_events.append(event)
                
                # No need to force recursion - it has emerged naturally
        
        # Check for opposite polarity grains in proximity
        # This is a key pattern for circular recursion
        for grain_id1, grain1 in self.grains.items():
            if grain1.is_in_superposition():
                continue
                
            # Only consider grains with extreme polarity
            if abs(grain1.polarity) < 0.8:
                continue
                
            for grain_id2, grain2 in self.grains.items():
                if grain_id1 == grain_id2 or grain2.is_in_superposition():
                    continue
                    
                # Only consider grains with extreme opposite polarity
                if abs(grain2.polarity) < 0.8 or grain1.polarity * grain2.polarity >= 0:
                    continue
                    
                # Check if they're in phase proximity
                d_theta = self._angular_difference_2pi(grain1.theta, grain2.theta)
                d_phi = self._angular_difference_2pi(grain1.phi, grain2.phi)
                
                phase_distance = math.sqrt(d_theta**2 + d_phi**2)
                
                if phase_distance < 0.5:
                    # Opposite polarity grains in proximity
                    # This is a key signature of circular recursion
                    
                    # Record event
                    event = {
                        'type': 'polar_opposition_proximity',
                        'time': self.time,
                        'grain_id1': grain_id1,
                        'grain_id2': grain_id2,
                        'polarity1': grain1.polarity,
                        'polarity2': grain2.polarity,
                        'phase_distance': phase_distance,
                        'position1': (grain1.theta, grain1.phi),
                        'position2': (grain2.theta, grain2.phi)
                    }
                    
                    recursion_events.append(event)
                    
                    # No need to force any behavior - merely observe
        
        # Look for phase space regions with high recursion potential
        high_potential_regions = []
        
        for theta in np.linspace(0, TWO_PI, 12, endpoint=False):
            for phi in np.linspace(0, TWO_PI, 12, endpoint=False):
                # Get recursion potential at this location
                potential = self.recursion_detector.get_recursion_potential(theta, phi)
                
                if potential > 0.6:
                    high_potential_regions.append((theta, phi, potential))
        
        # Sort by potential (highest first)
        high_potential_regions.sort(key=lambda x: x[2], reverse=True)
        
        # Record events for highest potential regions
        for theta, phi, potential in high_potential_regions[:min(3, len(high_potential_regions))]:
            event = {
                'type': 'high_recursion_potential_region',
                'time': self.time,
                'position': (theta, phi),
                'potential': potential,
                'exclusion_level': self.phase_exclusion.get_exclusion(theta, phi),
                'tension_level': self.recursive_tension.get_phase_tension(theta, phi),
                'ancestry_density': self.ancestry_constraint.get_ancestry_density(theta, phi)
            }
            
            recursion_events.append(event)
            
            # No need to create anything - just identify
        
        return recursion_events
    
    def _identify_emergent_structures(self) -> List[Dict[str, Any]]:
        """
        Identify emergent structures that arise naturally from the dynamics.
        
        Returns:
            List of structure event dictionaries
        """
        structure_events = []
        
        # Identify vortices - regions where phase circulates around a center
        vortices = self._identify_vortices()
        self.vortices = vortices
        
        if vortices:
            event = {
                'type': 'vortices_identified',
                'time': self.time,
                'count': len(vortices),
                'vortices': vortices
            }
            
            structure_events.append(event)
        
        # Identify lightlike pathways - paths of low saturation grains
        structure_pathways, decay_pathways = self._identify_lightlike_pathways()
        
        self.lightlike_pathways = {
            'structure': structure_pathways,
            'decay': decay_pathways
        }
        
        if structure_pathways:
            event = {
                'type': 'structure_pathways_identified',
                'time': self.time,
                'count': len(structure_pathways),
                'pathways': structure_pathways
            }
            
            structure_events.append(event)
            
        if decay_pathways:
            event = {
                'type': 'decay_pathways_identified',
                'time': self.time,
                'count': len(decay_pathways),
                'pathways': decay_pathways
            }
            
            structure_events.append(event)
        
        # Identify phase domains - regions of similar phase properties
        phase_domains = self._identify_phase_domains()
        
        if phase_domains:
            event = {
                'type': 'phase_domains_identified',
                'time': self.time,
                'count': len(phase_domains),
                'domains': phase_domains
            }
            
            structure_events.append(event)
        
        # Identify closed phase trajectories - signs of recurring patterns
        closed_trajectories = self._identify_closed_trajectories()
        
        if closed_trajectories:
            event = {
                'type': 'closed_trajectories_identified',
                'time': self.time,
                'count': len(closed_trajectories),
                'trajectories': closed_trajectories
            }
            
            structure_events.append(event)
            
        return structure_events
    
    def _identify_vortices(self) -> List[Dict[str, Any]]:
        """
        Identify vortices in the phase space.
        
        Returns:
            List of vortex dictionaries
        """
        vortices = []
        
        # Find grains with high activation as potential vortex centers
        potential_centers = []
        
        for grain_id, grain in self.grains.items():
            if grain.is_in_superposition():
                continue
                
            if grain.grain_activation > 0.6:
                potential_centers.append(grain_id)
        
        # For each potential center, check for circulating relations
        for center_id in potential_centers:
            center_grain = self.grains[center_id]
            
            # Get positions
            center_theta = center_grain.theta
            center_phi = center_grain.phi
            
            # Find related grains
            related_grains = []
            
            for related_id in center_grain.relations:
                if related_id in self.grains and not self.grains[related_id].is_in_superposition():
                    related_grains.append(related_id)
            
            # Need at least 3 related grains to form vortex
            if len(related_grains) < 3:
                continue
                
            # Check for circular pattern
            # Calculate angular positions relative to center
            angles = []
            
            for related_id in related_grains:
                related_grain = self.grains[related_id]
                
                # Calculate relative position
                d_theta = related_grain.theta - center_theta
                d_phi = related_grain.phi - center_phi
                
                # Calculate angle in theta-phi plane
                angle = math.atan2(d_phi, d_theta)
                
                # Store with grain ID
                angles.append((related_id, angle))
            
            # Sort by angle
            angles.sort(key=lambda x: x[1])
            
            # Calculate circulation
            circulation = 0.0
            
            for i in range(len(angles)):
                current_id, _ = angles[i]
                next_id, _ = angles[(i + 1) % len(angles)]
                
                # Get grain objects
                current_grain = self.grains[current_id]
                next_grain = self.grains[next_id]
                
                # Get polarities
                current_polarity = current_grain.polarity
                next_polarity = next_grain.polarity
                
                # Calculate polarity flow
                polarity_flow = current_polarity * next_polarity
                
                # Contribute to circulation
                circulation += polarity_flow
            
            # Normalize circulation
            circulation /= len(angles)
            
            # Determine vortex direction
            direction = 'clockwise' if circulation > 0 else 'counterclockwise'
            
            # Calculate vortex strength
            strength = abs(circulation) * center_grain.grain_activation
            
            # Only record significant vortices
            if strength > 0.3:
                vortex = {
                    'center_id': center_id,
                    'theta': center_theta,
                    'phi': center_phi,
                    'circulation': circulation,
                    'strength': strength,
                    'direction': direction,
                    'polarity': center_grain.polarity,
                    'related_grains': related_grains
                }
                
                vortices.append(vortex)
        
        # Sort by strength (strongest first)
        vortices.sort(key=lambda v: v['strength'], reverse=True)
        
        return vortices
    
    def _identify_lightlike_pathways(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Identify lightlike pathways of low saturation grains.
        
        Returns:
            Tuple of (structure_pathways, decay_pathways)
        """
        structure_pathways = []
        decay_pathways = []
        
        # Find potential pathway seeds (low saturation, high activation)
        potential_seeds = []
        
        for grain_id, grain in self.grains.items():
            if grain.is_in_superposition():
                continue
                
            if grain.grain_saturation < 0.3 and grain.grain_activation > 0.5:
                potential_seeds.append((grain_id, grain.polarity))
        
        # Separate seeds by polarity
        structure_seeds = [(g_id, pol) for g_id, pol in potential_seeds if pol > 0.2]
        decay_seeds = [(g_id, pol) for g_id, pol in potential_seeds if pol < -0.2]
        
        # Sort by polarity magnitude (strongest first)
        structure_seeds.sort(key=lambda x: x[1], reverse=True)
        decay_seeds.sort(key=lambda x: -x[1], reverse=True)
        
        # Trace pathways from structure seeds
        for seed_id, _ in structure_seeds:
            # Skip if already in a pathway
            if any(seed_id in pathway['nodes'] for pathway in structure_pathways):
                continue
                
            # Trace pathway
            pathway = self._trace_pathway_from_seed(seed_id, polarity_type='structure')
            
            if pathway and len(pathway['nodes']) >= 3:
                structure_pathways.append(pathway)
        
        # Trace pathways from decay seeds
        for seed_id, _ in decay_seeds:
            # Skip if already in a pathway
            if any(seed_id in pathway['nodes'] for pathway in decay_pathways):
                continue
                
            # Trace pathway
            pathway = self._trace_pathway_from_seed(seed_id, polarity_type='decay')
            
            if pathway and len(pathway['nodes']) >= 3:
                decay_pathways.append(pathway)
        
        # Sort by length (longest first)
        structure_pathways.sort(key=lambda p: len(p['nodes']), reverse=True)
        decay_pathways.sort(key=lambda p: len(p['nodes']), reverse=True)
        
        return structure_pathways, decay_pathways
    
    def _trace_pathway_from_seed(self, seed_id: str, 
                               polarity_type: str = 'structure') -> Optional[Dict[str, Any]]:
        """
        Trace a pathway starting from a seed grain.
        
        Args:
            seed_id: Seed grain ID
            polarity_type: Type of pathway ('structure' or 'decay')
            
        Returns:
            Pathway dictionary or None if no pathway found
        """
        # Get seed grain
        seed_grain = self.grains.get(seed_id)
        if not seed_grain or seed_grain.is_in_superposition():
            return None
            
        # Initialize pathway
        pathway = {
            'nodes': [seed_id],
            'node_positions': [(seed_grain.theta, seed_grain.phi)],
            'avg_polarity': seed_grain.polarity,
            'avg_saturation': seed_grain.grain_saturation,
            'lightlike_ratio': 1.0 if seed_grain.grain_saturation < 0.3 else 0.0,
            'type': polarity_type
        }
        
        # Define direction filter based on type
        if polarity_type == 'structure':
            # For structure pathways, follow positive polarity
            polarity_filter = lambda p: p > 0.2
        else:
            # For decay pathways, follow negative polarity
            polarity_filter = lambda p: p < -0.2
        
        # Maximum pathway length
        max_length = 10
        
        # Current position
        current_id = seed_id
        
        # Trace pathway
        while len(pathway['nodes']) < max_length:
            current_grain = self.grains[current_id]
            
            # Find next grain to add
            best_next = None
            best_score = -1.0
            
            for related_id, relation_strength in current_grain.relations.items():
                # Skip if already in pathway
                if related_id in pathway['nodes']:
                    continue
                    
                # Skip if not in grains
                if related_id not in self.grains:
                    continue
                    
                related_grain = self.grains[related_id]
                
                # Skip superposition grains
                if related_grain.is_in_superposition():
                    continue
                    
                # Check polarity direction
                if not polarity_filter(related_grain.polarity):
                    continue
                    
                # Score based on saturation and polarity
                saturation_score = 1.0 - related_grain.grain_saturation  # Lower is better
                polarity_score = abs(related_grain.polarity)  # Higher is better
                
                # Combined score
                score = saturation_score * 0.6 + polarity_score * 0.4
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_next = related_id
            
            # If no next grain found, end pathway
            if best_next is None:
                break
                
            # Add next grain to pathway
            next_grain = self.grains[best_next]
            
            pathway['nodes'].append(best_next)
            pathway['node_positions'].append((next_grain.theta, next_grain.phi))
            
            # Update averages
            node_count = len(pathway['nodes'])
            pathway['avg_polarity'] = (pathway['avg_polarity'] * (node_count - 1) + next_grain.polarity) / node_count
            pathway['avg_saturation'] = (pathway['avg_saturation'] * (node_count - 1) + next_grain.grain_saturation) / node_count
            
            # Update lightlike ratio
            lightlike_count = sum(1 for n_id in pathway['nodes'] if self.grains[n_id].grain_saturation < 0.3)
            pathway['lightlike_ratio'] = lightlike_count / node_count
            
            # Move to next grain
            current_id = best_next
        
        # Valid pathway needs at least 3 nodes
        if len(pathway['nodes']) < 3:
            return None
            
        return pathway
    
    def _identify_phase_domains(self) -> List[Dict[str, Any]]:
        """
        Identify phase domains - regions with similar phase properties.
        
        Returns:
            List of phase domain dictionaries
        """
        # Initialize domains
        domains = []
        
        # Get all non-superposition grains
        active_grains = []
        
        for grain_id, grain in self.grains.items():
            if not grain.is_in_superposition():
                active_grains.append(grain_id)
        
        # Skip if too few grains
        if len(active_grains) < 5:
            return domains
            
        # Track which grains have been assigned to domains
        assigned = set()
        
        # Cluster grains by polarity and phase position
        for seed_id in active_grains:
            # Skip if already assigned
            if seed_id in assigned:
                continue
                
            seed_grain = self.grains[seed_id]
            
            # Find grains with similar polarity and phase
            domain_grains = []
            
            for grain_id in active_grains:
                if grain_id in assigned:
                    continue
                    
                grain = self.grains[grain_id]
                
                # Check polarity similarity
                polarity_diff = abs(grain.polarity - seed_grain.polarity)
                
                # Check phase proximity
                d_theta = self._angular_difference_2pi(grain.theta, seed_grain.theta)
                d_phi = self._angular_difference_2pi(grain.phi, seed_grain.phi)
                
                phase_distance = math.sqrt(d_theta**2 + d_phi**2)
                
                # Combine criteria
                if polarity_diff < 0.3 and phase_distance < 1.0:
                    domain_grains.append(grain_id)
                    assigned.add(grain_id)
            
            # Valid domain needs at least 3 grains
            if len(domain_grains) >= 3:
                # Calculate domain center
                domain_theta = sum(self.grains[g_id].theta for g_id in domain_grains) / len(domain_grains)
                domain_phi = sum(self.grains[g_id].phi for g_id in domain_grains) / len(domain_grains)
                
                # Calculate average polarity
                domain_polarity = sum(self.grains[g_id].polarity for g_id in domain_grains) / len(domain_grains)
                
                # Calculate domain radius
                max_distance = max(math.sqrt(
                    self._angular_difference_2pi(self.grains[g_id].theta, domain_theta)**2 +
                    self._angular_difference_2pi(self.grains[g_id].phi, domain_phi)**2
                ) for g_id in domain_grains)
                
                # Create domain
                domain = {
                    'grains': domain_grains,
                    'center': (domain_theta, domain_phi),
                    'radius': max_distance,
                    'avg_polarity': domain_polarity,
                    'avg_saturation': sum(self.grains[g_id].grain_saturation for g_id in domain_grains) / len(domain_grains),
                    'avg_activation': sum(self.grains[g_id].grain_activation for g_id in domain_grains) / len(domain_grains)
                }
                
                domains.append(domain)
        
        # Sort by size (largest first)
        domains.sort(key=lambda d: len(d['grains']), reverse=True)
        
        return domains
    
    def _identify_closed_trajectories(self) -> List[Dict[str, Any]]:
        """
        Identify closed phase trajectories - signs of recurring patterns.
        
        Returns:
            List of closed trajectory dictionaries
        """
        closed_trajectories = []
        
        # Get all non-superposition grains
        for grain_id, grain in self.grains.items():
            if grain.is_in_superposition():
                continue
                
            # Need at least 5 points in trajectory
            if len(grain.phase_trajectory) < 5:
                continue
                
            # Check if trajectory forms a closed curve
            if grain.detect_closed_trajectory():
                # Calculate trajectory properties
                
                # Calculate average radius from center
                trajectory = list(grain.phase_trajectory)
                
                # Calculate center
                center_theta = sum(t[0] for t in trajectory) / len(trajectory)
                center_phi = sum(t[1] for t in trajectory) / len(trajectory)
                
                # Calculate average radius
                avg_radius = sum(math.sqrt(
                    self._angular_difference_2pi(t[0], center_theta)**2 +
                    self._angular_difference_2pi(t[1], center_phi)**2
                ) for t in trajectory) / len(trajectory)
                
                # Calculate trajectory length
                length = 0.0
                for i in range(len(trajectory)):
                    t1 = trajectory[i]
                    t2 = trajectory[(i + 1) % len(trajectory)]
                    
                    d_theta = self._angular_difference_2pi(t1[0], t2[0])
                    d_phi = self._angular_difference_2pi(t1[1], t2[1])
                    
                    segment_length = math.sqrt(d_theta**2 + d_phi**2)
                    length += segment_length
                
                # Create trajectory
                closed_trajectory = {
                    'grain_id': grain_id,
                    'trajectory': trajectory,
                    'center': (center_theta, center_phi),
                    'avg_radius': avg_radius,
                    'length': length,
                    'polarity': grain.polarity,
                    'avg_saturation': grain.grain_saturation,
                    'ancestry_size': len(grain.ancestry)
                }
                
                closed_trajectories.append(closed_trajectory)
        
        # Sort by length (longest first)
        closed_trajectories.sort(key=lambda t: t['length'], reverse=True)
        
        return closed_trajectories
    
    def _manifest_memory_emergence(self):
        """
        Relational memory naturally emerges from interaction patterns.
        
        This process allows the manifold to accumulate history
        without explicit external mechanisms.
        """
        # Process each relation memory entry
        for relation_key, interactions in self.relation_memory.items():
            # Skip if fewer than 3 interactions
            if len(interactions) < 3:
                continue
                
            # Get grain IDs
            grain_id1, grain_id2 = relation_key
            
            # Ensure grains exist
            if grain_id1 not in self.grains or grain_id2 not in self.grains:
                continue
                
            grain1 = self.grains[grain_id1]
            grain2 = self.grains[grain_id2]
            
            # Count collapse events
            collapse_count = sum(1 for interaction in interactions 
                               if interaction.get('type') == 'collapse')
                               
            # Count tension events
            tension_count = sum(1 for interaction in interactions 
                              if interaction.get('type') == 'tension')
            
            # Calculate interaction density
            total_interaction_density = (collapse_count * 0.7 + tension_count * 0.3) / len(interactions)
            
            # Memory emerges when interaction density is high enough
            if total_interaction_density > 0.5 and len(interactions) >= 3:
                # Implement memory emergence through constraint adjustments
                
                # Check if this relation involves strong polarity opposing pairs
                if (abs(grain1.polarity) > 0.8 and abs(grain2.polarity) > 0.8 and
                    grain1.polarity * grain2.polarity < 0):
                    
                    # This is a potential circular pattern
                    # Record in recursion detector
                    self.recursion_detector.record_recursive_event(
                        'memory_circular_pattern',
                        {
                            'grain_id1': grain_id1,
                            'grain_id2': grain_id2,
                            'polarity1': grain1.polarity,
                            'polarity2': grain2.polarity,
                            'interaction_density': total_interaction_density,
                            'collapse_count': collapse_count
                        }
                    )
                
                # Record in relation memory
                self.relation_memory[relation_key].append({
                    'time': self.time,
                    'type': 'memory_emergence',
                    'density': total_interaction_density
                })
                
                # Check for shared ancestry between the grains
                shared_ancestry = grain1.ancestry.intersection(grain2.ancestry)
                if shared_ancestry:
                    # Record in ancestry constraint surface
                    self.ancestry_constraint.record_shared_ancestry(
                        grain_id1, grain_id2,
                        len(shared_ancestry)
                    )
    
    def _observe_system_metrics(self):
        """
        Observe system-level metrics without imposing structure.
        These metrics emerge naturally from the interactions.
        """
        # Calculate field coherence
        coherence_values = []
        
        for grain in self.grains.values():
            # Skip superposition grains
            if grain.is_in_superposition():
                continue
                
            # Get coherence if available
            coherence_values.append(grain.coherence)
        
        # Calculate average coherence
        if coherence_values:
            self.field_coherence = sum(coherence_values) / len(coherence_values)
        else:
            self.field_coherence = 0.5  # Default medium coherence
            
        # Calculate system tension
        tension_values = []
        
        for grain in self.grains.values():
            # Skip superposition grains
            if grain.is_in_superposition():
                continue
                
            # Get constraints
            tension_values.append(grain.constraint_tension)
        
        # Calculate average tension
        if tension_values:
            self.system_tension = sum(tension_values) / len(tension_values)
        else:
            self.system_tension = 0.0  # Default no tension
            
        # Calculate polarity circular coherence
        polarity_values = []
        
        for grain in self.grains.values():
            # Skip superposition grains
            if grain.is_in_superposition():
                continue
                
            # Get polarity
            polarity_values.append(grain.polarity)
        
        # Calculate circular coherence if we have values
        if len(polarity_values) >= 2:
            # Convert to angles on circle
            polarity_angles = [(p + 1.0) * PI for p in polarity_values]
            
            # Calculate circular coherence
            x_sum = sum(math.cos(angle) for angle in polarity_angles)
            y_sum = sum(math.sin(angle) for angle in polarity_angles)
            
            # Calculate mean resultant length
            r = math.sqrt(x_sum**2 + y_sum**2) / len(polarity_angles)
            
            self.circular_coherence = r
        else:
            self.circular_coherence = 1.0  # Default perfect coherence for < 2 grains
        
        # Calculate emergent symmetry level
        if self.vortices:
            # Direction symmetry - balance of clockwise and counterclockwise
            clockwise = sum(1 for v in self.vortices if v['direction'] == 'clockwise')
            counterclockwise = len(self.vortices) - clockwise
            
            direction_symmetry = 1.0 - abs(clockwise - counterclockwise) / max(1, len(self.vortices))
            
            # Polarity symmetry - balance of positive and negative
            positive = sum(1 for v in self.vortices if v['polarity'] > 0)
            negative = len(self.vortices) - positive
            
            polarity_symmetry = 1.0 - abs(positive - negative) / max(1, len(self.vortices))
            
            # Combine with weights
            self.symmetry_level = direction_symmetry * 0.5 + polarity_symmetry * 0.5
        else:
            self.symmetry_level = 0.5  # Default medium symmetry
    
    def _angular_difference(self, a: float, b: float) -> float:
        """
        Calculate angular difference between two values in polarity space [-1, 1].
        
        Args:
            a: First value
            b: Second value
            
        Returns:
            Angular difference [0-1]
        """
        # Map polarity values [-1,1] to angles [0,2π]
        angle_a = (a + 1) * PI
        angle_b = (b + 1) * PI
        
        # Calculate smallest circular distance
        diff = abs(angle_a - angle_b)
        if diff > PI:
            diff = TWO_PI - diff
            
        # Normalize to [0,1] range
        return diff / PI
    
    def _shortest_path_direction(self, a: float, b: float) -> float:
        """
        Determine the direction of the shortest path in polarity space [-1, 1].
        
        Args:
            a: First value
            b: Second value
            
        Returns:
            Direction of shortest path (1 for clockwise, -1 for counterclockwise)
        """
        # Map polarity values [-1,1] to angles [0,2π]
        angle_a = (a + 1) * PI
        angle_b = (b + 1) * PI
        
        # Calculate vector from a to b
        diff = (angle_b - angle_a) % TWO_PI
        
        # Determine direction
        if diff <= PI:
            return 1.0  # Clockwise
        else:
            return -1.0  # Counterclockwise
    
    def _angular_difference_2pi(self, a: float, b: float) -> float:
        """
        Calculate angular difference between two angles [0, 2π].
        
        Args:
            a: First angle
            b: Second angle
            
        Returns:
            Angular difference [0-π]
        """
        # Calculate smallest circular distance
        diff = abs(a - b)
        if diff > PI:
            diff = TWO_PI - diff
            
        return diff
    
    def _shortest_path_direction_2pi(self, a: float, b: float) -> float:
        """
        Determine the direction of the shortest path between angles [0, 2π].
        
        Args:
            a: First angle
            b: Second angle
            
        Returns:
            Direction of shortest path (1 for clockwise, -1 for counterclockwise)
        """
        # Calculate vector from a to b
        diff = (b - a) % TWO_PI
        
        # Determine direction
        if diff <= PI:
            return 1.0  # Clockwise
        else:
            return -1.0  # Counterclockwise
    
    def get_grain_ancestry(self, grain_id: str) -> Set[str]:
        """
        Get the ancestry set for a grain.
        
        Args:
            grain_id: ID of the grain
            
        Returns:
            Set of ancestor grain IDs
        """
        grain = self.get_grain(grain_id)
        if not grain:
            return set()
        return grain.ancestry
    
    def get_recursive_collapse_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about recursive collapses in the system.
        
        Returns:
            Dictionary with metrics about recursive collapses
        """
        # Count recursive collapses (where target has source in ancestry)
        recursive_count = 0
        self_collapse_count = 0
        field_genesis_count = 0
        
        for event in self.collapse_history:
            # Count field genesis events
            if event.get('field_genesis', False):
                field_genesis_count += 1
                
            source_id = event.get('source')
            target_id = event.get('target')
            
            if not source_id or not target_id:
                continue
                
            # Count self-collapse events
            if source_id == target_id:
                self_collapse_count += 1
                
            # Check for recursive collapse
            target_grain = self.get_grain(target_id)
            if target_grain and source_id in target_grain.ancestry:
                recursive_count += 1
        
        return {
            'recursive_collapse_count': recursive_count,
            'self_collapse_count': self_collapse_count,
            'field_genesis_count': field_genesis_count,
            'total_collapses': len(self.collapse_history)
        }
    
    def get_circular_recursion_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about circular recursion in the system.
        
        Returns:
            Dictionary with metrics about circular recursion
        """
        return {
            'circular_recursion_count': len(self.recursion_detector.circular_recursion_grains),
            'circular_events_count': len(self.recursion_detector.recursive_events),
            'circular_coherence': self.circular_coherence,
            'polarity_wraparound_events': len(self.recursive_tension.circular_events),
            'circular_ancestry_events': len(self.recursion_detector.closed_trajectories),
            'symmetry_level': self.symmetry_level
        }