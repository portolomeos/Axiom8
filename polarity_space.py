"""
Polarity Space - Core representation of directional alignment and interaction geometry

Implements the polarity space P using the Collapse Epistemology Tensor approach,
representing directional interactions and memory inheritance in the Collapse Geometry framework
without fixed vector coordinates. All structures emerge from relational dynamics.
"""

import math
import numpy as np
import random
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Callable
import uuid
from collections import defaultdict


# ===============================================================================
# Core Vectorial Utilities - Native to Phase Space
# ===============================================================================

def weighted_blend(a: float, b: float, weight: float) -> float:
    """Blend two values with weight factor."""
    return (1 - weight) * a + weight * b

def angle_blend(theta1: float, theta2: float, weight: float) -> float:
    """Blend two angles using the shortest path on the circle."""
    delta = ((theta2 - theta1 + math.pi) % (2 * math.pi)) - math.pi
    return (theta1 + weight * delta) % (2 * math.pi)

def circular_mean(angles: List[float], weights: List[float] = None) -> float:
    """Calculate circular mean of angles with optional weights."""
    if not angles:
        return 0.0
        
    # Default to equal weights if not provided
    if weights is None:
        weights = [1.0] * len(angles)
    
    # Convert to Cartesian coordinates
    x_sum = sum(math.cos(a) * w for a, w in zip(angles, weights))
    y_sum = sum(math.sin(a) * w for a, w in zip(angles, weights))
    
    # Convert back to angle
    return math.atan2(y_sum, x_sum) % (2 * math.pi)

def angular_difference(a: float, b: float) -> float:
    """Calculate minimum angular difference between two angles."""
    diff = abs(a - b) % (2 * math.pi)
    return min(diff, 2 * math.pi - diff)

def phase_tensor_product(tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
    """Multiply two phase tensors preserving circularity."""
    return np.matmul(tensor1, tensor2)


# ===============================================================================
# Field Value Classes - Represent Fundamental Field Values
# ===============================================================================

class PhaseDifference:
    """Represents a difference in phase space (Δθ, Δϕ)."""
    
    def __init__(self, delta_theta: float = 0.0, delta_phi: float = 0.0):
        self.delta_theta = delta_theta
        self.delta_phi = delta_phi
    
    def magnitude(self) -> float:
        """Calculate magnitude of phase difference."""
        return math.sqrt(self.delta_theta**2 + self.delta_phi**2)
    
    def blend(self, other: 'PhaseDifference', weight: float) -> 'PhaseDifference':
        """Blend with another phase difference."""
        return PhaseDifference(
            weighted_blend(self.delta_theta, other.delta_theta, weight),
            weighted_blend(self.delta_phi, other.delta_phi, weight)
        )
    
    def scale(self, factor: float) -> 'PhaseDifference':
        """Scale the phase difference."""
        return PhaseDifference(self.delta_theta * factor, self.delta_phi * factor)


class CollapseFlow:
    """Represents the collapse flow in phase space (Δθ, Δϕ, Δpolarity)."""
    
    def __init__(self, delta_theta: float = 0.0, delta_phi: float = 0.0, delta_polarity: float = 0.0):
        self.delta_theta = delta_theta
        self.delta_phi = delta_phi 
        self.delta_polarity = delta_polarity
    
    def magnitude(self) -> float:
        """Calculate magnitude of collapse flow."""
        return math.sqrt(self.delta_theta**2 + self.delta_phi**2 + self.delta_polarity**2)
    
    def is_structure_biased(self) -> bool:
        """Determine if flow is biased toward structure formation."""
        # Consider all components, weighted toward polarity
        weighted_sum = (self.delta_theta + self.delta_phi + self.delta_polarity * 2) / 4
        return weighted_sum > 0
    
    def is_decay_biased(self) -> bool:
        """Determine if flow is biased toward decay."""
        # Consider all components, weighted toward polarity
        weighted_sum = (self.delta_theta + self.delta_phi + self.delta_polarity * 2) / 4
        return weighted_sum < 0
    
    def scale(self, factor: float) -> 'CollapseFlow':
        """Scale the collapse flow."""
        return CollapseFlow(
            self.delta_theta * factor,
            self.delta_phi * factor,
            self.delta_polarity * factor
        )
    
    def add(self, other: 'CollapseFlow') -> 'CollapseFlow':
        """Add another collapse flow."""
        return CollapseFlow(
            self.delta_theta + other.delta_theta,
            self.delta_phi + other.delta_phi,
            self.delta_polarity + other.delta_polarity
        )
    
    def blend(self, other: 'CollapseFlow', weight: float) -> 'CollapseFlow':
        """Blend with another collapse flow."""
        return CollapseFlow(
            weighted_blend(self.delta_theta, other.delta_theta, weight),
            weighted_blend(self.delta_phi, other.delta_phi, weight),
            weighted_blend(self.delta_polarity, other.delta_polarity, weight)
        )
    
    def apply_lightlike_enhancement(self, lightlike_factor: float) -> 'CollapseFlow':
        """Enhance flow based on lightlike properties, preserving direction."""
        if lightlike_factor <= 0:
            return self
            
        # Calculate enhancement factor
        enhancement = 1.0 + lightlike_factor * 0.3
        
        # Apply to all components, preserving direction
        return self.scale(enhancement)


class PhaseTensor:
    """
    Represents a tensor in phase space, capturing directional relationships
    without reference to an external coordinate system.
    """
    
    def __init__(self, theta_theta: float = 1.0, theta_phi: float = 0.0, 
                phi_theta: float = 0.0, phi_phi: float = 1.0):
        """
        Initialize the phase tensor.
        
        Args:
            theta_theta: Component relating theta to theta
            theta_phi: Component relating theta to phi
            phi_theta: Component relating phi to theta
            phi_phi: Component relating phi to phi
        """
        # Create the 2x2 tensor
        self.tensor = np.array([
            [theta_theta, theta_phi],
            [phi_theta, phi_phi]
        ])
    
    @classmethod
    def from_numpy(cls, array: np.ndarray) -> 'PhaseTensor':
        """Create from numpy array."""
        instance = cls()
        instance.tensor = array
        return instance
    
    @classmethod
    def from_orientation(cls, orientation: float, strength: float) -> 'PhaseTensor':
        """Create from orientation angle and strength."""
        theta_theta = math.cos(orientation) * strength
        phi_phi = math.sin(orientation) * strength
        theta_phi = math.sin(orientation) * math.cos(orientation) * strength
        phi_theta = theta_phi  # Symmetric tensor
        
        return cls(theta_theta, theta_phi, phi_theta, phi_phi)
    
    def eigendecomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get eigenvalues and eigenvectors."""
        return np.linalg.eigh(self.tensor)
    
    def anisotropy(self) -> float:
        """Calculate anisotropy from eigenvalues."""
        eigenvalues, _ = self.eigendecomposition()
        if max(abs(eigenvalues)) < 0.001:
            return 0.0
            
        return abs(max(eigenvalues) - min(eigenvalues)) / max(abs(eigenvalues))
    
    def primary_direction(self) -> PhaseDifference:
        """Get primary eigenvector as a phase difference."""
        _, eigenvectors = self.eigendecomposition()
        primary_evec = eigenvectors[:, np.argmax(np.abs(self.eigendecomposition()[0]))]
        
        return PhaseDifference(primary_evec[0], primary_evec[1])
    
    def determinant(self) -> float:
        """Calculate tensor determinant."""
        return np.linalg.det(self.tensor)
    
    def trace(self) -> float:
        """Calculate tensor trace."""
        return np.trace(self.tensor)
    
    def apply_to_phase(self, phase_diff: PhaseDifference) -> PhaseDifference:
        """Apply tensor transformation to a phase difference."""
        result = np.matmul(self.tensor, np.array([phase_diff.delta_theta, phase_diff.delta_phi]))
        return PhaseDifference(result[0], result[1])


# ===============================================================================
# Core Relation Class
# ===============================================================================

class EpistemologyRelation:
    """
    Represents a relational memory connection between configurations using the
    Collapse Epistemology Tensor E(x,t) = (R, F, Φ) approach.
    
    Supports bidirectional collapse dynamics (both structure and decay).
    """
    
    def __init__(self, strength: float = 0.0, resolution: float = 0.5, 
               frustration: float = 0.0, fidelity: float = 0.5,
               orientation: float = 0.0):
        """
        Initialize a relation with epistemology components.
        
        Args:
            strength: Basic relation strength and direction, -1.0 to 1.0
                     Negative values represent decay-biased relations
            resolution: How well tension resolves, 0.0 to 1.0
            frustration: How blocked contradiction is, 0.0 to 1.0
            fidelity: How aligned collapse is with memory, 0.0 to 1.0
            orientation: Angular orientation in radians, 0.0 to 2π
        """
        self.strength = strength                   # Can be negative for decay bias
        self.resolution = self._clamp(resolution)
        self.frustration = self._clamp(frustration)
        self.fidelity = self._clamp(fidelity)
        self.orientation = orientation % (2 * math.pi)  # Ensure 0 to 2π range
        
        # Enhanced properties for bidirectional collapse dynamics
        self.lightlike_factor = 0.0                # How strongly this relation exhibits lightlike properties 
        self.flow_resistance = 0.0                 # Resistance to flow (higher saturation = higher resistance)
        self.coherence = 1.0                       # Quantum coherence of the relation (1.0 = max)
        self.cascade_potential = 0.0               # Potential to trigger cascade collapses (any direction)
        self.collapse_polarity = math.copysign(1.0, strength) if strength != 0 else 0.0  # Direction of collapse
        
        # Flow history for continuity tracking
        self._flow_history = []
        
        # Phase tensor representation
        self.phase_tensor = PhaseTensor.from_orientation(orientation, strength)
    
    def _clamp(self, value: float) -> float:
        """Clamp a value to range [0.0, 1.0]"""
        return max(0.0, min(1.0, value))
    
    def update(self, new_strength: float = None, new_resolution: float = None,
             new_frustration: float = None, new_fidelity: float = None,
             new_orientation: float = None, blending_factor: float = 0.2,
             saturation: float = None):
        """
        Update the relation with new epistemology values using smooth blending.
        
        Args:
            new_strength: Updated strength value (can be negative for decay bias)
            new_resolution: Updated resolution value
            new_frustration: Updated frustration value
            new_fidelity: Updated fidelity value
            new_orientation: Updated angular orientation
            blending_factor: How quickly the relation updates (0-1)
            saturation: Grain saturation level (constraints updates)
        """
        # Apply saturation constraint - higher saturation = less flexible relations
        effective_blending = blending_factor
        if saturation is not None:
            # Saturation reduces update flexibility (fewer degrees of freedom)
            effective_blending = blending_factor * (1.0 - saturation * 0.8)
            effective_blending = max(0.05, effective_blending)  # Ensure minimum blending
        
        # Update strength with blending
        if new_strength is not None:
            previous_strength = self.strength
            self.strength = weighted_blend(self.strength, new_strength, effective_blending)
            
            # Update collapse polarity if strength sign changes
            if (previous_strength * self.strength <= 0) and self.strength != 0:
                self.collapse_polarity = math.copysign(1.0, self.strength)
        
        # Update resolution with blending
        if new_resolution is not None:
            self.resolution = self._clamp(weighted_blend(
                self.resolution, new_resolution, effective_blending))
        
        # Update frustration with blending
        if new_frustration is not None:
            self.frustration = self._clamp(weighted_blend(
                self.frustration, new_frustration, effective_blending))
        
        # Update fidelity with blending
        if new_fidelity is not None:
            self.fidelity = self._clamp(weighted_blend(
                self.fidelity, new_fidelity, effective_blending))
        
        # Update orientation with circular blending
        if new_orientation is not None:
            self.orientation = angle_blend(
                self.orientation, new_orientation, effective_blending)
        
        # Update lightlike properties based on saturation
        if saturation is not None:
            # Low saturation = higher lightlike factor
            self.lightlike_factor = max(0.0, 1.0 - saturation * 4.0)
            
            # Flow resistance is directly related to saturation
            self.flow_resistance = saturation * 0.9
        
        # Update phase tensor
        self.phase_tensor = PhaseTensor.from_orientation(self.orientation, self.strength)
        
        # Update cascade potential
        self.update_cascade_potential()
    
    def update_cascade_potential(self):
        """Update the potential for this relation to trigger cascade collapses."""
        # Get flow tendency
        flow_tendency = self.get_flow_tendency()
        
        # Cascade potential formula: strong flow + lightlike behavior + good resolution
        self.cascade_potential = (
            abs(flow_tendency) * 0.4 +
            self.lightlike_factor * 0.3 +
            self.resolution * 0.2 +
            (1.0 - self.frustration) * 0.1
        )
        
        # Ensure minimum cascade potential for lightlike relations
        if self.lightlike_factor > 0.8 and abs(flow_tendency) > 0.3:
            self.cascade_potential = max(0.3, self.cascade_potential)
    
    def get_flow_tendency(self) -> float:
        """
        Calculate flow tendency from epistemology components.
        
        Returns:
            Flow tendency value (positive or negative)
        """
        # Calculate transmissibility from resolution and frustration
        transmissibility = self.resolution * (1.0 - self.frustration)
        
        # Calculate base flow
        base_flow = transmissibility * self.fidelity * self.strength
        
        # Apply lightlike properties to flow
        if self.lightlike_factor > 0.0:
            # Lightlike behavior enhances flow tendency (preserving sign)
            flow_sign = math.copysign(1.0, base_flow) if base_flow != 0 else 0.0
            flow_magnitude = abs(base_flow)
            
            # Enhanced flow with lightlike properties
            light_enhancement = self.lightlike_factor * 0.2
            enhanced_magnitude = flow_magnitude * (1.0 + light_enhancement)
            
            # Ensure minimum propagation for lightlike collapse cascades
            if self.lightlike_factor > 0.7 and flow_magnitude > 0:
                min_cascade_flow = 0.05 * self.lightlike_factor
                enhanced_magnitude = max(enhanced_magnitude, min_cascade_flow)
            
            # Reapply sign to get final flow
            final_flow = flow_sign * enhanced_magnitude
        else:
            # Regular non-lightlike flow
            final_flow = base_flow
        
        # Apply flow resistance from saturation
        if self.flow_resistance > 0:
            resistance_factor = 1.0 - self.flow_resistance * 0.5
            final_flow *= resistance_factor
        
        # Save to history for phase tracking
        self._flow_history.append(final_flow)
        if len(self._flow_history) > 20:
            self._flow_history.pop(0)
            
        return final_flow
    
    def get_collapse_flow(self) -> CollapseFlow:
        """
        Get collapse flow vector in phase space.
        
        Returns:
            CollapseFlow instance representing (Δθ, Δϕ, Δpolarity)
        """
        # Get flow tendency (scalar)
        flow = self.get_flow_tendency()
        
        # Calculate phase components from orientation
        theta_component = math.cos(self.orientation) * flow
        phi_component = math.sin(self.orientation) * flow
        
        # Polarity component from flow sign and strength
        polarity_component = flow * self.collapse_polarity
        
        # Create collapse flow
        collapse_flow = CollapseFlow(theta_component, phi_component, polarity_component)
        
        # Apply lightlike enhancement if applicable
        if self.lightlike_factor > 0.5:
            collapse_flow = collapse_flow.apply_lightlike_enhancement(self.lightlike_factor)
        
        return collapse_flow
    
    def get_backflow_potential(self) -> float:
        """
        Calculate backflow potential from epistemology components.
        High frustration + low resolution = backflow potential
        
        Returns:
            Backflow potential value (opposite sign of strength)
        """
        # Base backflow calculation - always opposite of strength
        base_backflow = -self.strength * self.frustration * (1.0 - self.resolution)
        
        # Apply lightlike factor for quantum backflow
        if self.lightlike_factor > 0.5 and self.frustration > 0.7:
            quantum_enhancement = self.lightlike_factor * 0.3
            base_backflow *= (1.0 + quantum_enhancement)
        
        return base_backflow
    
    def is_aligned_with(self, other: 'EpistemologyRelation', threshold: float = 0.7) -> bool:
        """
        Check if this relation is aligned with another in terms of flow direction.
        
        Args:
            other: Another relation to compare with
            threshold: Alignment threshold
            
        Returns:
            True if relations are aligned, False otherwise
        """
        # Get flow tendencies
        flow1 = self.get_flow_tendency()
        flow2 = other.get_flow_tendency()
        
        # Adjust threshold based on coherence
        effective_threshold = threshold
        avg_coherence = (self.coherence + other.coherence) / 2
        
        if avg_coherence > 0.7:
            coherence_adjustment = (avg_coherence - 0.7) * 0.5
            effective_threshold = max(0.3, threshold - coherence_adjustment)
        
        # Check alignment for significant flows
        if abs(flow1) > 0.05 and abs(flow2) > 0.05:
            if flow1 * flow2 > 0:  # Same sign = aligned
                similarity = min(abs(flow1), abs(flow2)) / max(abs(flow1), abs(flow2))
                return similarity > effective_threshold
        
        # Special case for lightlike near-zero flows
        if self.lightlike_factor > 0.8 and other.lightlike_factor > 0.8:
            if flow1 * flow2 > 0:  # Same direction, even if very small
                return True
        
        return False
    
    def get_phase_tensor(self) -> PhaseTensor:
        """Get the phase tensor representation of this relation."""
        return self.phase_tensor
    
    def __repr__(self):
        polarity = "positive" if self.strength >= 0 else "negative"
        return (f"EpistemologyRelation({polarity}, strength={self.strength:.2f}, "
              f"resolution={self.resolution:.2f}, "
              f"frustration={self.frustration:.2f}, "
              f"fidelity={self.fidelity:.2f}, "
              f"orientation={self.orientation:.2f})")


# ===============================================================================
# Field System - Unified Field Management
# ===============================================================================

class FieldSystem:
    """
    Unified system for managing all fields in the Collapse Geometry framework.
    All field properties are computed on-demand from relational data.
    """
    
    def __init__(self):
        """Initialize the field system."""
        # Relations between nodes
        self.relations = {}  # Maps (source_id, target_id) -> EpistemologyRelation
        
        # Phase positions and flows
        self.phase_positions = {}  # Maps node_id -> (theta, phi)
        self.phase_flows = {}      # Maps node_id -> (delta_theta, delta_phi)
        
        # Collapse field
        self.collapse_flows = {}   # Maps node_id -> CollapseFlow
        self.field_meta = {}       # Maps node_id -> dict of metadata
        
        # Coherence and polarity
        self.coherence = {}        # Maps node_id -> coherence value
        self.polarity = {}         # Maps node_id -> polarity value
        
        # Field topology
        self._curl_cache = {}      # Maps node_id -> curl value
        self._divergence_cache = {} # Maps node_id -> divergence value
        
        # History tracking
        self.history = defaultdict(list)  # Maps property -> list of historical values
        self.time = 0.0
    
    def get_relation(self, source_id: str, target_id: str) -> Optional[EpistemologyRelation]:
        """Get relation between two nodes."""
        return self.relations.get((source_id, target_id))
    
    def set_relation(self, source_id: str, target_id: str, relation: EpistemologyRelation):
        """Set relation between two nodes."""
        self.relations[(source_id, target_id)] = relation
        
        # Invalidate caches that depend on relations
        self._invalidate_caches(source_id)
        self._invalidate_caches(target_id)
    
    def update_relation(self, source_id: str, target_id: str, **kwargs):
        """Update relation between two nodes."""
        relation = self.get_relation(source_id, target_id)
        if relation:
            relation.update(**kwargs)
            
            # Invalidate caches that depend on relations
            self._invalidate_caches(source_id)
            self._invalidate_caches(target_id)
    
    def _invalidate_caches(self, node_id: str):
        """Invalidate cached values for a node."""
        for cache in [self._curl_cache, self._divergence_cache]:
            if node_id in cache:
                del cache[node_id]
    
    def get_phase_position(self, node_id: str) -> Tuple[float, float]:
        """Get phase position for a node."""
        # Return cached position if available
        if node_id in self.phase_positions:
            return self.phase_positions[node_id]
        
        # Initialize with random position if not set
        position = (random.random() * 2 * math.pi, random.random() * 2 * math.pi)
        self.phase_positions[node_id] = position
        return position
    
    def get_collapse_flow(self, node_id: str) -> CollapseFlow:
        """Get collapse flow for a node."""
        # Return cached flow if available
        if node_id in self.collapse_flows:
            return self.collapse_flows[node_id]
            
        # Default to zero flow
        return CollapseFlow()
    
    def get_polarity(self, node_id: str) -> float:
        """Get polarity value for a node."""
        return self.polarity.get(node_id, 0.0)
    
    def get_coherence(self, node_id: str) -> float:
        """Get coherence value for a node."""
        return self.coherence.get(node_id, 0.8)
    
    def get_lightlike_factor(self, node_id: str, saturation: float = None) -> float:
        """
        Get lightlike factor for a node based on saturation.
        
        Args:
            node_id: Node ID
            saturation: Optional saturation value (computed from node if None)
            
        Returns:
            Lightlike factor (0.0 to 1.0)
        """
        # Use provided saturation or fetch from metadata
        if saturation is None:
            saturation = self.field_meta.get(node_id, {}).get('saturation', 0.5)
            
        # Compute lightlike factor - lower saturation = higher lightlike
        return max(0.0, 1.0 - saturation * 4.0)
    
    def update_phase_positions(self, nodes: Dict[str, Any]):
        """
        Update emergent phase positions based on relational dynamics.
        
        Args:
            nodes: Dictionary of node_id -> node
        """
        # Phase accumulator for each node
        phase_accumulators = defaultdict(lambda: {'theta': 0.0, 'phi': 0.0, 'weight': 0.0})
        
        # Track phase flows
        phase_flows = {}
        
        # Process all relations
        for (source_id, target_id), relation in self.relations.items():
            if source_id not in nodes or target_id not in nodes:
                continue
                
            # Get flow tendency
            flow = relation.get_flow_tendency()
            
            # Get relation's phase tensor
            phase_tensor = relation.get_phase_tensor()
            
            # Source and target nodes
            source_node = nodes[source_id]
            target_node = nodes[target_id]
            
            # Get saturation values for lightlike enhancement
            source_saturation = getattr(source_node, 'grain_saturation', 0.5)
            target_saturation = getattr(target_node, 'grain_saturation', 0.5)
            
            # Check for lightlike properties
            source_lightlike = source_saturation < 0.2
            target_lightlike = target_saturation < 0.2
            
            # Base weight calculation
            weight = abs(flow) * 0.5 + 0.5  # Ensure some minimum weight
            
            # Phase components based on tensor
            theta_component = phase_tensor.tensor[0, 0] * flow  # θθ component
            phi_component = phase_tensor.tensor[1, 1] * flow    # φφ component
            
            # Cross-components for rotation
            theta_from_phi = phase_tensor.tensor[0, 1] * flow   # θφ component
            phi_from_theta = phase_tensor.tensor[1, 0] * flow   # φθ component
            
            # Apply lightlike enhancement
            if source_lightlike or target_lightlike:
                # Lightlike enhancement factor
                enhancement = max(
                    self.get_lightlike_factor(source_id, source_saturation),
                    self.get_lightlike_factor(target_id, target_saturation)
                )
                
                # Enhance flow components
                if enhancement > 0:
                    boost = 1.0 + enhancement * 0.3
                    theta_component *= boost
                    phi_component *= boost
                    theta_from_phi *= boost
                    phi_from_theta *= boost
            
            # Accumulate for source
            phase_accumulators[source_id]['theta'] += theta_component * weight
            phase_accumulators[source_id]['phi'] += phi_component * weight
            phase_accumulators[source_id]['weight'] += weight
            
            # Accumulate cross-components
            phase_accumulators[source_id]['theta'] += theta_from_phi * weight
            phase_accumulators[source_id]['phi'] += phi_from_theta * weight
            
            # Track flow for source
            if source_id not in phase_flows:
                phase_flows[source_id] = [0.0, 0.0]
            phase_flows[source_id][0] += theta_component
            phase_flows[source_id][1] += phi_component
            
            # Accumulate for target (opposite effect)
            phase_accumulators[target_id]['theta'] -= theta_component * weight
            phase_accumulators[target_id]['phi'] -= phi_component * weight
            phase_accumulators[target_id]['weight'] += weight
            
            # Accumulate cross-components
            phase_accumulators[target_id]['theta'] -= theta_from_phi * weight
            phase_accumulators[target_id]['phi'] -= phi_from_theta * weight
            
            # Track flow for target
            if target_id not in phase_flows:
                phase_flows[target_id] = [0.0, 0.0]
            phase_flows[target_id][0] -= theta_component
            phase_flows[target_id][1] -= phi_component
        
        # Update positions based on accumulated phase changes
        for node_id, accumulator in phase_accumulators.items():
            if accumulator['weight'] > 0:
                # Get previous position or initialize
                prev_theta, prev_phi = self.get_phase_position(node_id)
                
                # Get node saturation if available
                node = nodes.get(node_id)
                saturation = getattr(node, 'grain_saturation', 0.5)
                
                # Calculate blend factor based on saturation
                # Lower saturation = higher mobility
                base_blend = 0.2
                if saturation < 0.2:
                    # Lightlike structures move more freely
                    blend_factor = base_blend * (1.0 + (0.2 - saturation) * 2)
                else:
                    # Higher saturation constrains motion
                    blend_factor = base_blend * (1.0 - min(0.8, saturation - 0.2) * 0.7)
                
                # Normalize phase increments
                theta_increment = (accumulator['theta'] / accumulator['weight']) * blend_factor
                phi_increment = (accumulator['phi'] / accumulator['weight']) * blend_factor
                
                # Apply to previous position
                theta = (prev_theta + theta_increment) % (2 * math.pi)
                phi = (prev_phi + phi_increment) % (2 * math.pi)
                
                # Store updated position
                self.phase_positions[node_id] = (theta, phi)
                
                # Store in phase flows dictionary
                self.phase_flows[node_id] = (theta_increment, phi_increment)
        
        # Store flows in field_meta
        for node_id, (theta_flow, phi_flow) in phase_flows.items():
            if node_id not in self.field_meta:
                self.field_meta[node_id] = {}
            
            self.field_meta[node_id]['theta_flow'] = theta_flow
            self.field_meta[node_id]['phi_flow'] = phi_flow
    
    def compute_collapse_flows(self, nodes: Dict[str, Any]):
        """
        Compute collapse flows for all nodes.
        
        Args:
            nodes: Dictionary of node_id -> node
        """
        for node_id, node in nodes.items():
            # Get node saturation and polarity
            saturation = getattr(node, 'grain_saturation', 0.5)
            node_polarity = getattr(node, 'polarity', 0.0)
            
            # Components for collapse flow
            theta_flow = self.field_meta.get(node_id, {}).get('theta_flow', 0.0)
            phi_flow = self.field_meta.get(node_id, {}).get('phi_flow', 0.0)
            
            # Get ancestry curvature if available
            ancestry_curvature = getattr(node, 'ancestry_curvature', 0.0)
            
            # Create base collapse flow
            collapse_flow = CollapseFlow(theta_flow, phi_flow, node_polarity)
            
            # Apply lightlike enhancement for low saturation
            lightlike_factor = self.get_lightlike_factor(node_id, saturation)
            if lightlike_factor > 0:
                collapse_flow = collapse_flow.apply_lightlike_enhancement(lightlike_factor)
            
            # Apply ancestry curvature effect
            if ancestry_curvature > 0:
                # Curvature creates circular bending in phase space
                curvature_effect = CollapseFlow(
                    ancestry_curvature * 0.3,
                    ancestry_curvature * 0.2,
                    0.0
                )
                collapse_flow = collapse_flow.add(curvature_effect)
            
            # Store collapse flow
            self.collapse_flows[node_id] = collapse_flow
            
            # Store metadata
            self.field_meta[node_id] = {
                'saturation': saturation,
                'lightlike_factor': lightlike_factor,
                'flow_magnitude': collapse_flow.magnitude(),
                'theta_flow': theta_flow,
                'phi_flow': phi_flow
            }
            
            # Update polarity and coherence
            self.polarity[node_id] = node_polarity
            self.coherence[node_id] = getattr(node, 'phase_coherence', 0.8)
    
    def compute_field_topology(self, nodes: Dict[str, Any]):
        """
        Compute topological properties (curl, divergence) for the field.
        
        Args:
            nodes: Dictionary of node_id -> node
        """
        # Process each node
        for node_id, node in nodes.items():
            # Skip if no relations
            if not hasattr(node, 'relations'):
                continue
                
            # Get neighbors
            neighbors = list(node.relations.keys())
            if len(neighbors) < 3:
                continue
            
            # Calculate curl (circulation)
            self.compute_curl(node_id, neighbors, nodes)
            
            # Calculate divergence
            self.compute_divergence(node_id, neighbors, nodes)
    
    def compute_curl(self, node_id: str, neighbor_ids: List[str], nodes: Dict[str, Any]):
        """
        Calculate curl (circulation) around a node in phase space.
        
        Args:
            node_id: Center node ID
            neighbor_ids: List of neighbor node IDs
            nodes: Dictionary of node_id -> node
        """
        if len(neighbor_ids) < 3:
            self._curl_cache[node_id] = 0.0
            return
            
        # Get center position
        center_theta, center_phi = self.get_phase_position(node_id)
        
        # Get lightlike factor for enhanced circulation
        lightlike_factor = self.get_lightlike_factor(node_id)
        coherence = self.get_coherence(node_id)
        
        # Sort neighbors by angle around center in phase space
        neighbors_with_angle = []
        for n_id in neighbor_ids:
            if n_id not in nodes:
                continue
                
            # Get neighbor position
            n_theta, n_phi = self.get_phase_position(n_id)
            
            # Angular position relative to center
            d_theta = n_theta - center_theta
            d_phi = n_phi - center_phi
            
            # Normalize to [-π, π]
            d_theta = ((d_theta + math.pi) % (2 * math.pi)) - math.pi
            d_phi = ((d_phi + math.pi) % (2 * math.pi)) - math.pi
            
            # Calculate angle in θ-φ plane
            angle = math.atan2(d_phi, d_theta)
            neighbors_with_angle.append((n_id, angle))
        
        # Sort by angle
        neighbors_with_angle.sort(key=lambda x: x[1])
        
        # Calculate circulation
        circulation = 0.0
        
        for i in range(len(neighbors_with_angle)):
            current_id, _ = neighbors_with_angle[i]
            next_id, _ = neighbors_with_angle[(i + 1) % len(neighbors_with_angle)]
            
            # Get flows
            current_flow = self.get_collapse_flow(current_id)
            
            # Get positions
            current_theta, current_phi = self.get_phase_position(current_id)
            next_theta, next_phi = self.get_phase_position(next_id)
            
            # Calculate segment vector
            d_theta = next_theta - current_theta
            d_phi = next_phi - current_phi
            
            # Normalize to [-π, π]
            d_theta = ((d_theta + math.pi) % (2 * math.pi)) - math.pi
            d_phi = ((d_phi + math.pi) % (2 * math.pi)) - math.pi
            
            # Calculate circulation contribution (cross product in phase space)
            contribution = current_flow.delta_theta * d_phi - current_flow.delta_phi * d_theta
            circulation += contribution
        
        # Apply lightlike enhancement
        if lightlike_factor > 0.5 and coherence > 0.7:
            # Lightlike coherent structures have enhanced circulation sensitivity
            enhancement = (coherence - 0.7) * 0.5 + lightlike_factor * 0.3
            circulation *= (1.0 + enhancement)
        
        # Store in cache
        self._curl_cache[node_id] = circulation
        
        # Store in metadata
        if node_id not in self.field_meta:
            self.field_meta[node_id] = {}
        self.field_meta[node_id]['curl'] = circulation
    
    def compute_divergence(self, node_id: str, neighbor_ids: List[str], nodes: Dict[str, Any]):
        """
        Calculate divergence at a node in phase space.
        
        Args:
            node_id: Center node ID
            neighbor_ids: List of neighbor node IDs
            nodes: Dictionary of node_id -> node
        """
        if len(neighbor_ids) < 3:
            self._divergence_cache[node_id] = 0.0
            return
            
        # Get center position
        center_theta, center_phi = self.get_phase_position(node_id)
        
        # Get center flow
        center_flow = self.get_collapse_flow(node_id)
        
        # Calculate inflow and outflow
        inflow = 0.0
        outflow = 0.0
        
        for neighbor_id in neighbor_ids:
            if neighbor_id not in nodes:
                continue
                
            # Get neighbor position
            n_theta, n_phi = self.get_phase_position(neighbor_id)
            
            # Calculate phase difference
            d_theta = n_theta - center_theta
            d_phi = n_phi - center_phi
            
            # Normalize to [-π, π]
            d_theta = ((d_theta + math.pi) % (2 * math.pi)) - math.pi
            d_phi = ((d_phi + math.pi) % (2 * math.pi)) - math.pi
            
            # Calculate radial distance in phase space
            radial_distance = math.sqrt(d_theta**2 + d_phi**2)
            if radial_distance < 0.001:
                continue
                
            # Get neighbor flow
            neighbor_flow = self.get_collapse_flow(neighbor_id)
            
            # Calculate radial components
            # Project flows onto radial direction
            radial_theta = d_theta / radial_distance
            radial_phi = d_phi / radial_distance
            
            # Calculate radial flow components
            center_radial = (center_flow.delta_theta * radial_theta + 
                            center_flow.delta_phi * radial_phi)
                            
            neighbor_radial = (neighbor_flow.delta_theta * radial_theta + 
                              neighbor_flow.delta_phi * radial_phi)
            
            # Accumulate inflow and outflow
            if center_radial > 0:
                outflow += center_radial
            else:
                inflow -= center_radial
                
            if neighbor_radial < 0:
                inflow += abs(neighbor_radial)
            else:
                outflow += neighbor_radial
        
        # Divergence = outflow - inflow
        divergence = outflow - inflow
        
        # Store in cache
        self._divergence_cache[node_id] = divergence
        
        # Store in metadata
        if node_id not in self.field_meta:
            self.field_meta[node_id] = {}
        self.field_meta[node_id]['divergence'] = divergence
    
    def get_curl(self, node_id: str) -> float:
        """Get curl value for a node."""
        return self._curl_cache.get(node_id, 0.0)
    
    def get_divergence(self, node_id: str) -> float:
        """Get divergence value for a node."""
        return self._divergence_cache.get(node_id, 0.0)
    
    def identify_field_singularities(self, nodes: Dict[str, Any]):
        """
        Identify singularities in the field based on topological properties.
        
        Args:
            nodes: Dictionary of node_id -> node
            
        Returns:
            Dictionary with categorized singularities
        """
        sources = []
        sinks = []
        saddles = []
        vortices = []
        zeros = []
        
        for node_id, node in nodes.items():
            # Get topological properties
            curl = self.get_curl(node_id)
            divergence = self.get_divergence(node_id)
            
            # Get flow properties
            if node_id in self.collapse_flows:
                flow = self.collapse_flows[node_id]
                magnitude = flow.magnitude()
            else:
                magnitude = 0.0
                
            # Check for zero points
            if magnitude < 0.05:
                # Check if superposition
                is_superposition = hasattr(node, 'is_in_superposition') and node.is_in_superposition()
                zeros.append({
                    'node_id': node_id,
                    'type': 'zero',
                    'is_superposition': is_superposition
                })
                continue
            
            # Check for sources (positive divergence)
            if divergence > 0.3:
                sources.append({
                    'node_id': node_id,
                    'type': 'source',
                    'divergence': divergence,
                    'magnitude': magnitude
                })
            
            # Check for sinks (negative divergence)
            elif divergence < -0.3:
                sinks.append({
                    'node_id': node_id,
                    'type': 'sink',
                    'divergence': divergence,
                    'magnitude': magnitude
                })
            
            # Check for saddle points
            elif abs(divergence) < 0.1 and magnitude > 0.2:
                saddles.append({
                    'node_id': node_id,
                    'type': 'saddle',
                    'divergence': divergence,
                    'magnitude': magnitude
                })
            
            # Check for vortices (significant curl)
            if abs(curl) > 0.3:
                # Determine rotation direction
                rotation = "clockwise" if curl > 0 else "counterclockwise"
                
                # Check for quantization
                is_quantized = False
                winding_number = 0
                
                if self.get_coherence(node_id) > 0.7:
                    # High coherence can lead to quantized vortices
                    winding_number = round(abs(curl) / (2 * math.pi))
                    is_quantized = winding_number > 0
                
                vortices.append({
                    'node_id': node_id,
                    'type': 'vortex',
                    'curl': curl,
                    'rotation': rotation,
                    'magnitude': magnitude,
                    'is_quantized': is_quantized,
                    'winding_number': winding_number
                })
        
        return {
            'sources': sources,
            'sinks': sinks,
            'saddles': saddles,
            'vortices': vortices,
            'zeros': zeros
        }
    
    def find_cascade_pathways(self, nodes: Dict[str, Any]):
        """
        Find collapse cascade pathways in the field.
        
        Args:
            nodes: Dictionary of node_id -> node
            
        Returns:
            Dictionary with 'structure' and 'decay' pathways
        """
        # Identify singularities to use as starting points
        singularities = self.identify_field_singularities(nodes)
        
        # Define start points (sources or high magnitude)
        start_points = singularities['sources']
        
        # If no sources, use high magnitude points
        if not start_points:
            # Find high magnitude flows
            high_magnitude = []
            for node_id, node in nodes.items():
                if node_id in self.collapse_flows:
                    flow = self.collapse_flows[node_id]
                    magnitude = flow.magnitude()
                    if magnitude > 0.3:
                        high_magnitude.append({
                            'node_id': node_id,
                            'magnitude': magnitude
                        })
            
            # Sort by magnitude
            high_magnitude.sort(key=lambda x: x['magnitude'], reverse=True)
            
            # Take top points
            start_points = high_magnitude[:min(5, len(high_magnitude))]
        
        # Track pathways
        structure_pathways = []
        decay_pathways = []
        
        # Track visited nodes
        visited = set()
        
        # Build pathways from start points
        for start_point in start_points:
            start_id = start_point['node_id']
            
            if start_id in visited:
                continue
            
            # Get start flow
            start_flow = self.get_collapse_flow(start_id)
            
            # Determine if structure or decay pathway
            is_structure = start_flow.is_structure_biased()
            
            # Start new pathway
            pathway = [start_id]
            visited.add(start_id)
            current_id = start_id
            
            # Maximum path length
            max_length = 10
            
            # Extend pathway
            while len(pathway) < max_length:
                # Get current node
                current_node = nodes.get(current_id)
                if not current_node or not hasattr(current_node, 'relations'):
                    break
                    
                # Get neighbors
                neighbors = list(current_node.relations.keys())
                
                # Find best next node
                best_next = None
                best_score = 0.1  # Minimum threshold
                
                # Get current flow
                current_flow = self.get_collapse_flow(current_id)
                
                for neighbor_id in neighbors:
                    if neighbor_id in visited:
                        continue
                    
                    # Get neighbor flow
                    neighbor_flow = self.get_collapse_flow(neighbor_id)
                    
                    # Skip if no meaningful flow
                    if neighbor_flow.magnitude() < 0.05:
                        continue
                    
                    # Calculate alignment score
                    alignment_score = 0.0
                    
                    # For structure pathways, prefer structure bias
                    if is_structure and neighbor_flow.is_structure_biased():
                        alignment_score += 0.3
                    # For decay pathways, prefer decay bias
                    elif not is_structure and neighbor_flow.is_decay_biased():
                        alignment_score += 0.3
                    
                    # Calculate flow direction continuity
                    # This ensures path doesn't zigzag
                    continuity_score = 0.0
                    
                    # Check theta component
                    if abs(current_flow.delta_theta) > 0.01 and abs(neighbor_flow.delta_theta) > 0.01:
                        if current_flow.delta_theta * neighbor_flow.delta_theta > 0:
                            continuity_score += 0.2
                    
                    # Check phi component
                    if abs(current_flow.delta_phi) > 0.01 and abs(neighbor_flow.delta_phi) > 0.01:
                        if current_flow.delta_phi * neighbor_flow.delta_phi > 0:
                            continuity_score += 0.2
                    
                    # Check polarity component
                    if abs(current_flow.delta_polarity) > 0.01 and abs(neighbor_flow.delta_polarity) > 0.01:
                        if current_flow.delta_polarity * neighbor_flow.delta_polarity > 0:
                            continuity_score += 0.1
                    
                    # Check for lightlike factor
                    lightlike_score = self.get_lightlike_factor(neighbor_id)
                    
                    # Combined score
                    score = alignment_score + continuity_score + lightlike_score * 0.3
                    
                    if score > best_score:
                        best_score = score
                        best_next = neighbor_id
                
                if best_next:
                    # Continue pathway
                    pathway.append(best_next)
                    visited.add(best_next)
                    current_id = best_next
                else:
                    # End of pathway
                    break
            
            # Add pathway if significant
            if len(pathway) >= 3:
                # Calculate average properties
                avg_flow = CollapseFlow()
                lightlike_count = 0
                
                for path_id in pathway:
                    # Add flow components
                    if path_id in self.collapse_flows:
                        flow = self.collapse_flows[path_id]
                        avg_flow = avg_flow.add(flow)
                    
                    # Count lightlike nodes
                    lightlike_factor = self.get_lightlike_factor(path_id)
                    if lightlike_factor > 0.5:
                        lightlike_count += 1
                
                # Normalize average flow
                avg_flow = avg_flow.scale(1.0 / len(pathway)) if pathway else CollapseFlow()
                
                # Calculate lightlike ratio
                lightlike_ratio = lightlike_count / len(pathway) if pathway else 0.0
                
                # Create pathway entry
                pathway_entry = {
                    'nodes': pathway,
                    'length': len(pathway),
                    'type': 'structure' if is_structure else 'decay',
                    'avg_flow': (avg_flow.delta_theta, avg_flow.delta_phi, avg_flow.delta_polarity),
                    'lightlike_ratio': lightlike_ratio,
                    'start_node': start_id,
                    'end_node': pathway[-1] if pathway else None
                }
                
                # Add to appropriate list
                if is_structure:
                    structure_pathways.append(pathway_entry)
                else:
                    decay_pathways.append(pathway_entry)
        
        return {
            'structure': structure_pathways,
            'decay': decay_pathways
        }
    
    def advance_time(self, dt: float = 1.0, nodes: Dict[str, Any] = None):
        """
        Advance the field system by a time step.
        
        Args:
            dt: Time delta
            nodes: Dictionary of node_id -> node
        """
        self.time += dt
        
        # Skip update if no nodes provided
        if not nodes:
            return
        
        # Update phase positions
        self.update_phase_positions(nodes)
        
        # Compute collapse flows
        self.compute_collapse_flows(nodes)
        
        # Compute field topology
        self.compute_field_topology(nodes)


# ===============================================================================
# Factory Functions
# ===============================================================================

def create_epistemology_relation(strength=0.0, resolution=0.5, frustration=0.0, fidelity=0.5, orientation=0.0):
    """Create a new epistemology relation with given parameters."""
    return EpistemologyRelation(strength, resolution, frustration, fidelity, orientation)

def create_field_system():
    """Create a new unified field system."""
    return FieldSystem()

def create_collapse_flow(delta_theta=0.0, delta_phi=0.0, delta_polarity=0.0):
    """Create a new collapse flow vector."""
    return CollapseFlow(delta_theta, delta_phi, delta_polarity)

def create_phase_tensor(orientation=0.0, strength=1.0):
    """Create a new phase tensor from orientation and strength."""
    return PhaseTensor.from_orientation(orientation, strength)

def is_lightlike(field_system: FieldSystem, node_id: str):
    """Determine if a node exhibits lightlike behavior in the field system."""
    lightlike_factor = field_system.get_lightlike_factor(node_id)
    return lightlike_factor > 0.5

def is_structure_biased(field_system: FieldSystem, node_id: str):
    """Determine if a node is biased toward structure formation."""
    flow = field_system.get_collapse_flow(node_id)
    return flow.is_structure_biased()

def is_decay_biased(field_system: FieldSystem, node_id: str):
    """Determine if a node is biased toward decay."""
    flow = field_system.get_collapse_flow(node_id)
    return flow.is_decay_biased()

# Added by fix script
class PolarityField:
    """
    Represents the polarity field P in Collapse Geometry.
    This manages polarity relationships between grains.
    """
    
    def __init__(self):
        """Initialize the polarity field."""
        self.polarities = {}  # Maps node_id -> (strength, direction)
    
    def update_polarity(self, node_id, strength, direction):
        """
        Update polarity for a node.
        
        Args:
            node_id: Node ID
            strength: Polarity strength (0.0 to 1.0)
            direction: Polarity direction (-1.0 to 1.0)
        """
        self.polarities[node_id] = (strength, direction)
    
    def get_polarity(self, node_id):
        """
        Get polarity for a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            Tuple of (strength, direction) or (0.0, 0.0) if not found
        """
        return self.polarities.get(node_id, (0.0, 0.0))