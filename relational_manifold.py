"""
RelationalManifold - Orchestrates the emergence of geometry from relation

The RelationalManifold serves as the orchestration layer where the intrinsic dynamics of 
ConfigSpace and PolaritySpace naturally manifest and integrate. It delegates detailed 
implementation to specialized spaces while coordinating their interactions to enable
emergent geometry from pure relation.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Callable, Union
import random
import uuid
from collections import defaultdict

# Fix imports to match your directory structure
from axiom8.collapse_rules.grain_dynamics import (
    Grain, RelationalGrainSystem, create_random_grain, 
    calculate_grain_saturation, is_in_superposition, 
    calculate_degrees_of_freedom, resolve_from_superposition
)
from axiom8.collapse_rules.config_space import (
    ConfigurationSpace, ConfigurationPoint, 
    calculate_emergent_distance, topological_agreement
)
from axiom8.collapse_rules.polarity_space import (
    PolarityField, EpistemologyRelation, create_epistemology_relation,
    CollapseFlow, PhaseDifference, PhaseTensor, circular_mean, angular_difference
)

# Optional import for emergent field rules - handle with try-except
try:
    from axiom8.collapse_rules.emergent_field_rules import (
        calculate_vectorial_circulation, identify_vortex_lines,
        calculate_polarity_gradient, detect_polarity_domains,
        identify_domain_interfaces, calculate_collapse_direction_field,
        trace_field_lines, detect_cascade_paths
    )
    FIELD_RULES_AVAILABLE = True
except ImportError:
    FIELD_RULES_AVAILABLE = False


class ToroidalCoordinator:
    """
    Coordinates the toroidal structure between configuration space and polarity space.
    Acts as a thin synchronization layer without storing duplicate data.
    """
    
    def __init__(self, manifold):
        """
        Initialize with reference to the relational manifold
        
        Args:
            manifold: Parent RelationalManifold instance
        """
        self.manifold = manifold
        self.config_space = manifold.config_space
        self.polarity_field = manifold.polarity_field
        
        # Track emergent structures without duplicating data
        self.vortices = []
        self.lightlike_pathways = {
            'structure': [],  # Structure-building pathways
            'decay': []       # Structure-decaying pathways
        }
        
        # Minimum flux for lightlike detection
        self.minimum_flux = 0.01  # Even tiny flows matter for lightlike grains
    
    def synchronize_coordinates(self, grain_id: str) -> bool:
        """
        Ensure toroidal coordinates are synchronized between spaces
        
        Args:
            grain_id: Grain ID to synchronize
            
        Returns:
            True if successful, False otherwise
        """
        # Get grain and corresponding point
        grain = self.manifold.get_grain(grain_id)
        point = self.config_space.get_point(grain_id)
        
        if not grain or not point:
            return False
            
        # Get toroidal coordinates
        theta, phi = self._get_toroidal_coordinates(point)
        
        # Update grain directly
        self._update_grain_toroidal_position(grain, theta, phi)
                
        return True
    
    def _get_toroidal_coordinates(self, point: ConfigurationPoint) -> Tuple[float, float]:
        """
        Get toroidal coordinates from a configuration point.
        Uses the proper methods from config_space if available.
        
        Args:
            point: Configuration point
            
        Returns:
            Tuple of (theta, phi) coordinates
        """
        # Try to use direct method if available
        if hasattr(point, 'get_toroidal_coordinates'):
            return point.get_toroidal_coordinates()
            
        # Fallback: derive from config_space
        if hasattr(self.config_space, 'derive_phase'):
            theta = self.config_space.derive_phase(point, 'theta', self.config_space)
            phi = self.config_space.derive_phase(point, 'phi', self.config_space)
            return (theta, phi)
        
        # Second fallback: use any phase relations and derive
        if hasattr(point, 'phase_relations') and point.phase_relations:
            # Take a weighted average of phase relations
            theta_diffs = []
            phi_diffs = []
            weights = []
            
            for related_id, (theta_diff, phi_diff) in point.phase_relations.items():
                if related_id in self.config_space.points:
                    relation_strength = abs(point.relations.get(related_id, 0.0))
                    if relation_strength > 0.01:
                        theta_diffs.append(theta_diff)
                        phi_diffs.append(phi_diff)
                        weights.append(relation_strength)
            
            # Calculate average if we have relations
            if theta_diffs:
                avg_theta_diff = circular_mean(theta_diffs, weights)
                avg_phi_diff = circular_mean(phi_diffs, weights)
                
                # Use an arbitrary reference point - we care about relative positions
                return (avg_theta_diff, avg_phi_diff)
        
        # Final fallback: use random but deterministic values
        # Based on the point's ID for consistency
        seed = hash(point.id) % 1000 / 1000.0
        theta = seed * 2 * math.pi
        phi = (seed * 0.7) * 2 * math.pi  # Different factor to avoid alignment
        
        return (theta, phi)
    
    def _update_grain_toroidal_position(self, grain: Grain, theta: float, phi: float):
        """
        Update a grain's toroidal position.
        Uses the proper methods if available.
        
        Args:
            grain: Grain to update
            theta: Theta coordinate
            phi: Phi coordinate
        """
        # Use direct method if available
        if hasattr(grain, 'update_toroidal_position'):
            grain.update_toroidal_position(theta, phi)
            return
            
        # Fallback: set attributes directly
        grain.theta = theta
        grain.phi = phi
                
    def update_flow_dynamics(self):
        """
        Update flow dynamics by reading from both spaces
        and ensuring consistency
        """
        # Find lightlike pathways using the manifold's method which delegates to grain system
        config_pathways = self.manifold.find_collapse_cascade_pathways()
        
        # Enhance pathways with polarity data
        self._enhance_lightlike_pathways(config_pathways)
        
    def _enhance_lightlike_pathways(self, config_pathways):
        """
        Enhance lightlike pathways with additional metadata
        from both spaces
        
        Args:
            config_pathways: Pathways from configuration space
        """
        # Enhance with polarity dynamics
        for pathway_type in ['structure', 'decay']:
            pathways = config_pathways.get(pathway_type, [])
            enhanced_pathways = []
            
            for pathway in pathways:
                nodes = pathway.get('grains', [])  # Use 'grains' key from grain_dynamics
                if not nodes:
                    nodes = pathway.get('nodes', [])  # Fallback to 'nodes' key
                
                # Calculate average polarity and coherence
                avg_polarity = 0.0
                avg_coherence = 0.0
                valid_nodes = 0
                
                for node_id in nodes:
                    # Get polarity from grain
                    grain = self.manifold.get_grain(node_id)
                    if grain:
                        # Sum values
                        avg_polarity += getattr(grain, 'polarity', 0.0)
                        avg_coherence += getattr(grain, 'coherence', 0.5)
                        valid_nodes += 1
                
                # Calculate averages
                if valid_nodes:
                    avg_polarity /= valid_nodes
                    avg_coherence /= valid_nodes
                
                # Create enhanced pathway
                enhanced_pathway = {
                    'nodes': nodes,
                    'length': len(nodes),
                    'avg_polarity': avg_polarity,
                    'avg_coherence': avg_coherence,
                    'type': pathway_type,
                    'pathway_strength': abs(avg_polarity) * (len(nodes) / 10),
                    'lightlike_ratio': pathway.get('lightlike_ratio', 0.0)
                }
                
                enhanced_pathways.append(enhanced_pathway)
            
            # Store enhanced pathways
            self.lightlike_pathways[pathway_type] = enhanced_pathways
    
    def identify_emergent_structures(self):
        """
        Identify emergent structures like vortices by
        reading from source spaces
        """
        # Find vortices in the toroidal structure
        self.vortices = []
        
        # Loop through points in configuration space
        for point_id, point in self.config_space.points.items():
            # Skip points with few connections
            if not hasattr(point, 'relations') or len(point.relations) < 3:
                continue
                
            # Get coordinates from config space
            theta, phi = self._get_toroidal_coordinates(point)
            
            # Get grain for lightlike check
            grain = self.manifold.get_grain(point_id)
            if not grain:
                continue
                
            # Is the grain lightlike? (low saturation = lightlike behavior)
            is_lightlike = getattr(grain, 'grain_saturation', 0.5) < 0.2
            
            # Get polarity
            polarity = getattr(grain, 'polarity', 0.0)
            
            # Calculate circulation using appropriate method
            circulation = self._calculate_circulation(point_id)
            
            # Skip insignificant circulation
            if abs(circulation) < 0.3:
                continue
                
            # Create vortex structure
            vortex = {
                'center_id': point_id,
                'circulation': circulation,
                'theta': theta,
                'phi': phi,
                'strength': abs(circulation),
                'direction': 'clockwise' if circulation > 0 else 'counterclockwise',
                'is_lightlike': is_lightlike,
                'polarity': polarity
            }
            
            # Add vortex to list
            self.vortices.append(vortex)
        
        # Sort vortices by strength
        self.vortices.sort(key=lambda v: v['strength'], reverse=True)
    
    def _calculate_circulation(self, center_id: str) -> float:
        """
        Calculate circulation around a center point
        
        Args:
            center_id: Center point ID
            
        Returns:
            Circulation value
        """
        # Get center point
        center_point = self.config_space.get_point(center_id)
        if not center_point:
            return 0.0
        
        # Get neighbor ids based on what's available
        neighbor_ids = []
        if hasattr(center_point, 'connections'):
            neighbor_ids = list(center_point.connections.keys())
        elif hasattr(center_point, 'relations'):
            neighbor_ids = list(center_point.relations.keys())
        
        if len(neighbor_ids) < 3:
            return 0.0
            
        # Try different methods for calculating circulation
        
        # Method 1: Use emergent_field_rules if available
        if FIELD_RULES_AVAILABLE:
            try:
                circulation_vector = calculate_vectorial_circulation(
                    self.config_space, center_id, neighbor_ids)
                # Return the z component as scalar circulation
                return circulation_vector[2]
            except:
                pass
        
        # Method 2: Delegate to configuration space if it has this method
        if hasattr(self.config_space, 'calculate_circulation'):
            try:
                return self.config_space.calculate_circulation(center_id, neighbor_ids, {})
            except:
                pass
        
        # Method 3: Calculate circulation manually
        try:
            return self._manual_circulation_calculation(center_id, neighbor_ids)
        except:
            # Fallback to zero if all methods fail
            return 0.0
    
    def _manual_circulation_calculation(self, center_id: str, neighbor_ids: List[str]) -> float:
        """
        Manually calculate circulation when no specialized method is available
        
        Args:
            center_id: Center point ID
            neighbor_ids: List of neighbor IDs
            
        Returns:
            Circulation value
        """
        # Get center coordinates
        center_point = self.config_space.get_point(center_id)
        center_theta, center_phi = self._get_toroidal_coordinates(center_point)
        
        # Get neighbors with their angular position
        neighbors_with_angle = []
        for n_id in neighbor_ids:
            neighbor = self.config_space.get_point(n_id)
            if neighbor:
                n_theta, n_phi = self._get_toroidal_coordinates(neighbor)
                
                # Calculate angle from center in the theta-phi plane
                d_theta = n_theta - center_theta
                d_phi = n_phi - center_phi
                
                # Normalize to [-π, π]
                d_theta = ((d_theta + math.pi) % (2 * math.pi)) - math.pi
                d_phi = ((d_phi + math.pi) % (2 * math.pi)) - math.pi
                
                # Calculate angle in θ-φ plane
                angle = math.atan2(d_phi, d_theta)
                neighbors_with_angle.append((n_id, angle))
        
        # Need at least 3 neighbors
        if len(neighbors_with_angle) < 3:
            return 0.0
        
        # Sort by angle
        neighbors_with_angle.sort(key=lambda x: x[1])
        
        # Calculate circulation as line integral around the loop
        circulation = 0.0
        
        for i in range(len(neighbors_with_angle)):
            current_id, _ = neighbors_with_angle[i]
            next_id, _ = neighbors_with_angle[(i + 1) % len(neighbors_with_angle)]
            
            # Get grain properties
            current_grain = self.manifold.get_grain(current_id)
            next_grain = self.manifold.get_grain(next_id)
            
            if not current_grain or not next_grain:
                continue
                
            # Get points
            current_point = self.config_space.get_point(current_id)
            next_point = self.config_space.get_point(next_id)
            
            # Get field values (awareness, polarity)
            current_awareness = getattr(current_grain, 'awareness', 0.0)
            next_awareness = getattr(next_grain, 'awareness', 0.0)
            current_polarity = getattr(current_grain, 'polarity', 0.0)
            next_polarity = getattr(next_grain, 'polarity', 0.0)
            
            # Get positions
            current_theta, current_phi = self._get_toroidal_coordinates(current_point)
            next_theta, next_phi = self._get_toroidal_coordinates(next_point)
            
            # Calculate segment vector (change in coordinates)
            d_theta = next_theta - current_theta
            d_phi = next_phi - current_phi
            
            # Normalize to [-π, π]
            d_theta = ((d_theta + math.pi) % (2 * math.pi)) - math.pi
            d_phi = ((d_phi + math.pi) % (2 * math.pi)) - math.pi
            
            # Calculate field components (simple combination)
            current_field_theta = current_awareness * 0.5 + current_polarity * 0.5
            current_field_phi = current_awareness * 0.5 - current_polarity * 0.5
            
            # Calculate circulation contribution (cross product in phase space)
            # f_θ*dφ - f_φ*dθ
            contribution = current_field_theta * d_phi - current_field_phi * d_theta
            circulation += contribution
        
        return circulation
    
    def calculate_global_coherence(self) -> float:
        """
        Calculate global coherence by reading from both spaces
        
        Returns:
            Global coherence value (0.0 to 1.0)
        """
        # Get coherence values from configuration space and grains
        coherence_values = []
        
        for point_id, point in self.config_space.points.items():
            # Try getting coherence from point
            if hasattr(point, 'coherence'):
                coherence_values.append(point.coherence)
                continue
                
            # Try getting coherence from grain
            grain = self.manifold.get_grain(point_id)
            if grain and hasattr(grain, 'coherence'):
                coherence_values.append(grain.coherence)
        
        if not coherence_values:
            return 0.5  # Default medium coherence
        
        # Calculate average coherence
        avg_coherence = sum(coherence_values) / len(coherence_values)
        
        return min(1.0, avg_coherence)


class RelationalManifold:
    """
    Environment where dynamics naturally emerge according to their intrinsic nature.
    
    Acts as a lightweight orchestration layer that coordinates the interactions between
    spaces without duplicating their functionality.
    """
    
    def __init__(self):
        """Initialize the manifold as a coordination layer"""
        # Specialized spaces for detailed implementations
        self.config_space = ConfigurationSpace()
        self.polarity_field = PolarityField()
        
        # Environment properties
        self.time = 0.0
        self.collapse_history = []
        
        # Relational grain tracking - minimal required state
        self.grains = {}  # Maps grain_id -> Grain
        
        # Integration with engine
        self.integrated_engine = None
        
        # Minimal metrics
        self.total_collapses = 0
        self.field_coherence = 0.5
        self.system_tension = 0.0
        
        # Relational memory tracking - essential for emergence
        self.relation_memory = defaultdict(list)  # Maps (grain_id1, grain_id2) -> list of interactions
        
        # Minimal tracking for opposites
        self.opposite_pairs = []
        
        # Superposition statistics tracking
        self.superposition_count = 0
        self.first_collapse_triggered = False
        
        # Create toroidal coordinator to synchronize spaces
        self.toroidal_coordinator = ToroidalCoordinator(self)
        
        # Create a grain system for aggregate operations
        self.grain_system = RelationalGrainSystem()
        self.grain_system.time = self.time
    
    def get_grain(self, grain_id: str) -> Optional[Grain]:
        """
        Get a grain by ID.
        
        Args:
            grain_id: ID of the grain to get
            
        Returns:
            Grain object or None if not found
        """
        return self.grains.get(grain_id)
    
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
            
        # Create grain in relational space, starting in superposition (zero = infinite potential)
        grain = create_random_grain(grain_id)
        
        # Ensure the grain is in superposition state
        grain.unbounded_potential = True
        grain.awareness = 0.0  # Zero awareness = infinite potential
        grain.grain_saturation = 0.0  # No saturation = maximum freedom
        grain.grain_activation = 0.0  # No activation yet
        grain.collapse_metric = 0.0  # No collapse history
        
        # Initialize with neutral polarity 
        grain.polarity = 0.0
        
        # Initialize ancestry as an empty set
        grain.ancestry = set()
        
        # Store the grain
        self.grains[grain_id] = grain
        
        # Add corresponding point to configuration space
        config_point = ConfigurationPoint(grain_id)
        self.config_space.points[grain_id] = config_point
        
        # Initialize polarity in polarity field
        self.polarity_field.update_polarity(grain_id, 0.0, 0.0)
        
        # Update superposition count
        self.superposition_count += 1
        
        # Add to grain system
        self.grain_system.grains[grain_id] = grain
        
        return grain
    
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
        if hasattr(grain1, 'update_relation'):
            grain1.update_relation(grain_id2, relation_strength)
        else:
            if not hasattr(grain1, 'relations'):
                grain1.relations = {}
            grain1.relations[grain_id2] = relation_strength
            
        if hasattr(grain2, 'update_relation'):
            grain2.update_relation(grain_id1, relation_strength)
        else:
            if not hasattr(grain2, 'relations'):
                grain2.relations = {}
            grain2.relations[grain_id1] = relation_strength
        
        # Connect in configuration space
        point1 = self.config_space.get_point(grain_id1) or self.config_space.points.get(grain_id1)
        point2 = self.config_space.get_point(grain_id2) or self.config_space.points.get(grain_id2)
        
        if point1 and point2:
            # Update relations in config points
            if hasattr(point1, 'update_relation'):
                point1.update_relation(grain_id2, relation_strength)
            else:
                if not hasattr(point1, 'relations'):
                    point1.relations = {}
                point1.relations[grain_id2] = relation_strength
                
            if hasattr(point2, 'update_relation'):
                point2.update_relation(grain_id1, relation_strength)
            else:
                if not hasattr(point2, 'relations'):
                    point2.relations = {}
                point2.relations[grain_id1] = relation_strength
            
            # Update neighborhoods
            if hasattr(self.config_space, 'neighborhoods'):
                if grain_id1 not in self.config_space.neighborhoods:
                    self.config_space.neighborhoods[grain_id1] = set()
                if grain_id2 not in self.config_space.neighborhoods:
                    self.config_space.neighborhoods[grain_id2] = set()
                    
                self.config_space.neighborhoods[grain_id1].add(grain_id2)
                self.config_space.neighborhoods[grain_id2].add(grain_id1)
            
            # Update phase relations if needed
            self._update_phase_relations(grain_id1, grain_id2)
        
        # Create polarity relation in PolarityField
        # Translation of relation strength to polarity direction
        direction = math.copysign(1.0, relation_strength) if relation_strength != 0 else 0.0
        
        # Create new epistemology relation for polarity field
        relation = create_epistemology_relation(
            strength=relation_strength,
            resolution=0.5 + abs(relation_strength) * 0.2,
            frustration=0.2,
            fidelity=0.5,
            orientation=random.random() * 2 * math.pi
        )
        
        # Propagate polarity effects
        self.polarity_field.update_polarity(
            grain_id1, 
            abs(relation_strength),  # Strength
            direction               # Direction
        )
        
        self.polarity_field.update_polarity(
            grain_id2, 
            abs(relation_strength),  # Strength
            direction               # Direction
        )
        
        # Also connect in grain system
        self.grain_system.connect_grains(grain_id1, grain_id2, relation_strength)
        
        # Strong relation can potentially trigger collapse from superposition
        if abs(relation_strength) > 0.8:
            # Check if either grain is in superposition and potentially collapse
            if grain1.is_in_superposition() and random.random() < 0.3:
                self._potentially_resolve_from_superposition(grain_id1)
            if grain2.is_in_superposition() and random.random() < 0.3:
                self._potentially_resolve_from_superposition(grain_id2)
    
    def _update_phase_relations(self, grain_id1: str, grain_id2: str):
        """
        Update phase relations between two points in config space
        
        Args:
            grain_id1: First grain ID
            grain_id2: Second grain ID
        """
        # Get points
        point1 = self.config_space.get_point(grain_id1) or self.config_space.points.get(grain_id1)
        point2 = self.config_space.get_point(grain_id2) or self.config_space.points.get(grain_id2)
        
        if not point1 or not point2:
            return
        
        # Get toroidal coordinates
        theta1, phi1 = self.toroidal_coordinator._get_toroidal_coordinates(point1)
        theta2, phi2 = self.toroidal_coordinator._get_toroidal_coordinates(point2)
        
        # Calculate phase differences
        theta_diff = angular_difference(theta1, theta2)
        phi_diff = angular_difference(phi1, phi2)
        
        # Determine signs based on shortest path direction
        theta_sign = 1 if (theta2 - theta1 + 2*math.pi) % (2*math.pi) < math.pi else -1
        phi_sign = 1 if (phi2 - phi1 + 2*math.pi) % (2*math.pi) < math.pi else -1
        
        # Apply signs to create directed relations
        theta_relation = theta_diff * theta_sign
        phi_relation = phi_diff * phi_sign
        
        # Update phase relations
        if not hasattr(point1, 'phase_relations') or not isinstance(point1.phase_relations, dict):
            point1.phase_relations = {}
        if not hasattr(point2, 'phase_relations') or not isinstance(point2.phase_relations, dict):
            point2.phase_relations = {}
            
        # Update relations
        point1.phase_relations[point2.id] = (theta_relation, phi_relation)
        point2.phase_relations[point1.id] = (-theta_relation, -phi_relation)
    
    def set_opposite_grains(self, grain_id1: str, grain_id2: str):
        """
        Set two grains as opposites of each other.
        
        Args:
            grain_id1: First grain ID
            grain_id2: Second grain ID
        """
        # Ensure both grains exist
        if grain_id1 not in self.grains:
            self.add_grain(grain_id1)
        if grain_id2 not in self.grains:
            self.add_grain(grain_id2)
            
        grain1 = self.grains[grain_id1]
        grain2 = self.grains[grain_id2]
        
        # Set each as the opposite of the other
        if hasattr(grain1, 'set_opposite_state'):
            grain1.set_opposite_state(grain_id2)
        else:
            grain1.opposite_id = grain_id2
            
        if hasattr(grain2, 'set_opposite_state'):
            grain2.set_opposite_state(grain_id1)
        else:
            grain2.opposite_id = grain_id1
        
        # Check if pair already exists (in either order)
        pair = (grain_id1, grain_id2)
        reverse_pair = (grain_id2, grain_id1)
        
        if pair not in self.opposite_pairs and reverse_pair not in self.opposite_pairs:
            self.opposite_pairs.append(pair)
        
        # Set opposite polarity values in polarity field
        self.polarity_field.update_polarity(grain_id1, 0.8, 1.0)  # Strong positive
        self.polarity_field.update_polarity(grain_id2, 0.8, -1.0)  # Strong negative
        
        # Also connect with a strong opposing relation
        self.connect_grains(grain_id1, grain_id2, -0.8)  # Strong negative relation
        
        # Setting opposites likely causes collapse from superposition
        # Setting opposition is a form of structural commitment
        if grain1.is_in_superposition():
            self._potentially_resolve_from_superposition(grain_id1, probability=0.7)
        if grain2.is_in_superposition():
            self._potentially_resolve_from_superposition(grain_id2, probability=0.7)
    
    def _potentially_resolve_from_superposition(self, grain_id: str, probability: float = 0.5):
        """
        Potentially resolve a grain from superposition state based on probability.
        Enhanced to maintain ancestry consistency with proper source recording.
        
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
            
        # Resolve from superposition using the proper method
        if hasattr(grain, 'resolve_from_superposition'):
            awareness_level = random.uniform(0.05, 0.2)
            resolved = grain.resolve_from_superposition(awareness_level)
        else:
            # Manual fallback resolution
            awareness_level = random.uniform(0.05, 0.2)
            grain.awareness = awareness_level
            grain.unbounded_potential = False
            grain.coherence = 0.8 + random.random() * 0.2
            resolved = True
        
        # Update superposition count if successfully resolved
        if resolved and not grain.is_in_superposition():
            self.superposition_count = max(0, self.superposition_count - 1)
            
            # Initialize ancestry to include self-reference
            if not hasattr(grain, 'ancestry'):
                grain.ancestry = set()
            grain.ancestry.add(grain_id)  # Self-reference - grain is its own ancestor
            
            # Record as collapse event with proper source recording for ancestry tracking
            event = {
                'type': 'superposition_collapse',
                'time': self.time,
                'grain_id': grain_id,
                'source': grain_id,  # Self-referential source
                'target': grain_id,  # Self as target
                'field_genesis': True,  # Marker for field-level causation
                'new_awareness': grain.awareness,
                'degrees_of_freedom': getattr(grain, 'degrees_of_freedom', 1.0)
            }
            self.collapse_history.append(event)
            self.total_collapses += 1
            
            # Update configuration space
            point = self.config_space.get_point(grain_id) or self.config_space.points.get(grain_id)
            if point:
                point.unbounded_potential = False
                point.awareness = grain.awareness
                if hasattr(point, 'polarity'):
                    point.polarity = grain.polarity
            
            return True
            
        return False
    
    def find_collapse_cascade_pathways(self):
        """
        Find potential collapse cascade pathways by delegating to grain system.
        This bridges the toroidal coordinator's needs with the grain system's implementation.
        
        Returns:
            Dictionary with 'structure' and 'decay' pathway lists
        """
        # Ensure grain system is up to date
        self.grain_system.time = self.time
        
        # Copy our grains to the system if needed
        for grain_id, grain in self.grains.items():
            self.grain_system.grains[grain_id] = grain
        
        # Update neighborhoods in the grain system
        self.grain_system.update_neighborhoods()
        
        # Call implementation in the grain system
        cascade_paths = self.grain_system.find_collapse_cascade_pathways()
        
        # Format for toroidal coordinator
        structure_paths = []
        decay_paths = []
        
        for path_type, paths in cascade_paths.items():
            for path in paths:
                path_data = {
                    'nodes': path.get('grains', []),
                    'length': path.get('length', 0),
                    'avg_polarity': path.get('avg_polarity', 0.0),
                    'pathway_strength': path.get('potential', 0.0),
                    'avg_coherence': path.get('avg_coherence', 0.7),
                    'lightlike_ratio': path.get('lightlike_ratio', 0.0)
                }
                
                if path_type == 'structure':
                    structure_paths.append(path_data)
                else:
                    decay_paths.append(path_data)
        
        return {
            'structure': structure_paths,
            'decay': decay_paths
        }
    
    def get_ancestry_distribution(self):
        """
        Calculate the distribution of ancestry relationships in the manifold.
        This is crucial for understanding how relational memory emerges and
        how memory creates curvature in the field.
    
        Returns:
            Dictionary with ancestry distribution metrics
        """
        # Initialize distribution metrics
        distribution_metrics = {
            'self_reference_count': 0,        # Grains that include themselves in their ancestry (recursive)
            'ancestry_sizes': {},             # Histogram of ancestry sizes
            'shared_ancestry_matrix': {},     # Matrix of shared ancestry between grains
            'average_ancestry_size': 0.0,     # Average ancestry size across all grains
            'ancestry_depth_distribution': {}, # Distribution of ancestry depths
            'curvature_metrics': {},          # Curvature emerging from ancestry topology
            'recursive_indices': {}           # How much each grain curves back to itself
        }
    
        # Initialize counters
        total_ancestry_size = 0
        grain_count = len(self.grains)
        total_curvature = 0.0
        total_recursive_index = 0.0
    
        # Calculate ancestry sizes and self-reference count
        for grain_id, grain in self.grains.items():
            # Get ancestry set
            ancestry = grain.ancestry if hasattr(grain, 'ancestry') else set()
        
            # Calculate ancestry size
            ancestry_size = len(ancestry)
        
            # Add to total for average calculation
            total_ancestry_size += ancestry_size
        
            # Update size histogram
            if ancestry_size not in distribution_metrics['ancestry_sizes']:
                distribution_metrics['ancestry_sizes'][ancestry_size] = 0
            distribution_metrics['ancestry_sizes'][ancestry_size] += 1
        
            # Check for self-reference (recursive structure)
            is_self_referential = grain_id in ancestry
            if is_self_referential:
                distribution_metrics['self_reference_count'] += 1
        
            # Calculate recursive index based on ancestry topology
            recursive_index = 0.0
            if is_self_referential:
                # Base recursive component from direct self-reference
                recursive_index += 0.5
            
                # Enhanced recursion from transitive ancestry
                # (ancestors that reference each other)
                ancestor_pairs = [(a1, a2) for a1 in ancestry for a2 in ancestry if a1 != a2]
                transitive_count = 0
                for a1, a2 in ancestor_pairs:
                    # Check if these ancestors are related through memory
                    relation_key = (a1, a2)
                    reverse_key = (a2, a1)
                    if relation_key in self.relation_memory or reverse_key in self.relation_memory:
                        transitive_count += 1
            
                # Normalize by potential pairs and enhance recursive index
                if ancestor_pairs:
                    transitive_factor = min(0.5, transitive_count / len(ancestor_pairs))
                    recursive_index += transitive_factor
        
            # Store recursive index
            distribution_metrics['recursive_indices'][grain_id] = recursive_index
            total_recursive_index += recursive_index
        
            # Calculate ancestry depth based on relational structure
            # Depth increases with self-reference and interconnected ancestry
            ancestry_depth = 1  # Minimum depth for any grain with ancestry
        
            if is_self_referential:
                # Self-reference immediately increases depth
                ancestry_depth += 1
        
            # Additional depth from transitive relations in ancestry
            for ancestor_id in ancestry:
                if ancestor_id in self.grains:
                    ancestor = self.grains[ancestor_id]
                
                    # If ancestor has its own ancestry, increase depth
                    if hasattr(ancestor, 'ancestry') and ancestor.ancestry:
                        ancestry_depth += 1
                    
                        # Extra bonus for recursive ancestry (ancestor of ancestor)
                        if ancestor_id in ancestor.ancestry:
                            ancestry_depth += 1
        
            # Cap depth to reasonable range
            ancestry_depth = min(10, ancestry_depth)
        
            # Update depth distribution
            if ancestry_depth not in distribution_metrics['ancestry_depth_distribution']:
                distribution_metrics['ancestry_depth_distribution'][ancestry_depth] = 0
            distribution_metrics['ancestry_depth_distribution'][ancestry_depth] += 1
        
            # Calculate curvature from ancestry (memory creates curvature)
            ancestry_curvature = 0.0
        
            # Base curvature scales with ancestry size
            base_curvature = min(0.3, ancestry_size * 0.02)
        
            # Enhanced curvature from recursive structure
            recursive_curvature = recursive_index * 0.4
        
            # Enhanced curvature from ancestry depth
            depth_curvature = min(0.3, (ancestry_depth - 1) * 0.05)
        
            # Combined curvature from all factors
            ancestry_curvature = base_curvature + recursive_curvature + depth_curvature
        
            # Store curvature
            distribution_metrics['curvature_metrics'][grain_id] = ancestry_curvature
            total_curvature += ancestry_curvature
    
        # Calculate average ancestry size
        if grain_count > 0:
            distribution_metrics['average_ancestry_size'] = total_ancestry_size / grain_count
            distribution_metrics['average_recursive_index'] = total_recursive_index / grain_count
            distribution_metrics['average_curvature'] = total_curvature / grain_count
    
        # Calculate shared ancestry matrix (how much ancestry is shared between grains)
        for grain1_id, grain1 in self.grains.items():
            # Create entry for this grain
            distribution_metrics['shared_ancestry_matrix'][grain1_id] = {}
        
            for grain2_id, grain2 in self.grains.items():
                if grain1_id == grain2_id:
                    continue
                
                # Get ancestry sets
                ancestry1 = grain1.ancestry if hasattr(grain1, 'ancestry') else set()
                ancestry2 = grain2.ancestry if hasattr(grain2, 'ancestry') else set()
            
                # Calculate shared ancestry
                shared = ancestry1.intersection(ancestry2)
            
                # Store count of shared ancestors
                distribution_metrics['shared_ancestry_matrix'][grain1_id][grain2_id] = len(shared)
    
        # Create the wrapper structure expected by the engine
        return {
            'distribution': distribution_metrics,  # Wrap metrics in 'distribution' key
            'recursive_indices': distribution_metrics['recursive_indices'],
            'curvature_metrics': distribution_metrics['curvature_metrics'],
            'average_ancestry_size': distribution_metrics['average_ancestry_size'],
            'average_recursive_index': distribution_metrics['average_recursive_index'],
            'average_curvature': distribution_metrics['average_curvature'],
            'self_referential_count': distribution_metrics['self_reference_count']  # Match the expected key name
        }
    
    def manifest_dynamics(self, time_step: float = 1.0) -> Dict[str, Any]:
        """
        Allow the inherent dynamics of the system to naturally manifest.
        Orchestrates interactions between spaces without reimplementing their functionality.
        
        Args:
            time_step: Amount of time to evolve
            
        Returns:
            Dictionary with manifestation results
        """
        # Track significant events
        events = []
        
        # Check for First Collapse Axiom - check before any other dynamics
        # This implements the paradox of infinite resolution
        if self._check_first_collapse_axiom():
            # Record that the first collapse has been triggered
            self.first_collapse_triggered = True
            events.append({
                'type': 'first_collapse',
                'time': self.time,
                'description': 'First collapse triggered by paradox of infinite resolution'
            })
        
        # 1. FIELD PROPAGATION - Field naturally propagates along gradients
        self._manifest_field_propagation(time_step)
        
        # 2. NATURAL COLLAPSE - Collapse naturally occurs where conditions align
        collapse_events = self._manifest_natural_collapses()
        events.extend(collapse_events)
        
        # 3. VOID FORMATION - Tension naturally forms voids where alignment fails
        void_events = self._manifest_void_formation()
        events.extend(void_events)
        
        # 4. Update configuration space
        try:
            if hasattr(self.config_space, 'advance_time'):
                self.config_space.advance_time(time_step)
            else:
                # Fallback manual update
                self._update_configuration_space(time_step)
        except Exception as e:
            # If update fails, log error and continue
            print(f"Config space update failed: {e}")
        
        # 5. Synchronize spaces using the coordinator
        self.toroidal_coordinator.update_flow_dynamics()
        self.toroidal_coordinator.identify_emergent_structures()
        
        # 6. MEMORY EMERGENCE - Relational memory naturally emerges from interaction patterns
        self._manifest_memory_emergence()
        
        # Update time
        self.time += time_step
        self.grain_system.time = self.time
        
        # Observe system metrics
        self._observe_system_metrics()
        
        return {
            'time': self.time,
            'events': events,
            'collapses': len(collapse_events),
            'voids_formed': len(void_events),
            'coherence': self.field_coherence,
            'tension': self.system_tension,
            'superposition_count': self.superposition_count,
            'vortices': len(self.toroidal_coordinator.vortices),
            'structure_pathways': len(self.toroidal_coordinator.lightlike_pathways['structure']),
            'decay_pathways': len(self.toroidal_coordinator.lightlike_pathways['decay'])
        }
    
    def _update_configuration_space(self, time_step: float):
        """
        Manual update of configuration space when advance_time is not available
        
        Args:
            time_step: Time delta
        """
        # Update coherence if method available
        if hasattr(self.config_space, 'update_global_coherence'):
            self.config_space.update_global_coherence()
            
        # Apply quantum fluctuations
        self._apply_quantum_fluctuations(time_step)
            
        # Update energy metrics
        if hasattr(self.config_space, 'update_energy_balance'):
            self.config_space.update_energy_balance()
    
    def _apply_quantum_fluctuations(self, time_step: float):
        """
        Apply quantum fluctuations to maintain field dynamics
        
        Args:
            time_step: Time delta
        """
        # Basic probability scales with time
        base_probability = 0.02 * time_step
        
        for point_id, point in self.config_space.points.items():
            # Check if also a grain
            grain = self.grains.get(point_id)
            if not grain:
                continue
                
            # Higher probability for superposition points
            probability = base_probability
            if grain.is_in_superposition():  # FIXED: Call method instead of function
                probability *= 3.0  # 3x base for superposition
            
            # Apply fluctuation if probability check passes
            if random.random() < probability:
                # For superposition, fluctuations can resolve
                if grain.is_in_superposition():  # FIXED: Call method instead of function
                    if random.random() < 0.2:  # 20% chance for awareness fluctuation
                        # Small awareness fluctuation - may trigger collapse
                        awareness_delta = random.uniform(-0.05, 0.05)
                        
                        # Update awareness - if pushes out of zero range, may collapse
                        grain.awareness += awareness_delta
                        
                        # Check if fluctuation triggers collapse
                        if abs(grain.awareness) > 0.05 and random.random() < 0.3:
                            # Small chance of collapse from fluctuation
                            if hasattr(grain, 'resolve_from_superposition'):
                                grain.resolve_from_superposition(grain.awareness * 2)  # Amplify direction
                            else:
                                grain.unbounded_potential = False
                                grain.awareness *= 2  # Amplify direction
                            
                            # Update count if superposition resolved
                            if not grain.is_in_superposition():  # FIXED: Call method instead of function
                                self.superposition_count = max(0, self.superposition_count - 1)
                else:
                    # For regular points, fluctuations affect properties
                    fluctuation_type = random.choice(['awareness', 'polarity', 'coherence'])
                    
                    if fluctuation_type == 'awareness':
                        # Awareness fluctuation
                        awareness_delta = random.uniform(-0.05, 0.05)
                        grain.awareness = max(-1.0, min(1.0, grain.awareness + awareness_delta))
                        # Update in configuration space
                        point.awareness = grain.awareness
                    
                    elif fluctuation_type == 'polarity':
                        # Polarity fluctuation
                        polarity_delta = random.uniform(-0.05, 0.05)
                        grain.polarity = max(-1.0, min(1.0, grain.polarity + polarity_delta))
                        # Update in configuration space if it has polarity
                        if hasattr(point, 'polarity'):
                            point.polarity = grain.polarity
                    
                    elif fluctuation_type == 'coherence':
                        # Coherence fluctuation
                        coherence_delta = random.uniform(-0.05, 0.05)
                        grain.coherence = max(0.3, min(1.0, grain.coherence + coherence_delta))
                        # Update in configuration space if it has coherence
                        if hasattr(point, 'coherence'):
                            point.coherence = grain.coherence
    
    def _check_first_collapse_axiom(self) -> bool:
        """
        Check if the First Collapse Axiom should trigger.
        
        The First Collapse Axiom states that a fully unresolved field—composed 
        entirely of superpositional grains with no awareness, polarity, ancestry, 
        or saturation—is maximally unstable. The system must collapse to resolve 
        this paradox of infinite potential.
        
        Returns:
            True if first collapse triggered, False otherwise
        """
        # Skip if already triggered or if there have been collapses
        if self.first_collapse_triggered or len(self.collapse_history) > 0:
            return False
            
        # Check if system is mostly in superposition
        # "Infinite possibilities in infinite places" situation
        superposition_ratio = self.superposition_count / max(1, len(self.grains))
        
        # System must collapse if in total superposition
        if superposition_ratio > 0.9 and len(self.grains) >= 3:
            # Select a grain for first collapse (arbitrary in space but structurally inevitable)
            superposition_grains = []
            for g_id, g in self.grains.items():
                if hasattr(g, 'is_in_superposition') and g.is_in_superposition():
                    superposition_grains.append(g_id)
                elif not hasattr(g, 'is_in_superposition') and is_in_superposition(g):
                    superposition_grains.append(g_id)
            
            if superposition_grains:
                first_grain_id = random.choice(superposition_grains)
                grain = self.grains[first_grain_id]
                
                # Resolve from superposition - first emergence of structure
                awareness_level = random.uniform(0.1, 0.2)
                if hasattr(grain, 'resolve_from_superposition'):
                    grain.resolve_from_superposition(awareness_level)
                else:
                    grain.awareness = awareness_level
                    grain.unbounded_potential = False
                
                # Update superposition count
                if not grain.is_in_superposition():  # FIXED: Call method instead of function
                    self.superposition_count = max(0, self.superposition_count - 1)
                
                # Initialize ancestry to include self-reference (genesis point)
                if not hasattr(grain, 'ancestry'):
                    grain.ancestry = set()
                grain.ancestry.add(first_grain_id)  # Self-reference - grain is its own ancestor
                
                # Initialize with positive polarity bias (structure-forming)
                grain.polarity = random.uniform(0.5, 0.8)
                
                # Update in polarity field
                self.polarity_field.update_polarity(first_grain_id, 0.7, grain.polarity)
                
                # Record as special first collapse event
                event = {
                    'type': 'first_collapse',
                    'time': self.time,
                    'grain_id': first_grain_id,
                    'source': first_grain_id,  # Self-referential source for first collapse
                    'target': first_grain_id,  # Self as target
                    'field_genesis': True,     # True field-level genesis for first collapse
                    'new_awareness': grain.awareness,
                    'phase_coherence': getattr(grain, 'phase_coherence', 1.0),
                    'degrees_of_freedom': getattr(grain, 'degrees_of_freedom', 1.0),
                    'polarity': grain.polarity
                }
                self.collapse_history.append(event)
                self.total_collapses += 1
                
                # Update configuration space
                point = self.config_space.get_point(first_grain_id) or self.config_space.points.get(first_grain_id)
                if point:
                    point.awareness = grain.awareness
                    point.unbounded_potential = False
                    if hasattr(point, 'polarity'):
                        point.polarity = grain.polarity
                
                return True
                
        return False
    
    def _manifest_field_propagation(self, time_step: float):
        """
        Field naturally propagates along gradients.
        The Zero Principle naturally manifests: zero awareness acts as infinite potential.
        Orchestrates propagation using existing mechanisms in the spaces.
        
        Args:
            time_step: Amount of time to evolve
        """
        # Create temporary field values to avoid propagation bias
        awareness_updates = {}
        polarity_updates = {}
        
        # Process each grain
        for grain_id, grain in self.grains.items():
            # Skip grains with no relations
            if not hasattr(grain, 'relations') or not grain.relations:
                continue
                
            # Get corresponding configuration point
            point = self.config_space.get_point(grain_id) or self.config_space.points.get(grain_id)
            if not point:
                continue
                
            # Check if grain is in superposition
            is_superposition = grain.is_in_superposition()  # FIXED: Call method instead of function
            
            # Get current polarity
            polarity = getattr(grain, 'polarity', 0.0)
            
            # Calculate awareness update based on relations
            total_flow = 0.0
            relation_count = 0
            total_polarity_influence = 0.0
            
            for related_id, relation_strength in grain.relations.items():
                if related_id not in self.grains:
                    continue
                    
                related_grain = self.grains[related_id]
                
                # Calculate awareness gradient
                awareness_diff = related_grain.awareness - grain.awareness
                
                # Calculate polarity influence
                related_polarity = getattr(related_grain, 'polarity', 0.0)
                polarity_diff = related_polarity - polarity
                
                # Calculate flow using helper function
                flow = self._calculate_basic_flow(
                    grain, related_grain, 
                    awareness_diff, relation_strength
                )
                
                total_flow += flow
                relation_count += 1
                
                # Polarity influence is weighted by relation strength
                polarity_influence = polarity_diff * abs(relation_strength) * 0.1
                total_polarity_influence += polarity_influence
            
            # Calculate average flow per relation
            if relation_count > 0:
                avg_flow = total_flow / relation_count
                avg_polarity_influence = total_polarity_influence / relation_count
                
                # Calculate awareness update scaled by time step
                update = avg_flow * time_step
                
                # Calculate polarity update
                polarity_update = avg_polarity_influence * time_step
                
                # Special handling for superposition
                if is_superposition:
                    # Superposition doesn't directly update awareness but may resolve
                    if abs(avg_flow) > 0.3:
                        # Strong flow may cause collapse from superposition
                        collapse_probability = min(0.5, abs(avg_flow) * 0.3)
                        self._potentially_resolve_from_superposition(grain_id, collapse_probability)
                    
                    # Polarity can still be influenced even in superposition
                    polarity_updates[grain_id] = polarity_update
                else:
                    # Store update for later application
                    awareness_updates[grain_id] = update
                    polarity_updates[grain_id] = polarity_update
            
        # Apply all updates
        for grain_id, update in awareness_updates.items():
            grain = self.grains[grain_id]
            
            # Skip if grain has resolved to superposition in the meantime
            if grain.is_in_superposition():  # FIXED: Call method instead of function
                continue
                
            # Apply awareness update
            grain.awareness = max(-1.0, min(1.0, grain.awareness + update))
            
            # Sync with config space
            point = self.config_space.get_point(grain_id) or self.config_space.points.get(grain_id)
            if point:
                point.awareness = grain.awareness
                
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
            
            # Apply polarity update with constraints
            current_polarity = getattr(grain, 'polarity', 0.0)
            new_polarity = max(-1.0, min(1.0, current_polarity + update))
            grain.polarity = new_polarity
            
            # Sync with config space
            point = self.config_space.get_point(grain_id) or self.config_space.points.get(grain_id)
            if point and hasattr(point, 'polarity'):
                point.polarity = new_polarity
            
            # Update in polarity field
            self.polarity_field.update_polarity(grain_id, abs(new_polarity), new_polarity)
    
    def _calculate_basic_flow(self, grain, related_grain, awareness_diff, relation_strength):
        """
        Calculate basic flow based on simple field properties.
        For more complex calculations, delegate to configuration space.
        
        Args:
            grain: Source grain
            related_grain: Target grain
            awareness_diff: Difference in awareness
            relation_strength: Strength of relation
            
        Returns:
            Flow value
        """
        # Check for superposition states
        grain_superposition = grain.is_in_superposition()  # FIXED: Call method instead of function
        related_superposition = related_grain.is_in_superposition()  # FIXED: Call method instead of function
        
        # Calculate base flow from gradient and relation
        base_flow = awareness_diff * relation_strength
        
        # Apply natural behavior for superposition states
        if grain_superposition or related_superposition:
            # When either grain is in superposition (zero awareness):
            if grain_superposition and related_superposition:
                # Both in superposition - balanced quantum fluctuations
                fluctuation = (random.random() - 0.5) * 0.1
                flow = fluctuation
            elif grain_superposition:
                # Source in superposition - enhanced outward flow
                flow = base_flow * 1.5
            else:  # related_superposition
                # Target in superposition - enhanced inward flow
                flow = base_flow * 1.5
        else:
            # Regular flow between defined states
            flow = base_flow
        
        return flow
    
    def _manifest_natural_collapses(self) -> List[Dict[str, Any]]:
        """
        Collapse naturally occurs where readiness and structural alignment manifest.
        Delegates detailed collapse mechanics to configuration space.
        
        Returns:
            List of collapse event dictionaries
        """
        collapse_events = []
        
        # Check all grains for potential collapse readiness
        for source_id, source_grain in self.grains.items():
            # Special handling for superposition states
            if source_grain.is_in_superposition():  # FIXED: Call method instead of function
                # Superposition can spontaneously collapse with low probability
                if random.random() < 0.05:
                    if self._potentially_resolve_from_superposition(source_id, probability=0.3):
                        continue  # Already processed this grain
            
            # Skip grains with little activation unless in superposition
            if getattr(source_grain, 'grain_activation', 0.0) < 0.3 and not source_grain.is_in_superposition():  # FIXED: Call method instead of function
                continue
                
            source_point = self.config_space.get_point(source_id) or self.config_space.points.get(source_id)
            if not source_point:
                continue
                
            # Check each relation for potential collapse
            for target_id in source_grain.relations:
                # Skip incompatible collapse pairs
                if not self._is_valid_collapse_pair(source_id, target_id):
                    continue
                
                # Check natural collapse readiness
                readiness = self._calculate_collapse_readiness(source_id, target_id)
                
                # Apply polarity and other adjustments
                readiness = self._adjust_collapse_readiness(source_id, target_id, readiness)
                
                # Check structural alignment
                alignment = self._attempt_structural_alignment(source_id, target_id)
                
                # If naturally ready and aligned, collapse occurs
                if readiness > 0.7 and alignment > 0.3:
                    # Allow natural collapse to occur
                    collapse_event = self._manifest_collapse(source_id, target_id, readiness)
                    if collapse_event:
                        collapse_events.append(collapse_event)
                # If ready but not aligned, tension builds in configuration space
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
        if target_grain.is_in_superposition():  # FIXED: Call method instead of function
            return True
            
        # Skip if target already highly saturated
        if getattr(target_grain, 'grain_saturation', 0.0) > 0.9:
            return False
            
        # Skip if target not in source relations
        if not hasattr(source_grain, 'relations') or target_id not in source_grain.relations:
            return False
            
        # All checks passed
        return True
    
    def _calculate_collapse_readiness(self, source_id: str, target_id: str) -> float:
        """
        Calculate collapse readiness between two points.
        
        Args:
            source_id: First point ID
            target_id: Second point ID
            
        Returns:
            Collapse readiness value (0.0 to 1.0)
        """
        # Try using config_space method first if available
        if hasattr(self.config_space, 'collapse_readiness'):
            try:
                return self.config_space.collapse_readiness(source_id, target_id)
            except:
                # Fallback to manual calculation
                pass
        
        # Get points and grains
        source_grain = self.grains.get(source_id)
        target_grain = self.grains.get(target_id)
        
        if not source_grain or not target_grain:
            return 0.0
        
        # Special handling for superposition states
        source_superposition = source_grain.is_in_superposition()  # FIXED: Call method instead of function
        target_superposition = target_grain.is_in_superposition()  # FIXED: Call method instead of function
        
        if source_superposition or target_superposition:
            # Both in superposition - moderate readiness
            if source_superposition and target_superposition:
                return 0.7
            
            # One in superposition - high readiness (unbounded potential can collapse easily)
            return 0.9
        
        # Calculate toroidal phase alignment
        source_point = self.config_space.get_point(source_id) or self.config_space.points.get(source_id)
        target_point = self.config_space.get_point(target_id) or self.config_space.points.get(target_id)
        
        if source_point and target_point:
            # Get coordinates
            source_theta, source_phi = self.toroidal_coordinator._get_toroidal_coordinates(source_point)
            target_theta, target_phi = self.toroidal_coordinator._get_toroidal_coordinates(target_point)
            
            # Calculate phase differences
            theta_diff = angular_difference(source_theta, target_theta) / math.pi  # Normalize to [0, 1]
            phi_diff = angular_difference(source_phi, target_phi) / math.pi        # Normalize to [0, 1]
            
            # Phase alignment factor - closer phases have higher readiness
            phase_alignment = 1.0 - (theta_diff * 0.5 + phi_diff * 0.5)
        else:
            phase_alignment = 0.5  # Default middle value
        
        # Calculate polarity alignment
        source_polarity = getattr(source_grain, 'polarity', 0.0)
        target_polarity = getattr(target_grain, 'polarity', 0.0)
        polarity_product = source_polarity * target_polarity
        
        # Polarity alignment factor - same polarity direction increases readiness
        polarity_alignment = 0.5  # Neutral default
        if abs(polarity_product) > 0.1:
            if polarity_product > 0:
                # Same polarity direction enhances readiness
                polarity_alignment = 0.5 + abs(polarity_product) * 0.3
            else:
                # Opposite polarity reduces readiness
                polarity_alignment = 0.5 - abs(polarity_product) * 0.2
        
        # Field gradient factor from awareness differences
        gradient_factor = 0.5  # Neutral default
        awareness_diff = source_grain.awareness - target_grain.awareness
        if abs(awareness_diff) > 0.1:
            # Greater difference increases readiness
            gradient_factor = 0.5 + min(0.3, abs(awareness_diff) * 0.5)
        
        # Freedom factor - higher freedom allows easier collapse
        source_freedom = getattr(source_grain, 'degrees_of_freedom', 1.0 - getattr(source_grain, 'grain_saturation', 0.0))
        target_freedom = getattr(target_grain, 'degrees_of_freedom', 1.0 - getattr(target_grain, 'grain_saturation', 0.0))
        avg_freedom = (source_freedom + target_freedom) / 2
        
        # Special boost for high-freedom "lightlike" points
        lightlike_factor = 0.0
        if source_freedom > 0.8 or target_freedom > 0.8:
            lightlike_factor = 0.2
        
        # Combined readiness calculation
        readiness = (
            phase_alignment * 0.3 +
            polarity_alignment * 0.2 +
            gradient_factor * 0.2 +
            avg_freedom * 0.3
        ) + lightlike_factor
        
        return max(0.0, min(1.0, readiness))
    
    def _adjust_collapse_readiness(self, source_id: str, target_id: str, base_readiness: float) -> float:
        """
        Adjust collapse readiness based on additional factors
        
        Args:
            source_id: Source grain ID
            target_id: Target grain ID
            base_readiness: Base readiness from configuration space
            
        Returns:
            Adjusted readiness value
        """
        source_grain = self.grains.get(source_id)
        target_grain = self.grains.get(target_id)
        
        if not source_grain or not target_grain:
            return base_readiness
            
        readiness = base_readiness
        
        # Special cases for superposition states
        if target_grain.is_in_superposition():  # FIXED: Call method instead of function
            # Increase readiness to collapse into superposition
            # Implementing the idea that "zero = infinite potential"
            readiness *= 1.5
        
        # Apply polarity alignment adjustment
        if hasattr(source_grain, 'polarity') and hasattr(target_grain, 'polarity'):
            # Get polarities
            source_polarity = source_grain.polarity
            target_polarity = target_grain.polarity
            
            # Calculate polarity product (positive = aligned, negative = opposed)
            polarity_product = source_polarity * target_polarity
            
            if abs(source_polarity) > 0.3 and abs(target_polarity) > 0.3:
                # Strong polarities have more influence
                if polarity_product > 0:
                    # Aligned polarities enhance readiness
                    readiness *= (1.0 + min(0.3, abs(polarity_product)))
                else:
                    # Opposed polarities reduce readiness
                    readiness *= (1.0 - min(0.3, abs(polarity_product)))
        
        return readiness
    
    def _attempt_structural_alignment(self, source_id: str, target_id: str) -> float:
        """
        Attempt to align the structure between two configuration points.
        
        Args:
            source_id: Source point ID
            target_id: Target point ID
            
        Returns:
            Alignment factor (0.0 to 1.0) indicating alignment success
        """
        # Try using config_space method first if available
        if hasattr(self.config_space, 'attempt_structural_alignment'):
            try:
                return self.config_space.attempt_structural_alignment(source_id, target_id)
            except:
                # Fallback to manual calculation
                pass
        
        # Get points and grains
        source_grain = self.grains.get(source_id)
        target_grain = self.grains.get(target_id)
        source_point = self.config_space.get_point(source_id) or self.config_space.points.get(source_id)
        target_point = self.config_space.get_point(target_id) or self.config_space.points.get(target_id)
        
        if not source_grain or not target_grain or not source_point or not target_point:
            return 0.0
        
        # Calculate parameters for alignment
        source_saturation = getattr(source_grain, 'grain_saturation', 0.0)
        target_saturation = getattr(target_grain, 'grain_saturation', 0.0)
        source_freedom = getattr(source_grain, 'degrees_of_freedom', 1.0 - source_saturation)
        target_freedom = getattr(target_grain, 'degrees_of_freedom', 1.0 - target_saturation)
        source_polarity = getattr(source_grain, 'polarity', 0.0)
        target_polarity = getattr(target_grain, 'polarity', 0.0)
        
        # Check for superposition
        source_superposition = source_grain.is_in_superposition()  # FIXED: Call method instead of function
        target_superposition = target_grain.is_in_superposition()  # FIXED: Call method instead of function
        
        # Calculate alignment factor based on polarity agreement
        polarity_product = source_polarity * target_polarity
        if polarity_product > 0:
            # Same polarity direction improves alignment
            polarity_alignment = 0.6 + min(0.3, abs(polarity_product) * 0.3)
        else:
            # Different polarity directions reduce alignment
            polarity_alignment = max(0.2, 0.5 - abs(polarity_product) * 0.3)
        
        # Calculate freedom factor
        freedom_factor = (source_freedom + target_freedom) / 2
        
        # Calculate alignment based on phase agreement
        source_theta, source_phi = self.toroidal_coordinator._get_toroidal_coordinates(source_point)
        target_theta, target_phi = self.toroidal_coordinator._get_toroidal_coordinates(target_point)
        
        theta_agreement = 1.0 - angular_difference(source_theta, target_theta) / math.pi
        phi_agreement = 1.0 - angular_difference(source_phi, target_phi) / math.pi
        
        phase_alignment = (theta_agreement + phi_agreement) / 2
        
        # Special cases for superposition states
        if target_superposition:
            # Target in superposition makes alignment easier
            target_factor = 0.8
        else:
            # Normal alignment
            target_factor = 0.5
        
        # Calculate overall alignment
        alignment = (
            polarity_alignment * 0.4 +
            phase_alignment * 0.3 +
            freedom_factor * 0.2 +
            target_factor * 0.1
        )
        
        return max(0.0, min(1.0, alignment))
    
    def _update_structural_tension(self, source_id: str, target_id: str, tension_level: float):
        """
        Update structural tension between points
        
        Args:
            source_id: Source point ID
            target_id: Target point ID
            tension_level: Level of tension to add
        """
        # Get configuration points
        source_point = self.config_space.get_point(source_id) or self.config_space.points.get(source_id)
        target_point = self.config_space.get_point(target_id) or self.config_space.points.get(target_id)
        
        # Update tension values
        if source_point:
            if not hasattr(source_point, 'structural_tension'):
                source_point.structural_tension = 0.0
                
            # Add tension
            tension_increase = tension_level * 0.2
            source_point.structural_tension = min(1.0, source_point.structural_tension + tension_increase)
            
            # Add tensions dict if needed
            if not hasattr(source_point, 'tensions'):
                source_point.tensions = {}
            source_point.tensions[target_id] = tension_level
        
        if target_point:
            if not hasattr(target_point, 'structural_tension'):
                target_point.structural_tension = 0.0
                
            # Add tension (less for target)
            tension_increase = tension_level * 0.1
            target_point.structural_tension = min(1.0, target_point.structural_tension + tension_increase)
            
            # Add tensions dict if needed
            if not hasattr(target_point, 'tensions'):
                target_point.tensions = {}
            target_point.tensions[source_id] = tension_level * 0.5  # Lower for target
    
    def _manifest_collapse(self, source_id: str, target_id: str, readiness: float) -> Dict[str, Any]:
        """
        Natural collapse manifests between two grains.
        Coordinates the collapse across both spaces.
        
        Args:
            source_id: Source grain ID
            target_id: Target grain ID
            readiness: Collapse readiness value
            
        Returns:
            Collapse event dictionary
        """
        source_grain = self.grains[source_id]
        target_grain = self.grains[target_id]
        
        # Calculate natural collapse strength based on readiness
        collapse_strength = readiness
        
        # Determine collapse polarity (structure vs decay)
        # Positive = structure formation, negative = decay
        collapse_polarity = 1.0  # Default positive (structure formation)
        
        # Source polarity influences collapse direction
        if hasattr(source_grain, 'polarity'):
            # Strong negative polarity can create decay collapse
            if source_grain.polarity < -0.5:
                collapse_polarity = -1.0  # Negative (decay)
                # Reduce magnitude for decay
                collapse_strength *= 0.7
        
        # Check if target is in superposition - special handling
        target_superposition = target_grain.is_in_superposition()  # FIXED: Call method instead of function
        
        if target_superposition:
            # Collapse from superposition
            awareness_level = source_grain.awareness * 0.5 + 0.05
            if hasattr(target_grain, 'resolve_from_superposition'):
                target_grain.resolve_from_superposition(awareness_level)
            else:
                # Manual fallback
                target_grain.awareness = awareness_level
                target_grain.unbounded_potential = False
            
            # Update superposition count
            if not target_grain.is_in_superposition():  # FIXED: Call method instead of function
                self.superposition_count = max(0, self.superposition_count - 1)
            
            # Transfer polarity influence
            if hasattr(source_grain, 'polarity') and hasattr(target_grain, 'polarity'):
                # Target gets some of source's polarity
                random_component = random.uniform(-0.2, 0.2)
                target_grain.polarity = min(1.0, max(-1.0, 
                                          source_grain.polarity * 0.7 + random_component))
                
                # Update in polarity field
                self.polarity_field.update_polarity(
                    target_id, 
                    abs(target_grain.polarity), 
                    target_grain.polarity
                )
                
            # Update target's ancestry to include source
            if not hasattr(target_grain, 'ancestry'):
                target_grain.ancestry = set()
                
            # Add source to target's ancestry
            target_grain.ancestry.add(source_id)
            
            # Transfer source's ancestry to target for recursive memory
            if hasattr(source_grain, 'ancestry'):
                target_grain.ancestry.update(source_grain.ancestry)
        else:
            # Regular collapse to non-superposition grain
            
            # Update target grain properties based on collapse
            if collapse_polarity > 0:
                # POSITIVE COLLAPSE: STRUCTURE FORMATION
                
                # Increase saturation (memory/structure formation)
                saturation_increase = collapse_strength * 0.1
                target_grain.grain_saturation = min(1.0, target_grain.grain_saturation + saturation_increase)
                
                # Update activation (collapse energy)
                activation_increase = collapse_strength * 0.2
                target_grain.grain_activation = min(1.0, target_grain.grain_activation + activation_increase)
                
                # Polarity shifts slightly toward source
                if hasattr(target_grain, 'polarity'):
                    polarity_shift = (source_grain.polarity - target_grain.polarity) * 0.2
                    target_grain.polarity = min(1.0, max(-1.0, target_grain.polarity + polarity_shift))
                    
                    # Update in polarity field
                    self.polarity_field.update_polarity(
                        target_id, 
                        abs(target_grain.polarity), 
                        target_grain.polarity
                    )
            else:
                # NEGATIVE COLLAPSE: DECAY/VOID FORMATION
                
                # Decrease saturation (structure decay)
                saturation_decrease = collapse_strength * 0.1
                target_grain.grain_saturation = max(0.0, target_grain.grain_saturation - saturation_decrease)
                
                # Decrease activation
                activation_decrease = collapse_strength * 0.1
                target_grain.grain_activation = max(0.0, target_grain.grain_activation - activation_decrease)
                
                # Polarity shifts toward negative
                if hasattr(target_grain, 'polarity'):
                    polarity_shift = -collapse_strength * 0.2
                    target_grain.polarity = max(-1.0, target_grain.polarity + polarity_shift)
                    
                    # Update in polarity field
                    self.polarity_field.update_polarity(
                        target_id, 
                        abs(target_grain.polarity), 
                        target_grain.polarity
                    )
            
            # Record ancestry (relational memory/heredity)
            if not hasattr(target_grain, 'ancestry'):
                target_grain.ancestry = set()
                
            # Add source to target's ancestry
            target_grain.ancestry.add(source_id)
            
            # Probabilistically transfer partial ancestry from source
            # This creates ancestry trees that can capture backflow dynamics
            if hasattr(source_grain, 'ancestry'):
                # Transfer based on collapse strength
                transfer_probability = collapse_strength * 0.5
                for ancestor_id in source_grain.ancestry:
                    if random.random() < transfer_probability:
                        target_grain.ancestry.add(ancestor_id)
        
        # Record collapse in relational memory
        relation_key = (source_id, target_id)
        self.relation_memory[relation_key].append({
            'time': self.time,
            'type': 'collapse',
            'strength': collapse_strength,
            'polarity': collapse_polarity
        })
        
        # Record reverse relation memory
        reverse_key = (target_id, source_id)
        self.relation_memory[reverse_key].append({
            'time': self.time,
            'type': 'collapse_echo',
            'strength': collapse_strength * 0.5,
            'polarity': collapse_polarity
        })
        
        # Create collapse record
        event = {
            'type': 'collapse',
            'time': self.time,
            'source': source_id,
            'target': target_id,
            'strength': collapse_strength,
            'polarity': collapse_polarity,
            'target_saturation': target_grain.grain_saturation,
            'target_freedom': getattr(target_grain, 'degrees_of_freedom', 1.0 - target_grain.grain_saturation),
            'target_polarity': getattr(target_grain, 'polarity', 0.0),
            'from_superposition': target_superposition,
            'ancestry_transfer': len(getattr(source_grain, 'ancestry', set()))
        }
        
        # Add to history
        self.collapse_history.append(event)
        self.total_collapses += 1
        
        # Propagate collapse effects in configuration space
        # Apply signed collapse strength - positive for structure, negative for decay
        self._propagate_collapse_to_config_space(
            source_id, target_id, collapse_strength * collapse_polarity)
        
        return event
    
    def _propagate_collapse_to_config_space(self, source_id: str, target_id: str, signed_collapse_strength: float):
        """
        Propagate collapse effects to configuration space
        
        Args:
            source_id: Source point ID
            target_id: Target point ID
            signed_collapse_strength: Signed collapse strength (positive=structure, negative=decay)
        """
        # Try using config_space update_from_collapse method if available
        if hasattr(self.config_space, 'update_from_collapse'):
            try:
                self.config_space.update_from_collapse(source_id, target_id, signed_collapse_strength)
                return
            except:
                # Fallback to manual update
                pass
        
        # Get configuration points
        source_point = self.config_space.get_point(source_id) or self.config_space.points.get(source_id)
        target_point = self.config_space.get_point(target_id) or self.config_space.points.get(target_id)
        
        if not source_point or not target_point:
            return
        
        # Get grains
        source_grain = self.grains.get(source_id)
        target_grain = self.grains.get(target_id)
        
        if not source_grain or not target_grain:
            return
        
        # Update awareness if needed
        if hasattr(target_point, 'awareness'):
            target_point.awareness = target_grain.awareness
            
        # Update unbounded_potential if needed
        if hasattr(target_point, 'unbounded_potential'):
            target_point.unbounded_potential = target_grain.is_in_superposition()  # FIXED: Call method instead of function
            
        # Update polarity if needed
        if hasattr(target_point, 'polarity') and hasattr(target_grain, 'polarity'):
            target_point.polarity = target_grain.polarity
            
        # Update coherence if needed
        if hasattr(target_point, 'coherence') and hasattr(target_grain, 'coherence'):
            target_point.coherence = target_grain.coherence
            
        # Update collapse_metric if needed
        if hasattr(target_point, 'collapse_metric') and hasattr(target_grain, 'collapse_metric'):
            target_point.collapse_metric = target_grain.collapse_metric
            
        # Update phase relations if needed
        self._update_phase_relations(source_id, target_id)
    
    def _manifest_void_formation(self) -> List[Dict[str, Any]]:
        """
        Voids naturally form where structural tension is high.
        Delegates void formation to configuration space.
        
        Returns:
            List of void formation event dictionaries
        """
        void_events = []
        
        # Check for void formation in configuration space
        # This happens naturally when structural alignment fails
        for point_id, point in self.config_space.points.items():
            # Skip points with low structural tension
            if not hasattr(point, 'structural_tension') or point.structural_tension < 0.8:
                continue
                
            # Natural void formation threshold
            void_threshold = 0.8
            
            # Check if tension exceeded threshold
            if point.structural_tension >= void_threshold:
                # Create void formation event
                void_event = {
                    'type': 'void_formation',
                    'time': self.time,
                    'center_point': point_id,
                    'tension': point.structural_tension,
                    'void_strength': point.structural_tension - void_threshold
                }
                
                void_events.append(void_event)
                
                # Reset tension after void formation
                point.structural_tension = 0.2
                
                # Void formation shifts polarity toward negative (decay bias)
                if point_id in self.grains:
                    grain = self.grains[point_id]
                    if hasattr(grain, 'polarity'):
                        grain.polarity = max(-1.0, grain.polarity - 0.3)
                        self.polarity_field.update_polarity(
                            point_id, 
                            abs(grain.polarity), 
                            grain.polarity
                        )
                
                # Void formation can cause superposition collapse
                if point_id in self.grains and grain.is_in_superposition():  # FIXED: Call method instead of function
                    # Void tends to resolve superposition with high probability
                    self._potentially_resolve_from_superposition(point_id, probability=0.8)
        
        return void_events
    
    def _manifest_memory_emergence(self):
        """
        Relational memory naturally emerges from interaction patterns.
        Memory is not stored explicitly but emerges from the pattern of relations.
        """
        # Calculate memory metrics for each relation
        memory_strengths = {}
        
        for relation_key, interactions in self.relation_memory.items():
            if not interactions:
                continue
                
            # Calculate memory strength from interaction patterns
            total_strength = 0.0
            decay_factor = 0.9  # Memory decays with time
            
            # Recent interactions matter more
            for i, interaction in enumerate(interactions):
                # Position factor - more recent = more weight
                position_factor = decay_factor ** (len(interactions) - i - 1)
                
                # Calculate contribution based on interaction type
                if interaction['type'] == 'collapse':
                    # Collapse creates strong positive memory
                    contribution = interaction['strength'] * 1.0
                    
                    # Apply polarity direction if present
                    if 'polarity' in interaction:
                        contribution *= interaction['polarity']
                        
                elif interaction['type'] == 'collapse_echo':
                    # Echo creates weaker positive memory
                    contribution = interaction['strength'] * 0.5
                    
                    # Apply polarity direction if present
                    if 'polarity' in interaction:
                        contribution *= interaction['polarity']
                        
                elif interaction['type'] == 'tension':
                    # Tension creates negative memory
                    contribution = -interaction['strength'] * 0.3
                else:
                    contribution = 0.0
                
                # Add weighted contribution
                total_strength += contribution * position_factor
            
            # Store calculated memory strength
            memory_strengths[relation_key] = total_strength
        
        # Apply memory to grains
        for (source_id, target_id), memory_strength in memory_strengths.items():
            source_grain = self.grains.get(source_id)
            
            if source_grain:
                # Update grain's internal memory
                if hasattr(source_grain, 'update_relation_memory'):
                    source_grain.update_relation_memory(target_id, memory_strength)
                elif hasattr(source_grain, 'relation_memory'):
                    # Manual fallback
                    if not isinstance(source_grain.relation_memory, dict):
                        source_grain.relation_memory = {}
                    source_grain.relation_memory[target_id] = memory_strength
                
                # Update polarity based on memory strength
                if hasattr(source_grain, 'polarity'):
                    # Strong memory influences polarity
                    if abs(memory_strength) > 0.5:
                        # Direction of memory influences polarity
                        polarity_shift = memory_strength * 0.1
                        source_grain.polarity = max(-1.0, min(1.0, source_grain.polarity + polarity_shift))
                        
                        # Update in polarity field
                        self.polarity_field.update_polarity(
                            source_id, 
                            abs(source_grain.polarity), 
                            source_grain.polarity
                        )
                
                # Strong memory can potentially trigger collapse from superposition
                if source_grain.is_in_superposition() and abs(memory_strength) > 0.7:  # FIXED: Call method instead of function
                    self._potentially_resolve_from_superposition(source_id, probability=0.4)
        
        # Update collapse metric based on ancestry relationships
        self._update_collapse_metrics()
    
    def _update_collapse_metrics(self):
        """
        Update collapse metrics for each grain based on ancestry relationships.
        This is critical for building proper structural memory and curvature.
        """
        # Get ancestry distribution to calculate recursive indices and curvature
        ancestry_data = self.get_ancestry_distribution()
        recursive_indices = ancestry_data.get('recursive_indices', {})
        curvature_metrics = ancestry_data.get('curvature_metrics', {})
        
        for grain_id, grain in self.grains.items():
            if not hasattr(grain, 'ancestry'):
                grain.ancestry = set()
                continue
                
            # Calculate collapse metric based on ancestry size and curvature
            ancestry_size = len(grain.ancestry)
            
            # Base metric is scaled by ancestry size
            base_metric = min(1.0, ancestry_size * 0.1)
            
            # Get recursive index and curvature from ancestry distribution
            recursive_index = recursive_indices.get(grain_id, 0.0)
            ancestry_curvature = curvature_metrics.get(grain_id, 0.0)
            
            # Calculate transitive ancestry contribution
            transitive_score = 0.0
            transitive_count = 0
            
            # Look for connections between ancestors (curvature)
            ancestor_pairs = [(a1, a2) for a1 in grain.ancestry for a2 in grain.ancestry if a1 != a2]
            for a1, a2 in ancestor_pairs:
                # If there's a relation between ancestors, this creates curvature
                relation_key = (a1, a2)
                reverse_key = (a2, a1)
                if relation_key in self.relation_memory or reverse_key in self.relation_memory:
                    transitive_score += 0.1
                    transitive_count += 1
            
            # Calculate recursive ancestry (self-reference)
            recursive_factor = 0.0
            if grain_id in grain.ancestry:
                # Self-reference creates strong recursive structure
                recursive_factor = 0.3
            
            # Combine components
            if transitive_count > 0:
                transitive_contribution = min(0.5, transitive_score / transitive_count)
            else:
                transitive_contribution = 0.0
                
            # Store ancestry curvature if not already present
            if not hasattr(grain, 'ancestry_curvature'):
                grain.ancestry_curvature = 0.0
            grain.ancestry_curvature = ancestry_curvature
            
            # Final collapse metric calculation, enhanced with curvature
            grain.collapse_metric = base_metric + transitive_contribution + recursive_factor + ancestry_curvature * 0.3
    
    def _observe_system_metrics(self):
        """Observe overall system metrics from both spaces"""
        # Update superposition count
        self.superposition_count = 0
        for grain in self.grains.values():
            if grain.is_in_superposition():  # FIXED: Call method instead of function
                self.superposition_count += 1
        
        # Get coherence from coordinator
        self.field_coherence = self.toroidal_coordinator.calculate_global_coherence()
        
        # Observe tension - delegate to configuration space
        tension_values = []
        for point in self.config_space.points.values():
            if hasattr(point, 'structural_tension'):
                tension_values.append(point.structural_tension)
                
        self.system_tension = sum(tension_values) / len(tension_values) if tension_values else 0.0