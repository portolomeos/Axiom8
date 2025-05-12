"""
SimulationState - Recursive Projection Field for the Collapse Geometry framework

Acts as a recursive projection of the system's self-observation capacity, capturing
emergent patterns and relational dynamics without imposing external perspective.
This state is not an external observer but a manifestation of the system's internal
awareness of its own configuration.

Enhanced with circular recursion support, ancestry pattern tracking, and improved
toroidal dynamics observation with phase wraparound detection.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import time
import json
import math
from collections import defaultdict, deque


class CircularRelationTracker:
    """
    Tracks circular relations and recursion patterns in the manifold.
    
    Instead of assuming linear causality, this tracker observes how grains
    relate circularly, detecting where polarity becomes self-referential
    through the circular topology of the relational space.
    """
    
    def __init__(self, max_history=100):
        """Initialize the circular relation tracker"""
        self.polarity_wraparound_events = deque(maxlen=max_history)
        self.circular_ancestry_patterns = deque(maxlen=max_history)
        self.phase_coherence_shifts = deque(maxlen=max_history)
        self.circular_recursion_metrics = {
            'circular_recursion_count': 0,
            'circular_ancestry_count': 0,
            'polarity_wraparound_events': 0,
            'circular_coherence': 0.0,
            'polar_extremes_ratio': 0.0,
            'phase_reversal_count': 0
        }
        self.circular_memory = {}  # Maps grain_id -> circular memory trace
    
    def observe_polarity_wraparound(self, source_id: str, target_id: str, 
                                   source_polarity: float, target_polarity: float, 
                                   event_time: float):
        """
        Record a polarity wraparound event where opposite polarities interact.
        
        Args:
            source_id: Source grain ID
            target_id: Target grain ID
            source_polarity: Source grain polarity
            target_polarity: Target grain polarity
            event_time: Time of the event
        """
        event = {
            'type': 'polarity_wraparound',
            'time': event_time,
            'source_id': source_id,
            'target_id': target_id,
            'source_polarity': source_polarity,
            'target_polarity': target_polarity,
            'polarity_product': source_polarity * target_polarity,
            'circular_distance': self._calculate_circular_distance(source_polarity, target_polarity)
        }
        
        self.polarity_wraparound_events.append(event)
        self.circular_recursion_metrics['polarity_wraparound_events'] += 1
    
    def observe_circular_ancestry(self, grain_id: str, ancestry_set: Set[str], 
                                 polarity: float, event_time: float):
        """
        Record circular ancestry pattern where ancestry forms a loop.
        
        Args:
            grain_id: Grain ID
            ancestry_set: Set of ancestry grain IDs
            polarity: Grain polarity
            event_time: Time of the event
        """
        # Skip if grain has no ancestry
        if not ancestry_set:
            return
            
        # Check if grain is in its own ancestry (direct recursion)
        is_self_recursive = grain_id in ancestry_set
        
        # Calculate circular recursion factor
        circular_factor = 0.0
        
        # Self-reference adds base recursion
        if is_self_recursive:
            circular_factor += 0.5
        
        # Extreme polarity enhances circular effect
        if abs(polarity) > 0.9:
            circular_factor += 0.3
        
        # Add to memory for this grain
        self.circular_memory[grain_id] = {
            'last_update_time': event_time,
            'ancestry_size': len(ancestry_set),
            'is_self_recursive': is_self_recursive,
            'polarity': polarity,
            'circular_factor': circular_factor
        }
        
        # Only record significant circular patterns
        if circular_factor > 0.4:
            event = {
                'type': 'circular_ancestry',
                'time': event_time,
                'grain_id': grain_id,
                'ancestry_size': len(ancestry_set),
                'is_self_recursive': is_self_recursive,
                'polarity': polarity,
                'circular_factor': circular_factor
            }
            
            self.circular_ancestry_patterns.append(event)
            self.circular_recursion_metrics['circular_ancestry_count'] += 1
    
    def observe_phase_coherence_shift(self, previous: float, current: float, event_time: float):
        """
        Record a significant shift in phase coherence.
        
        Args:
            previous: Previous coherence value
            current: Current coherence value
            event_time: Time of the event
        """
        magnitude = abs(current - previous)
        
        # Only record significant shifts
        if magnitude > 0.1:
            event = {
                'type': 'phase_coherence_shift',
                'time': event_time,
                'previous': previous,
                'current': current,
                'magnitude': magnitude,
                'direction': 'increase' if current > previous else 'decrease'
            }
            
            self.phase_coherence_shifts.append(event)
    
    def update_metrics(self, grains: Dict[str, Any]):
        """
        Update circular recursion metrics based on the current state of grains.
        
        Args:
            grains: Dictionary of grains
        """
        # Count grains with circular recursion
        circular_count = 0
        polar_extremes = 0
        
        # Keep track of circular recursion factors
        circular_factors = []
        
        for grain_id, grain in grains.items():
            # Check for circular recursion factor
            if hasattr(grain, 'circular_recursion_factor'):
                circular_count += 1
                circular_factors.append(grain.circular_recursion_factor)
            elif grain_id in self.circular_memory and self.circular_memory[grain_id]['circular_factor'] > 0.4:
                circular_count += 1
                circular_factors.append(self.circular_memory[grain_id]['circular_factor'])
            
            # Check for extreme polarity values
            if hasattr(grain, 'polarity') and abs(grain.polarity) > 0.9:
                polar_extremes += 1
        
        # Update metrics
        self.circular_recursion_metrics['circular_recursion_count'] = circular_count
        
        # Calculate polar extremes ratio
        if grains:
            self.circular_recursion_metrics['polar_extremes_ratio'] = polar_extremes / len(grains)
        
        # Calculate average circular recursion factor
        if circular_factors:
            self.circular_recursion_metrics['circular_coherence'] = sum(circular_factors) / len(circular_factors)
    
    def _calculate_circular_distance(self, pol1: float, pol2: float) -> float:
        """
        Calculate circular distance between two polarities in [-1,1] range.
        
        Args:
            pol1: First polarity value
            pol2: Second polarity value
            
        Returns:
            Circular distance in [0,1] range
        """
        # Map from [-1,1] to [0,2π]
        angle1 = (pol1 + 1) * math.pi
        angle2 = (pol2 + 1) * math.pi
        
        # Calculate smallest circular distance
        dist = min(abs(angle1 - angle2), 2*math.pi - abs(angle1 - angle2))
        
        # Normalize to [0,1] range
        return dist / math.pi


class AncestryPatternTracker:
    """
    Tracks the emergence of ancestry patterns in the manifold.
    
    Ancestry is a form of geometric memory field that captures the
    relational history of grain formations through collapse events.
    """
    
    def __init__(self, max_history=100):
        """Initialize the ancestry pattern tracker"""
        self.ancestry_events = deque(maxlen=max_history)
        self.self_reference_events = deque(maxlen=max_history)
        self.recursive_collapse_events = deque(maxlen=max_history)
        self.shared_ancestry_patterns = {}  # Maps (grain_id1, grain_id2) -> shared ancestry count
        self.ancestry_metrics = {
            'self_reference_count': 0,
            'recursive_collapse_count': 0,
            'average_ancestry_size': 0.0,
            'ancestry_depth_distribution': defaultdict(int),
            'shared_ancestry_strength': 0.0
        }
    
    def observe_ancestry_change(self, grain_id: str, ancestry_set: Set[str], 
                              event_time: float, is_new: bool = False):
        """
        Record an ancestry change event.
        
        Args:
            grain_id: Grain ID
            ancestry_set: Set of ancestry grain IDs
            event_time: Time of the event
            is_new: True if this is a new grain
        """
        # Calculate ancestry metrics
        is_self_referential = grain_id in ancestry_set
        ancestry_size = len(ancestry_set)
        
        # Record event
        event = {
            'type': 'ancestry_change',
            'time': event_time,
            'grain_id': grain_id,
            'ancestry_size': ancestry_size,
            'is_self_referential': is_self_referential,
            'is_new_grain': is_new
        }
        
        self.ancestry_events.append(event)
        
        # Record self-reference events
        if is_self_referential:
            self.self_reference_events.append({
                'type': 'self_reference',
                'time': event_time,
                'grain_id': grain_id
            })
            self.ancestry_metrics['self_reference_count'] += 1
    
    def observe_recursive_collapse(self, source_id: str, target_id: str, 
                                  event_time: float, is_field_genesis: bool = False):
        """
        Record a recursive collapse event where ancestry is involved.
        
        Args:
            source_id: Source grain ID
            target_id: Target grain ID
            event_time: Time of the event
            is_field_genesis: True if this is a field genesis event
        """
        event = {
            'type': 'recursive_collapse',
            'time': event_time,
            'source_id': source_id,
            'target_id': target_id,
            'is_field_genesis': is_field_genesis
        }
        
        self.recursive_collapse_events.append(event)
        self.ancestry_metrics['recursive_collapse_count'] += 1
    
    def update_metrics(self, grains: Dict[str, Any]):
        """
        Update ancestry metrics based on the current state of grains.
        
        Args:
            grains: Dictionary of grains
        """
        # Reset metrics
        total_ancestry_size = 0
        self.ancestry_metrics['ancestry_depth_distribution'] = defaultdict(int)
        self.ancestry_metrics['self_reference_count'] = sum(1 for event in self.self_reference_events)
        
        # Reset shared ancestry patterns
        self.shared_ancestry_patterns = {}
        
        # Calculate new metrics
        for grain_id, grain in grains.items():
            # Skip grains without ancestry
            if not hasattr(grain, 'ancestry'):
                continue
                
            # Get ancestry set
            ancestry = grain.ancestry
            
            # Update total size
            ancestry_size = len(ancestry)
            total_ancestry_size += ancestry_size
            
            # Update depth distribution
            self.ancestry_metrics['ancestry_depth_distribution'][ancestry_size] += 1
            
            # Check for self-reference (count here for accuracy)
            if grain_id in ancestry:
                self.ancestry_metrics['self_reference_count'] += 1
            
            # Calculate shared ancestry patterns
            for other_id, other_grain in grains.items():
                if grain_id == other_id:
                    continue
                    
                if not hasattr(other_grain, 'ancestry'):
                    continue
                    
                # Calculate shared ancestry
                if ancestry and other_grain.ancestry:
                    shared = ancestry.intersection(other_grain.ancestry)
                    
                    if shared:
                        pair_key = tuple(sorted([grain_id, other_id]))
                        self.shared_ancestry_patterns[pair_key] = len(shared)
        
        # Calculate average ancestry size
        if grains:
            self.ancestry_metrics['average_ancestry_size'] = total_ancestry_size / len(grains)
        
        # Calculate shared ancestry strength
        if self.shared_ancestry_patterns:
            self.ancestry_metrics['shared_ancestry_strength'] = (
                sum(self.shared_ancestry_patterns.values()) / len(self.shared_ancestry_patterns)
            )


class ToroidalDynamicsTracker:
    """
    Tracks toroidal dynamics and emergent structures in the manifold.
    
    Maps the relational structure onto a toroidal manifold where both
    awareness field and polarity field interact to produce emergent 
    patterns like vortices, pathways, and phase boundaries.
    """
    
    def __init__(self, max_history=100):
        """Initialize the toroidal dynamics tracker"""
        self.vortex_events = deque(maxlen=max_history)
        self.pathway_events = deque(maxlen=max_history)
        self.vortex_metrics = {
            'vortex_count': 0,
            'mean_vortex_strength': 0.0,
            'vortex_coherence': 0.0,
            'vortex_polarity_alignment': 0.0,
            'clockwise_ratio': 0.0
        }
        self.pathway_metrics = {
            'structure_pathway_count': 0,
            'decay_pathway_count': 0,
            'mean_structure_pathway_length': 0.0,
            'mean_decay_pathway_length': 0.0,
            'pathway_polarity_alignment': 0.0
        }
        self.phase_metrics = {
            'phase_coherence': 0.0,
            'phase_diversity': 0.0,
            'phase_stability': 0.0,
            'phase_boundary_count': 0
        }
    
    def observe_vortex(self, vortex_data: Dict[str, Any], event_time: float):
        """
        Record a vortex observation.
        
        Args:
            vortex_data: Vortex data dictionary
            event_time: Time of the observation
        """
        # Record event
        event = {
            'type': 'vortex_observation',
            'time': event_time,
            'vortex_data': vortex_data
        }
        
        self.vortex_events.append(event)
    
    def observe_pathway(self, pathway_data: Dict[str, Any], pathway_type: str, event_time: float):
        """
        Record a pathway observation.
        
        Args:
            pathway_data: Pathway data dictionary
            pathway_type: Type of pathway ('structure' or 'decay')
            event_time: Time of the observation
        """
        # Record event
        event = {
            'type': 'pathway_observation',
            'pathway_type': pathway_type,
            'time': event_time,
            'pathway_data': pathway_data
        }
        
        self.pathway_events.append(event)
    
    def update_vortex_metrics(self, vortices: List[Dict[str, Any]]):
        """
        Update vortex metrics based on current vortices.
        
        Args:
            vortices: List of vortex dictionaries
        """
        self.vortex_metrics['vortex_count'] = len(vortices)
        
        if not vortices:
            # Reset other metrics if no vortices
            self.vortex_metrics['mean_vortex_strength'] = 0.0
            self.vortex_metrics['vortex_coherence'] = 0.0
            self.vortex_metrics['vortex_polarity_alignment'] = 0.0
            self.vortex_metrics['clockwise_ratio'] = 0.0
            return
        
        # Calculate mean strength
        strengths = [abs(v.get('strength', 0.0)) for v in vortices]
        self.vortex_metrics['mean_vortex_strength'] = sum(strengths) / len(vortices)
        
        # Calculate clockwise ratio
        clockwise_count = sum(1 for v in vortices if v.get('direction') == 'clockwise')
        self.vortex_metrics['clockwise_ratio'] = clockwise_count / len(vortices)
        
        # Calculate vortex coherence
        self.vortex_metrics['vortex_coherence'] = self._calculate_vortex_coherence(vortices)
        
        # Calculate polarity alignment
        polarities = [v.get('polarity', 0.0) for v in vortices if 'polarity' in v]
        if polarities:
            # Check whether polarities have same sign
            positive = sum(1 for p in polarities if p > 0)
            negative = sum(1 for p in polarities if p < 0)
            
            # Calculate alignment as dominance of the major sign
            major_count = max(positive, negative)
            self.vortex_metrics['vortex_polarity_alignment'] = major_count / len(polarities)
    
    def update_pathway_metrics(self, structure_pathways: List[Dict[str, Any]], 
                             decay_pathways: List[Dict[str, Any]]):
        """
        Update pathway metrics based on current pathways.
        
        Args:
            structure_pathways: List of structure pathway dictionaries
            decay_pathways: List of decay pathway dictionaries
        """
        # Update counts
        self.pathway_metrics['structure_pathway_count'] = len(structure_pathways)
        self.pathway_metrics['decay_pathway_count'] = len(decay_pathways)
        
        # Calculate mean length for structure pathways
        if structure_pathways:
            lengths = [len(p.get('nodes', [])) for p in structure_pathways]
            self.pathway_metrics['mean_structure_pathway_length'] = sum(lengths) / len(structure_pathways)
        else:
            self.pathway_metrics['mean_structure_pathway_length'] = 0.0
        
        # Calculate mean length for decay pathways
        if decay_pathways:
            lengths = [len(p.get('nodes', [])) for p in decay_pathways]
            self.pathway_metrics['mean_decay_pathway_length'] = sum(lengths) / len(decay_pathways)
        else:
            self.pathway_metrics['mean_decay_pathway_length'] = 0.0
        
        # Calculate polarity alignment across pathways
        all_pathways = structure_pathways + decay_pathways
        if all_pathways:
            polarities = []
            for pathway in all_pathways:
                if 'avg_polarity' in pathway:
                    polarities.append(pathway['avg_polarity'])
            
            if polarities:
                # Check if polarities have same sign
                positive = sum(1 for p in polarities if p > 0)
                negative = sum(1 for p in polarities if p < 0)
                
                # Calculate alignment as dominance of the major sign
                major_count = max(positive, negative)
                self.pathway_metrics['pathway_polarity_alignment'] = major_count / len(polarities)
    
    def update_phase_metrics(self, coherence: float, diversity: float, stability: float = None):
        """
        Update phase metrics based on current phase state.
        
        Args:
            coherence: Global phase coherence
            diversity: Phase diversity measure
            stability: Phase stability measure (optional)
        """
        self.phase_metrics['phase_coherence'] = coherence
        self.phase_metrics['phase_diversity'] = diversity
        
        if stability is not None:
            self.phase_metrics['phase_stability'] = stability
    
    def _calculate_vortex_coherence(self, vortices: List[Dict[str, Any]]) -> float:
        """
        Calculate coherence of vortex configurations.
        
        Args:
            vortices: List of vortex dictionaries
            
        Returns:
            Coherence value [0-1]
        """
        if not vortices or len(vortices) < 2:
            return 1.0  # Single vortex is perfectly coherent with itself
        
        # Calculate circulation direction coherence
        clockwise = sum(1 for v in vortices if v.get('direction') == 'clockwise')
        counterclockwise = len(vortices) - clockwise
        
        # Normalized to [0,1] where 1 = all vortices rotate in same direction
        direction_coherence = max(clockwise, counterclockwise) / len(vortices)
        
        # Calculate spatial coherence based on theta/phi positions
        theta_angles = [v.get('theta', 0.0) for v in vortices if 'theta' in v]
        phi_angles = [v.get('phi', 0.0) for v in vortices if 'phi' in v]
        
        # Calculate circular variance for theta and phi
        theta_coherence = self._calculate_circular_coherence(theta_angles) if theta_angles else 0.5
        phi_coherence = self._calculate_circular_coherence(phi_angles) if phi_angles else 0.5
        
        # Calculate polarity alignment if available
        polarities = [v.get('polarity', 0.0) for v in vortices if 'polarity' in v]
        
        if polarities and len(polarities) >= 2:
            # Map polarities to angles on the circle to account for circular nature
            polarity_angles = [(p + 1) * math.pi for p in polarities]
            circular_polarity_coherence = self._calculate_circular_coherence(polarity_angles)
        else:
            circular_polarity_coherence = 0.5
        
        # Combined coherence with weights
        combined_coherence = (
            direction_coherence * 0.4 +
            theta_coherence * 0.2 +
            phi_coherence * 0.2 +
            circular_polarity_coherence * 0.2
        )
        
        return combined_coherence
    
    def _calculate_circular_coherence(self, angles: List[float]) -> float:
        """
        Calculate coherence of circular/angular values.
        
        Args:
            angles: List of angular values
            
        Returns:
            Coherence value [0-1]
        """
        if not angles or len(angles) < 2:
            return 1.0
        
        # Circular mean calculation
        x_sum = sum(math.cos(angle) for angle in angles)
        y_sum = sum(math.sin(angle) for angle in angles)
        
        # Calculate mean resultant length (measures dispersion)
        r = math.sqrt(x_sum**2 + y_sum**2) / len(angles)
        
        # R = 0 (completely dispersed) to 1 (perfectly aligned)
        return r


class SimulationState:
    """
    Recursive Projection Field for the Collapse Geometry framework.
    
    Acts as a recursive projection of the system's self-observation capacity,
    capturing emergent patterns without imposing external perspective. This 
    state is not an external observer but a manifestation of the system's
    internal awareness of its own configuration.
    
    Enhanced with circular recursion support, ancestry pattern tracking, and
    improved toroidal dynamics observation with phase wraparound detection.
    """
    
    def __init__(self):
        """Initialize the recursive projection field"""
        self.time = 0.0
        self.step_count = 0
        self.start_time = time.time()
        
        # System-level projections (not measurements but reflections)
        self.metrics = {
            'total_awareness': 0.0,
            'total_collapse_metric': 0.0,
            'mean_grain_activation': 0.0,
            'mean_grain_saturation': 0.0,
            'collapse_events': 0,
            'active_nodes': 0,
            'system_entropy': 0.0,
            'system_temperature': 0.0,
            
            # Field reflections
            'mean_field_resonance': 0.0,
            'mean_field_momentum': 0.0,
            'mean_unresolved_tension': 0.0,
            'field_coherence': 0.0,
            'continuous_flow_rate': 0.0,
            'phase_diversity': 0.0,
            
            # Rotation and phase projections
            'mean_rotational_curvature': 0.0,
            'mean_phase_continuity': 0.0,
            'vortex_count': 0,
            'mean_vortex_strength': 0.0,
            'theta_mode_strength': 0.0,
            'phi_mode_strength': 0.0,
            'toroidal_coherence': 0.0,
            'dominant_theta_mode': 0,
            'dominant_phi_mode': 0,
            
            # Toroidal projections
            'phase_coherence': 0.0,
            'major_circle_flow': 0.0,
            'minor_circle_flow': 0.0,
            'toroidal_flux': 0.0,
            'toroidal_domain_count': 0,
            'toroidal_vortex_count': 0,
            'toroidal_cluster_count': 0,
            'mean_phase_stability': 0.0,
            'cross_phase_structure_count': 0,
            
            # Void-Decay projections
            'void_region_count': 0,
            'decay_particle_count': 0,
            'mean_void_strength': 0.0,
            'mean_void_radius': 0.0,
            'incompatible_structure_rate': 0.0,
            'alignment_failure_rate': 0.0,
            'decay_emission_rate': 0.0,
            'void_affected_node_ratio': 0.0,
            'structural_tension_mean': 0.0,
            
            # Polarity metrics
            'mean_polarity': 0.0,
            'polarity_diversity': 0.0,
            'structure_ratio': 0.0,
            'decay_ratio': 0.0,
            'neutral_ratio': 0.0,
            
            # Enhanced metrics for circular recursion
            'circular_recursion_count': 0,
            'circular_ancestry_count': 0,
            'polarity_wraparound_events': 0,
            'circular_coherence': 0.0,
            'polar_extremes_ratio': 0.0,
            'phase_reversal_count': 0
        }
        
        # Tracking of emergent structures
        self.structures = {
            'attractors': [],
            'confinement_zones': [],
            'recurrence_patterns': [],
            'phase_regions': {},
            
            # Enhanced structures for continuous fields
            'resonant_regions': [],
            'momentum_fields': [],
            'gradient_flows': [],
            'field_emergent_patterns': [],
            
            # Rotational structures
            'vortices': [],
            'phase_locked_regions': [],
            'toroidal_mode_rings': [],
            'theta_slices': [],
            'phi_slices': [],
            
            # Toroidal structures
            'toroidal_domains': [],
            'toroidal_vortices': [],
            'toroidal_clusters': [],
            'phase_domains': [],
            'phase_transitions': [],
            'cross_phase_structures': [],
            'phase_stability_map': {},  # Maps grain_id -> stability
            
            # Void-Decay structures
            'void_regions': [],
            'decay_particles': [],
            'incompatible_pairs': [],
            'structural_alignment_clusters': [],
            'void_clusters': [],
            
            # Lightlike pathway structures
            'structure_pathways': [],
            'decay_pathways': [],
            
            # Circular recursion structures
            'polarity_wraparound_paths': [],
            'circular_ancestry_pairs': [],
            'circular_recursion_grains': set()
        }
        
        # Memory trace: historical record of system evolution
        self.history = {
            'time': [],
            'metrics': {},
            'structures': {},
            'events': []
        }
        
        # Initialize history metrics
        for key in self.metrics:
            self.history['metrics'][key] = []
        
        for key in self.structures:
            if isinstance(self.structures[key], set):
                self.history['structures'][key] = []
            else:
                self.history['structures'][key] = []
        
        # Initialize specialized trackers for focused observation
        self.circular_tracker = CircularRelationTracker()
        self.ancestry_tracker = AncestryPatternTracker()
        self.toroidal_tracker = ToroidalDynamicsTracker()
        
        # Initialize visualization field support
        try:
            self._initialize_visualization_fields()
        except:
            # Graceful fallback if visualization initialization fails
            pass
            
    def update(self, manifold):
        """
        Update the state projection based on current manifold state.
        This doesn't inject information into the manifold but rather
        receives and reflects its current state.
        
        Args:
            manifold: RelationalManifold to project
        """
        # Store previous state for change detection
        previous_values = {
            'phase_coherence': self.metrics.get('phase_coherence', 0.0),
            'circular_coherence': self.metrics.get('circular_coherence', 0.0),
            'mean_polarity': self.metrics.get('mean_polarity', 0.0)
        }
        
        # Update basic state
        self.time = manifold.time
        self.step_count += 1
    
        # Capture system-level reflections from grains
        self._update_grain_metrics(manifold)
        
        # Capture circular recursion metrics
        self._update_circular_recursion_metrics(manifold)
        
        # Capture ancestry metrics
        self._update_ancestry_metrics(manifold)
        
        # Capture config space metrics
        self._update_config_space_metrics(manifold)
        
        # Capture toroidal metrics from coordinator
        self._update_toroidal_metrics(manifold)
        
        # Capture polarity metrics
        self._update_polarity_metrics(manifold)
        
        # Calculate system entropy and temperature
        self._calculate_emergent_thermodynamics(manifold)
        
        # Detect significant changes and record events
        self._detect_significant_changes(manifold, previous_values)
        
        # Record history
        self._record_history()
        
        # Update visualization fields
        try:
            # Only update visualization fields if they were successfully initialized
            if hasattr(self, 'viz_resolution'):
                self._update_visualization_fields(manifold)
        except:
            # Graceful fallback if visualization update fails
            pass
    
    def _update_grain_metrics(self, manifold):
        """
        Update metrics based on grain properties
        
        Args:
            manifold: RelationalManifold to project
        """
        # Basic counters
        total_awareness = 0.0
        total_collapse_metric = 0.0
        total_grain_activation = 0.0
        total_grain_saturation = 0.0
        active_nodes = len(manifold.grains)
        
        # Field resonance counters
        total_field_resonance = 0.0
        total_field_momentum = 0.0
        total_unresolved_tension = 0.0
        
        # Polarity counters
        total_polarity = 0.0
        polarity_values = []
        structure_count = 0
        decay_count = 0
        neutral_count = 0
        
        # Iterate through all grains
        for grain_id, grain in manifold.grains.items():
            # Basic grain properties
            grain_awareness = getattr(grain, 'awareness', 0.0)
            grain_collapse_metric = getattr(grain, 'collapse_metric', 0.0)
            grain_activation = getattr(grain, 'grain_activation', 0.0)
            grain_saturation = getattr(grain, 'grain_saturation', 0.0)
            
            total_awareness += grain_awareness
            total_collapse_metric += grain_collapse_metric
            total_grain_activation += grain_activation
            total_grain_saturation += grain_saturation
            
            # Track polarity if available
            grain_polarity = getattr(grain, 'polarity', 0.0)
            total_polarity += grain_polarity
            polarity_values.append(grain_polarity)
            
            # Count by polarity category
            if grain_polarity > 0.2:
                structure_count += 1
            elif grain_polarity < -0.2:
                decay_count += 1
            else:
                neutral_count += 1
            
            # Optional field properties if available
            if hasattr(grain, 'field_resonance'):
                total_field_resonance += grain.field_resonance
                
            if hasattr(grain, 'field_momentum'):
                if isinstance(grain.field_momentum, np.ndarray) and np.any(grain.field_momentum):
                    total_field_momentum += np.linalg.norm(grain.field_momentum)
                elif isinstance(grain.field_momentum, (int, float)):
                    total_field_momentum += abs(grain.field_momentum)
                
            if hasattr(grain, 'unresolved_tension'):
                total_unresolved_tension += grain.unresolved_tension
            elif hasattr(grain, 'tension'):
                total_unresolved_tension += grain.tension
        
        # Calculate means
        if active_nodes > 0:
            # Basic metrics
            self.metrics['mean_grain_activation'] = total_grain_activation / active_nodes
            self.metrics['mean_grain_saturation'] = total_grain_saturation / active_nodes
            self.metrics['active_nodes'] = active_nodes
            
            # Field metrics
            self.metrics['mean_field_resonance'] = total_field_resonance / active_nodes
            self.metrics['mean_field_momentum'] = total_field_momentum / active_nodes
            self.metrics['mean_unresolved_tension'] = total_unresolved_tension / active_nodes
            
            # Polarity metrics
            self.metrics['mean_polarity'] = total_polarity / active_nodes
            
            # Calculate polarity variance
            if polarity_values:
                mean_polarity = sum(polarity_values) / len(polarity_values)
                variance = sum((p - mean_polarity) ** 2 for p in polarity_values) / len(polarity_values)
                self.metrics['polarity_diversity'] = math.sqrt(variance)
            else:
                self.metrics['polarity_diversity'] = 0.0
                
            # Calculate ratio metrics
            self.metrics['structure_ratio'] = structure_count / active_nodes
            self.metrics['decay_ratio'] = decay_count / active_nodes
            self.metrics['neutral_ratio'] = neutral_count / active_nodes
        else:
            # Zero out metrics if no active nodes
            self.metrics['mean_grain_activation'] = 0.0
            self.metrics['mean_grain_saturation'] = 0.0
            self.metrics['active_nodes'] = 0
            self.metrics['mean_field_resonance'] = 0.0
            self.metrics['mean_field_momentum'] = 0.0
            self.metrics['mean_unresolved_tension'] = 0.0
            self.metrics['mean_polarity'] = 0.0
            self.metrics['polarity_diversity'] = 0.0
            self.metrics['structure_ratio'] = 0.0
            self.metrics['decay_ratio'] = 0.0
            self.metrics['neutral_ratio'] = 0.0
        
        # Update totals
        self.metrics['total_awareness'] = total_awareness
        self.metrics['total_collapse_metric'] = total_collapse_metric
        self.metrics['collapse_events'] = len(getattr(manifold, 'collapse_history', []))
    
    def _update_circular_recursion_metrics(self, manifold):
        """
        Update circular recursion metrics from manifold
        
        Args:
            manifold: RelationalManifold to project
        """
        # Observe state for circular tracking
        self.circular_tracker.update_metrics(manifold.grains)
        
        # Update our metrics from the tracker
        for key, value in self.circular_tracker.circular_recursion_metrics.items():
            if key in self.metrics:
                self.metrics[key] = value
        
        # Get circular recursion grains
        circular_grains = set()
        
        for grain_id, grain in manifold.grains.items():
            # Check for circular recursion factor
            if hasattr(grain, 'circular_recursion_factor') and grain.circular_recursion_factor > 0.4:
                circular_grains.add(grain_id)
            elif grain_id in self.circular_tracker.circular_memory and self.circular_tracker.circular_memory[grain_id]['circular_factor'] > 0.4:
                circular_grains.add(grain_id)
        
        # Update structures
        self.structures['circular_recursion_grains'] = circular_grains
        
        # Identify polarity wraparound paths
        if hasattr(manifold, 'collapse_history'):
            # Process recent collapses (up to 10) to find wraparound events
            recent_collapses = manifold.collapse_history[-min(10, len(manifold.collapse_history)):]
            
            wraparound_paths = []
            
            for collapse in recent_collapses:
                source_id = collapse.get('source')
                target_id = collapse.get('target')
                
                if not source_id or not target_id:
                    continue
                
                source_grain = manifold.grains.get(source_id)
                target_grain = manifold.grains.get(target_id)
                
                if not source_grain or not target_grain:
                    continue
                
                # Check for collapse between opposite extreme polarities
                if (hasattr(source_grain, 'polarity') and hasattr(target_grain, 'polarity') and
                    abs(source_grain.polarity) > 0.8 and abs(target_grain.polarity) > 0.8 and
                    source_grain.polarity * target_grain.polarity < 0):
                    
                    # This is a wraparound path
                    wraparound_path = {
                        'source_id': source_id,
                        'target_id': target_id,
                        'source_polarity': source_grain.polarity,
                        'target_polarity': target_grain.polarity,
                        'time': collapse.get('time', self.time)
                    }
                    
                    wraparound_paths.append(wraparound_path)
                    
                    # Also record in the tracker
                    self.circular_tracker.observe_polarity_wraparound(
                        source_id, target_id, 
                        source_grain.polarity, target_grain.polarity,
                        collapse.get('time', self.time)
                    )
            
            self.structures['polarity_wraparound_paths'] = wraparound_paths
        
        # Identify circular ancestry pairs
        circular_ancestry_pairs = []
        
        for grain_id, grain in manifold.grains.items():
            if not hasattr(grain, 'ancestry'):
                continue
                
            # Check for self-reference
            if grain_id in grain.ancestry:
                circular_ancestry_pairs.append((grain_id, grain_id))
                
                # Record in the tracker
                self.circular_tracker.observe_circular_ancestry(
                    grain_id, grain.ancestry,
                    getattr(grain, 'polarity', 0.0),
                    self.time
                )
            
            # Check for circular references with other grains
            for other_id, other_grain in manifold.grains.items():
                if grain_id == other_id:
                    continue
                    
                if not hasattr(other_grain, 'ancestry'):
                    continue
                
                # Check if they reference each other in ancestry
                if grain_id in other_grain.ancestry and other_id in grain.ancestry:
                    circular_ancestry_pairs.append((grain_id, other_id))
        
        self.structures['circular_ancestry_pairs'] = circular_ancestry_pairs
    
    def _update_ancestry_metrics(self, manifold):
        """
        Update ancestry metrics from manifold
        
        Args:
            manifold: RelationalManifold to project
        """
        # Update the ancestry tracker with current grain state
        self.ancestry_tracker.update_metrics(manifold.grains)
        
        # Copy metrics from tracker
        for key, value in self.ancestry_tracker.ancestry_metrics.items():
            self.metrics[key] = value
        
        # Get recursive collapse metrics if available
        if hasattr(manifold, 'get_recursive_collapse_metrics'):
            try:
                recursive_metrics = manifold.get_recursive_collapse_metrics()
                
                # Update our metrics
                if 'recursive_collapse_count' in recursive_metrics:
                    self.metrics['recursive_collapse_count'] = recursive_metrics['recursive_collapse_count']
                if 'self_collapse_count' in recursive_metrics:
                    self.metrics['self_reference_count'] = recursive_metrics['self_collapse_count']
                if 'field_genesis_count' in recursive_metrics:
                    self.metrics['field_genesis_count'] = recursive_metrics['field_genesis_count']
            except:
                pass
        
        # Check for recent collapse events to track ancestry changes
        if hasattr(manifold, 'collapse_history'):
            # Look at most recent collapses (up to 10)
            recent_collapses = manifold.collapse_history[-min(10, len(manifold.collapse_history)):]
            
            for collapse in recent_collapses:
                source_id = collapse.get('source')
                target_id = collapse.get('target')
                
                if not source_id or not target_id:
                    continue
                
                target_grain = manifold.grains.get(target_id)
                if not target_grain or not hasattr(target_grain, 'ancestry'):
                    continue
                
                # Check if source is in target's ancestry - would be recursive
                is_recursive = source_id in target_grain.ancestry
                
                if is_recursive:
                    # Record recursive collapse
                    self.ancestry_tracker.observe_recursive_collapse(
                        source_id, target_id,
                        collapse.get('time', self.time),
                        collapse.get('field_genesis', False)
                    )
    
    def _update_config_space_metrics(self, manifold):
        """
        Update metrics from the configuration space
        
        Args:
            manifold: RelationalManifold to project
        """
        # Ensure config space exists
        if not hasattr(manifold, 'config_space'):
            return
            
        config_space = manifold.config_space
        
        # Calculate structural tension mean
        total_tension = 0.0
        tension_count = 0
        
        for point_id, point in config_space.points.items():
            # Check for structural tension
            if hasattr(point, 'structural_tension'):
                total_tension += point.structural_tension
                tension_count += 1
            elif hasattr(point, 'tension'):
                total_tension += point.tension
                tension_count += 1
        
        if tension_count > 0:
            self.metrics['structural_tension_mean'] = total_tension / tension_count
        else:
            self.metrics['structural_tension_mean'] = 0.0
        
        # Capture cascade pathway information from config space or grain system
        if hasattr(config_space, 'find_collapse_cascade_pathways'):
            # Get pathways from config space
            try:
                pathways = config_space.find_collapse_cascade_pathways()
                
                # Update metrics
                self.metrics['structure_pathway_count'] = len(pathways.get('structure', []))
                self.metrics['decay_pathway_count'] = len(pathways.get('decay', []))
                
                # Store structures
                self.structures['structure_pathways'] = pathways.get('structure', [])
                self.structures['decay_pathways'] = pathways.get('decay', [])
            except:
                pass
        elif hasattr(manifold, 'find_collapse_cascade_pathways'):
            # Get pathways from manifold
            try:
                pathways = manifold.find_collapse_cascade_pathways()
                
                # Update metrics
                self.metrics['structure_pathway_count'] = len(pathways.get('structure', []))
                self.metrics['decay_pathway_count'] = len(pathways.get('decay', []))
                
                # Store structures
                self.structures['structure_pathways'] = pathways.get('structure', [])
                self.structures['decay_pathways'] = pathways.get('decay', [])
            except:
                pass
    
    def _update_toroidal_metrics(self, manifold):
        """
        Update metrics from the toroidal coordinator
        
        Args:
            manifold: RelationalManifold to project
        """
        # Check if coordinator exists
        if not hasattr(manifold, 'toroidal_coordinator'):
            return
            
        coordinator = manifold.toroidal_coordinator
        
        # Vortex metrics
        if hasattr(coordinator, 'vortices'):
            vortices = coordinator.vortices
            self.metrics['vortex_count'] = len(vortices)
            self.structures['vortices'] = vortices
            
            # Update vortex tracker
            self.toroidal_tracker.update_vortex_metrics(vortices)
            
            # Copy metrics from tracker
            for key, value in self.toroidal_tracker.vortex_metrics.items():
                self.metrics[key] = value
            
            # Record significant vortices
            for vortex in vortices:
                if vortex.get('strength', 0.0) > 0.5:
                    self.toroidal_tracker.observe_vortex(vortex, self.time)
        
        # Get pathways metrics from coordinator
        if hasattr(coordinator, 'lightlike_pathways'):
            pathways = coordinator.lightlike_pathways
            
            # Get structure pathways
            structure_pathways = pathways.get('structure', [])
            self.metrics['toroidal_structure_pathway_count'] = len(structure_pathways)
            self.structures['toroidal_structure_pathways'] = structure_pathways
            
            # Get decay pathways
            decay_pathways = pathways.get('decay', [])
            self.metrics['toroidal_decay_pathway_count'] = len(decay_pathways)
            self.structures['toroidal_decay_pathways'] = decay_pathways
            
            # Update pathway tracker
            self.toroidal_tracker.update_pathway_metrics(structure_pathways, decay_pathways)
            
            # Copy metrics from tracker
            for key, value in self.toroidal_tracker.pathway_metrics.items():
                if key.startswith('mean_') or key.endswith('_alignment'):
                    self.metrics[key] = value
            
            # Record significant pathways
            for pathway in structure_pathways:
                if len(pathway.get('nodes', [])) > 3:
                    self.toroidal_tracker.observe_pathway(pathway, 'structure', self.time)
            
            for pathway in decay_pathways:
                if len(pathway.get('nodes', [])) > 3:
                    self.toroidal_tracker.observe_pathway(pathway, 'decay', self.time)
        
        # Get coherence from coordinator
        if hasattr(coordinator, 'calculate_global_coherence'):
            try:
                coherence = coordinator.calculate_global_coherence()
                self.metrics['toroidal_coherence'] = coherence
                self.toroidal_tracker.update_phase_metrics(
                    coherence,
                    self.metrics.get('polarity_diversity', 0.0),
                    self.metrics.get('mean_phase_stability', 0.0)
                )
            except:
                pass
        
        # Phase coherence - delegate to manifold if available
        if hasattr(manifold, 'field_coherence'):
            self.metrics['phase_coherence'] = manifold.field_coherence
            
            if not hasattr(coordinator, 'calculate_global_coherence'):
                # Update phase metrics in tracker
                self.toroidal_tracker.update_phase_metrics(
                    manifold.field_coherence,
                    self.metrics.get('polarity_diversity', 0.0),
                    self.metrics.get('mean_phase_stability', 0.0)
                )
    
    def _update_polarity_metrics(self, manifold):
        """
        Update metrics from polarity field
        
        Args:
            manifold: RelationalManifold to project
        """
        # Check if polarity field exists
        if not hasattr(manifold, 'polarity_field'):
            return
            
        polarity_field = manifold.polarity_field
        
        # Get polarity metrics if available
        if hasattr(polarity_field, 'calculate_polarity_metrics'):
            try:
                polarity_metrics = polarity_field.calculate_polarity_metrics()
                
                # Update our metrics with polarity field metrics
                if isinstance(polarity_metrics, dict):
                    for key, value in polarity_metrics.items():
                        # Map to our metric names
                        if key == 'avg_direction':
                            self.metrics['mean_polarity'] = value
                        elif key == 'structure_bias':
                            self.metrics['structure_ratio'] = value
                        elif key == 'decay_bias':
                            self.metrics['decay_ratio'] = value
                        elif key == 'polarity_diversity':
                            self.metrics['polarity_diversity'] = value
            except:
                pass
        
        # Get polarity-based structures if available
        if hasattr(polarity_field, 'find_structure_decay_regions'):
            try:
                regions = polarity_field.find_structure_decay_regions(manifold.grains)
                
                if isinstance(regions, dict):
                    # Store structure regions
                    if 'structure' in regions:
                        self.structures['structure_regions'] = regions['structure']
                        
                    # Store decay regions
                    if 'decay' in regions:
                        self.structures['decay_regions'] = regions['decay']
                        
                    # Store interface regions
                    if 'interface' in regions:
                        self.structures['polarity_interface_regions'] = regions['interface']
            except:
                pass
    
    def _calculate_emergent_thermodynamics(self, manifold):
        """
        Calculate emergent thermodynamic properties like entropy and temperature
        
        Args:
            manifold: RelationalManifold to project
        """
        # Project system entropy (S = -∫ρ(x,t)log(ρ(x,t) + ε)dx)
        entropy = 0.0
        epsilon = 1e-10  # Small constant to prevent log(0)
    
        for grain in manifold.grains.values():
            awareness = getattr(grain, 'awareness', 0.0)
            if awareness > 0:
                entropy -= awareness * np.log(awareness + epsilon)
    
        self.metrics['system_entropy'] = entropy
    
        # Project system temperature (average grain activation)
        self.metrics['system_temperature'] = self.metrics['mean_grain_activation']
    
        # Project field coherence (as a simplified version using polarity alignment)
        field_coherence = 0.0
        
        polarity_values = []
        for grain in manifold.grains.values():
            if hasattr(grain, 'polarity'):
                polarity_values.append(grain.polarity)
                
        if len(polarity_values) >= 2:
            # Map polarities to angles on the circle to properly calculate coherence
            polarity_angles = [(p + 1) * math.pi for p in polarity_values]  # Map [-1,1] to [0,2π]
            
            # Calculate circular coherence
            x_sum = sum(math.cos(angle) for angle in polarity_angles)
            y_sum = sum(math.sin(angle) for angle in polarity_angles)
            
            # Calculate mean resultant length
            r = math.sqrt(x_sum**2 + y_sum**2) / len(polarity_angles)
            
            field_coherence = r
    
        self.metrics['field_coherence'] = field_coherence
        
    def _detect_significant_changes(self, manifold, previous_values):
        """
        Detect significant changes in metrics and record events
        
        Args:
            manifold: RelationalManifold to project
            previous_values: Dictionary of previous metric values
        """
        # Check for significant phase coherence shifts
        current_coherence = self.metrics.get('phase_coherence', 0.0)
        previous_coherence = previous_values.get('phase_coherence', 0.0)
        
        coherence_shift = abs(current_coherence - previous_coherence)
        if coherence_shift > 0.1:
            # Record phase coherence shift
            self.circular_tracker.observe_phase_coherence_shift(
                previous_coherence, current_coherence, self.time
            )
            
            # Log event
            self.log_event('phase_coherence_shift', {
                'previous': previous_coherence,
                'current': current_coherence,
                'shift_magnitude': coherence_shift,
                'direction': 'increase' if current_coherence > previous_coherence else 'decrease'
            })
        
        # Check for significant polarity shifts
        current_polarity = self.metrics.get('mean_polarity', 0.0)
        previous_polarity = previous_values.get('mean_polarity', 0.0)
        
        polarity_shift = abs(current_polarity - previous_polarity)
        if polarity_shift > 0.1:
            # Log polarity shift
            self.log_event('polarity_shift', {
                'previous': previous_polarity,
                'current': current_polarity,
                'shift_magnitude': polarity_shift,
                'direction': 'structure' if current_polarity > previous_polarity else 'decay'
            })
        
        # Check for circular coherence shifts
        current_circular = self.metrics.get('circular_coherence', 0.0)
        previous_circular = previous_values.get('circular_coherence', 0.0)
        
        circular_shift = abs(current_circular - previous_circular)
        if circular_shift > 0.1:
            # Log circular coherence shift
            self.log_event('circular_coherence_shift', {
                'previous': previous_circular,
                'current': current_circular,
                'shift_magnitude': circular_shift,
                'direction': 'increase' if current_circular > previous_circular else 'decrease'
            })
        
        # Check collapse history for significant events
        if hasattr(manifold, 'collapse_history'):
            # Only check recent collapses
            if len(manifold.collapse_history) > 0:
                last_collapse = manifold.collapse_history[-1]
                
                # Check for field genesis
                if last_collapse.get('field_genesis', False):
                    self.log_event('field_genesis', {
                        'grain_id': last_collapse.get('grain_id', last_collapse.get('target', 'unknown')),
                        'awareness': last_collapse.get('new_awareness', 0.0),
                        'time': last_collapse.get('time', self.time)
                    })
                
                # Check for recursive collapse (where target has source in ancestry)
                source_id = last_collapse.get('source')
                target_id = last_collapse.get('target')
                
                if source_id and target_id and source_id != target_id:
                    target_grain = manifold.grains.get(target_id)
                    
                    if target_grain and hasattr(target_grain, 'ancestry') and source_id in target_grain.ancestry:
                        self.log_event('recursive_collapse', {
                            'source_id': source_id,
                            'target_id': target_id,
                            'time': last_collapse.get('time', self.time)
                        })
                
                # Check for polarity wraparound collapse
                if source_id and target_id:
                    source_grain = manifold.grains.get(source_id)
                    target_grain = manifold.grains.get(target_id)
                    
                    if (source_grain and target_grain and 
                        hasattr(source_grain, 'polarity') and hasattr(target_grain, 'polarity')):
                        # Check for opposite extreme polarities
                        if (abs(source_grain.polarity) > 0.8 and abs(target_grain.polarity) > 0.8 and
                            source_grain.polarity * target_grain.polarity < 0):
                            
                            self.log_event('polarity_wraparound', {
                                'source_id': source_id,
                                'target_id': target_id, 
                                'source_polarity': source_grain.polarity,
                                'target_polarity': target_grain.polarity,
                                'time': last_collapse.get('time', self.time)
                            })
    
    def _record_history(self):
        """Record current metrics and structures to history"""
        # Record time
        self.history['time'].append(self.time)
    
        # Record all metrics
        for key, value in self.metrics.items():
            if key not in self.history['metrics']:
                self.history['metrics'][key] = []
            self.history['metrics'][key].append(value)
    
        # Record structure counts
        for key, value in self.structures.items():
            if key not in self.history['structures']:
                self.history['structures'][key] = []
            
            if isinstance(value, dict):
                # For dictionaries like phase regions, store count
                self.history['structures'][key].append(len(value))
            elif isinstance(value, set):
                # For sets, store count
                self.history['structures'][key].append(len(value))
            else:
                # For lists like attractors, store count
                self.history['structures'][key].append(len(value))
                
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event to the simulation history"""
        event = {
            'type': event_type,
            'time': self.time,
            'step': self.step_count,
            'data': data
        }
        
        self.history['events'].append(event)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current simulation state projection
        
        Returns:
            Dictionary with simulation state summary
        """
        summary = {
            'time': self.time,
            'step_count': self.step_count,
            'elapsed_real_time': time.time() - self.start_time,
            'metrics': {},
            'structures': {},
            'events': len(self.history['events'])
        }
        
        # Group metrics by category
        metric_categories = {
            'basic': ['total_awareness', 'total_collapse_metric', 'mean_grain_activation', 
                     'mean_grain_saturation', 'collapse_events', 'active_nodes',
                     'system_entropy', 'system_temperature'],
            'field': ['mean_field_resonance', 'mean_field_momentum', 'mean_unresolved_tension',
                     'field_coherence', 'continuous_flow_rate', 'phase_diversity'],
            'toroidal': ['mean_rotational_curvature', 'mean_phase_continuity', 'vortex_count',
                        'mean_vortex_strength', 'toroidal_coherence', 'phase_coherence',
                        'major_circle_flow', 'minor_circle_flow', 'toroidal_flux'],
            'polarity': ['mean_polarity', 'polarity_diversity', 'structure_ratio',
                        'decay_ratio', 'neutral_ratio'],
            'circular': ['circular_recursion_count', 'circular_ancestry_count',
                        'polarity_wraparound_events', 'circular_coherence',
                        'polar_extremes_ratio', 'phase_reversal_count'],
            'ancestry': ['self_reference_count', 'recursive_collapse_count',
                        'average_ancestry_size']
        }
        
        # Add metrics by category
        for category, metric_keys in metric_categories.items():
            summary['metrics'][category] = {}
            for key in metric_keys:
                if key in self.metrics:
                    summary['metrics'][category][key] = self.metrics[key]
        
        # Group structures by category
        structure_categories = {
            'basic': ['attractors', 'confinement_zones', 'recurrence_patterns'],
            'field': ['resonant_regions', 'momentum_fields', 'gradient_flows', 
                     'field_emergent_patterns'],
            'toroidal': ['vortices', 'phase_locked_regions', 'toroidal_mode_rings',
                        'toroidal_domains', 'toroidal_vortices', 'toroidal_clusters',
                        'phase_domains', 'cross_phase_structures'],
            'pathways': ['structure_pathways', 'decay_pathways'],
            'circular': ['polarity_wraparound_paths', 'circular_ancestry_pairs',
                        'circular_recursion_grains']
        }
        
        # Add structure counts by category
        for category, structure_keys in structure_categories.items():
            summary['structures'][category] = {}
            for key in structure_keys:
                if key in self.structures:
                    if isinstance(self.structures[key], (list, set)):
                        summary['structures'][category][key] = len(self.structures[key])
                    elif isinstance(self.structures[key], dict):
                        summary['structures'][category][key] = len(self.structures[key])
        
        # Add tracker stats
        summary['trackers'] = {
            'circular': {
                'polarity_wraparound_events': len(self.circular_tracker.polarity_wraparound_events),
                'circular_ancestry_patterns': len(self.circular_tracker.circular_ancestry_patterns),
                'phase_coherence_shifts': len(self.circular_tracker.phase_coherence_shifts)
            },
            'ancestry': {
                'ancestry_events': len(self.ancestry_tracker.ancestry_events),
                'self_reference_events': len(self.ancestry_tracker.self_reference_events),
                'recursive_collapse_events': len(self.ancestry_tracker.recursive_collapse_events),
                'shared_ancestry_pairs': len(self.ancestry_tracker.shared_ancestry_patterns)
            },
            'toroidal': {
                'vortex_events': len(self.toroidal_tracker.vortex_events),
                'pathway_events': len(self.toroidal_tracker.pathway_events)
            }
        }
        
        return summary
    
    def get_circular_recursion_analysis(self) -> Dict[str, Any]:
        """
        Get a detailed analysis of circular recursion patterns.
        
        Returns:
            Dictionary with circular recursion analysis
        """
        # Collect basic metrics
        circular_metrics = {
            'circular_recursion_count': self.metrics.get('circular_recursion_count', 0),
            'circular_ancestry_count': self.metrics.get('circular_ancestry_count', 0),
            'polarity_wraparound_events': self.metrics.get('polarity_wraparound_events', 0),
            'circular_coherence': self.metrics.get('circular_coherence', 0.0),
            'polar_extremes_ratio': self.metrics.get('polar_extremes_ratio', 0.0),
            'phase_reversal_count': self.metrics.get('phase_reversal_count', 0)
        }
        
        # Collect recent events
        recent_wraparounds = list(self.circular_tracker.polarity_wraparound_events)
        recent_ancestry_patterns = list(self.circular_tracker.circular_ancestry_patterns)
        recent_phase_shifts = list(self.circular_tracker.phase_coherence_shifts)
        
        # Build the analysis
        analysis = {
            'metrics': circular_metrics,
            'structures': {
                'polarity_wraparound_paths': self.structures.get('polarity_wraparound_paths', []),
                'circular_ancestry_pairs': self.structures.get('circular_ancestry_pairs', []),
                'circular_recursion_grains': list(self.structures.get('circular_recursion_grains', set()))
            },
            'recent_events': {
                'wraparounds': recent_wraparounds,
                'ancestry_patterns': recent_ancestry_patterns,
                'phase_shifts': recent_phase_shifts
            },
            'event_counts': {
                'wraparound_count': len(recent_wraparounds),
                'ancestry_pattern_count': len(recent_ancestry_patterns),
                'phase_shift_count': len(recent_phase_shifts)
            },
            'circular_memory': {
                'grain_count': len(self.circular_tracker.circular_memory),
                'high_circular_factor_count': sum(1 for data in self.circular_tracker.circular_memory.values() 
                                              if data['circular_factor'] > 0.6)
            }
        }
        
        # Add history traces (last 20 points)
        if 'circular_recursion_count' in self.history['metrics']:
            analysis['history'] = {
                'time': self.history['time'][-20:],
                'circular_recursion_count': self.history['metrics']['circular_recursion_count'][-20:],
                'circular_coherence': self.history['metrics']['circular_coherence'][-20:] 
                                    if 'circular_coherence' in self.history['metrics'] else [],
                'polarity_wraparound_events': self.history['metrics']['polarity_wraparound_events'][-20:]
                                            if 'polarity_wraparound_events' in self.history['metrics'] else []
            }
        
        return analysis
    
    def get_ancestry_analysis(self) -> Dict[str, Any]:
        """
        Get a detailed analysis of ancestry patterns.
        
        Returns:
            Dictionary with ancestry analysis
        """
        # Collect basic metrics
        ancestry_metrics = {
            'self_reference_count': self.metrics.get('self_reference_count', 0),
            'recursive_collapse_count': self.metrics.get('recursive_collapse_count', 0),
            'average_ancestry_size': self.metrics.get('average_ancestry_size', 0.0),
            'shared_ancestry_strength': self.ancestry_tracker.ancestry_metrics.get('shared_ancestry_strength', 0.0)
        }
        
        # Add depth distribution
        ancestry_metrics['ancestry_depth_distribution'] = dict(self.ancestry_tracker.ancestry_metrics['ancestry_depth_distribution'])
        
        # Collect recent events
        recent_ancestry_events = list(self.ancestry_tracker.ancestry_events)
        recent_self_reference = list(self.ancestry_tracker.self_reference_events)
        recent_recursive_collapse = list(self.ancestry_tracker.recursive_collapse_events)
        
        # Build the analysis
        analysis = {
            'metrics': ancestry_metrics,
            'shared_ancestry': {
                'pair_count': len(self.ancestry_tracker.shared_ancestry_patterns),
                'top_pairs': sorted(self.ancestry_tracker.shared_ancestry_patterns.items(), 
                                  key=lambda x: x[1], reverse=True)[:10]
            },
            'recent_events': {
                'ancestry_changes': recent_ancestry_events,
                'self_references': recent_self_reference,
                'recursive_collapses': recent_recursive_collapse
            },
            'event_counts': {
                'ancestry_change_count': len(recent_ancestry_events),
                'self_reference_count': len(recent_self_reference),
                'recursive_collapse_count': len(recent_recursive_collapse)
            }
        }
        
        # Add history traces (last 20 points)
        if 'self_reference_count' in self.history['metrics']:
            analysis['history'] = {
                'time': self.history['time'][-20:],
                'self_reference_count': self.history['metrics']['self_reference_count'][-20:],
                'recursive_collapse_count': self.history['metrics']['recursive_collapse_count'][-20:]
                                          if 'recursive_collapse_count' in self.history['metrics'] else [],
                'average_ancestry_size': self.history['metrics']['average_ancestry_size'][-20:]
                                       if 'average_ancestry_size' in self.history['metrics'] else []
            }
        
        return analysis
    
    def get_toroidal_analysis(self) -> Dict[str, Any]:
        """
        Get a detailed analysis of toroidal dynamics.
        
        Returns:
            Dictionary with toroidal analysis
        """
        # Collect vortex metrics
        vortex_metrics = {
            'vortex_count': self.metrics.get('vortex_count', 0),
            'mean_vortex_strength': self.metrics.get('mean_vortex_strength', 0.0),
            'vortex_coherence': self.metrics.get('vortex_coherence', 0.0),
            'vortex_polarity_alignment': self.metrics.get('vortex_polarity_alignment', 0.0),
            'clockwise_ratio': self.metrics.get('clockwise_ratio', 0.0)
        }
        
        # Collect pathway metrics
        pathway_metrics = {
            'structure_pathway_count': self.metrics.get('structure_pathway_count', 0),
            'decay_pathway_count': self.metrics.get('decay_pathway_count', 0),
            'mean_structure_pathway_length': self.metrics.get('mean_structure_pathway_length', 0.0),
            'mean_decay_pathway_length': self.metrics.get('mean_decay_pathway_length', 0.0),
            'pathway_polarity_alignment': self.metrics.get('pathway_polarity_alignment', 0.0)
        }
        
        # Collect phase metrics
        phase_metrics = {
            'phase_coherence': self.metrics.get('phase_coherence', 0.0),
            'toroidal_coherence': self.metrics.get('toroidal_coherence', 0.0),
            'phase_diversity': self.metrics.get('phase_diversity', 0.0),
            'phase_stability': self.metrics.get('mean_phase_stability', 0.0)
        }
        
        # Collect structures
        vortex_structures = self.structures.get('vortices', [])
        structure_pathways = self.structures.get('structure_pathways', [])
        decay_pathways = self.structures.get('decay_pathways', [])
        
        # Collect recent events
        recent_vortex_events = list(self.toroidal_tracker.vortex_events)
        recent_pathway_events = list(self.toroidal_tracker.pathway_events)
        
        # Build the analysis
        analysis = {
            'metrics': {
                'vortex': vortex_metrics,
                'pathway': pathway_metrics,
                'phase': phase_metrics
            },
            'structures': {
                'vortices': vortex_structures,
                'structure_pathways': structure_pathways,
                'decay_pathways': decay_pathways
            },
            'recent_events': {
                'vortex_events': recent_vortex_events,
                'pathway_events': recent_pathway_events
            },
            'event_counts': {
                'vortex_event_count': len(recent_vortex_events),
                'pathway_event_count': len(recent_pathway_events)
            }
        }
        
        # Add history traces (last 20 points)
        if 'vortex_count' in self.history['metrics']:
            analysis['history'] = {
                'time': self.history['time'][-20:],
                'vortex_count': self.history['metrics']['vortex_count'][-20:],
                'phase_coherence': self.history['metrics']['phase_coherence'][-20:]
                                 if 'phase_coherence' in self.history['metrics'] else [],
                'toroidal_coherence': self.history['metrics']['toroidal_coherence'][-20:]
                                    if 'toroidal_coherence' in self.history['metrics'] else []
            }
        
        return analysis
    
    def export_history(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Export simulation history to a file or return as dictionary.
        
        Args:
            filename: Optional filename to write to
            
        Returns:
            Dictionary with history data
        """
        # Create history export with key metrics and events
        export_data = {
            'metadata': {
                'steps': self.step_count,
                'duration': self.time,
                'real_time_elapsed': time.time() - self.start_time
            },
            'time_series': {
                'time': self.history['time'],
                'metrics': {
                    # Basic metrics
                    'total_awareness': self.history['metrics'].get('total_awareness', []),
                    'mean_grain_activation': self.history['metrics'].get('mean_grain_activation', []),
                    'mean_grain_saturation': self.history['metrics'].get('mean_grain_saturation', []),
                    'collapse_events': self.history['metrics'].get('collapse_events', []),
                    'active_nodes': self.history['metrics'].get('active_nodes', []),
                    
                    # Field metrics
                    'field_coherence': self.history['metrics'].get('field_coherence', []),
                    'phase_coherence': self.history['metrics'].get('phase_coherence', []),
                    'toroidal_coherence': self.history['metrics'].get('toroidal_coherence', []),
                    
                    # Polarity metrics
                    'mean_polarity': self.history['metrics'].get('mean_polarity', []),
                    'structure_ratio': self.history['metrics'].get('structure_ratio', []),
                    'decay_ratio': self.history['metrics'].get('decay_ratio', []),
                    
                    # Circular recursion metrics
                    'circular_recursion_count': self.history['metrics'].get('circular_recursion_count', []),
                    'circular_coherence': self.history['metrics'].get('circular_coherence', []),
                    'polarity_wraparound_events': self.history['metrics'].get('polarity_wraparound_events', []),
                    
                    # Ancestry metrics
                    'self_reference_count': self.history['metrics'].get('self_reference_count', []),
                    'average_ancestry_size': self.history['metrics'].get('average_ancestry_size', [])
                },
                'structures': {
                    'vortex_count': self.history['structures'].get('vortices', []),
                    'structure_pathway_count': self.history['structures'].get('structure_pathways', []),
                    'decay_pathway_count': self.history['structures'].get('decay_pathways', []),
                    'circular_recursion_grains': self.history['structures'].get('circular_recursion_grains', [])
                }
            },
            'events': self.history['events'][-100:]  # Limit to last 100 events to avoid huge files
        }
        
        # Write to file if filename provided
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(export_data, f)
            except Exception as e:
                print(f"Error writing to file: {e}")
        
        return export_data
    
    def _initialize_visualization_fields(self, resolution=100):
        """
        Initialize visualization fields for torus visualizers.
        Sets up the fields that will map metrics to visualization data.
        
        Args:
            resolution: Resolution of field grids for visualization
        """
        # Store resolution
        self.viz_resolution = resolution
        
        # Create grid coordinates
        self._theta_grid, self._phi_grid = np.meshgrid(
            np.linspace(0, 2*np.pi, resolution),
            np.linspace(0, 2*np.pi, resolution)
        )
        
        # Initialize field arrays needed by visualizers
        self._awareness_field = np.zeros((resolution, resolution))
        self._phase_field = np.zeros((resolution, resolution))
        self._tension_field = np.zeros((resolution, resolution))
        self._backflow_field = np.zeros((resolution, resolution))
        self._stability_field = np.zeros((resolution, resolution))
        self._void_field = np.zeros((resolution, resolution))
        
        # NEW for circular recursion
        self._circular_recursion_field = np.zeros((resolution, resolution))
        self._ancestry_field = np.zeros((resolution, resolution))
        
        # Vector fields for flow visualization
        self._vector_field_theta = np.zeros((resolution, resolution))
        self._vector_field_phi = np.zeros((resolution, resolution))
        
        # Aliases used by some visualizers
        self._polarity_field_u = self._vector_field_theta
        self._polarity_field_v = self._vector_field_phi
    
    def _update_visualization_fields(self, manifold):
        """
        Update visualization fields based on current metrics and structures.
        Maps the state data to field representations for visualizers.
        
        Args:
            manifold: RelationalManifold for position reference
        """
        # Reset fields
        self._reset_visualization_fields()
        
        # Map metrics to fields
        self._map_metrics_to_fields()
        
        # Map structures to fields
        self._map_structures_to_fields(manifold)
        
        # Map circular recursion to fields (NEW)
        self._map_circular_recursion_to_fields(manifold)
        
        # Map ancestry to fields (NEW)
        self._map_ancestry_to_fields(manifold)
        
        # Generate vector fields
        self._generate_vector_fields(manifold)
        
        # Apply post-processing
        self._post_process_fields()
    
    def _reset_visualization_fields(self):
        """Reset all field arrays to zero"""
        # Reset all fields to zero
        self._awareness_field.fill(0)
        self._phase_field.fill(0)
        self._tension_field.fill(0)
        self._backflow_field.fill(0)
        self._stability_field.fill(0)
        self._void_field.fill(0)
        self._circular_recursion_field.fill(0)  # NEW
        self._ancestry_field.fill(0)  # NEW
        self._vector_field_theta.fill(0)
        self._vector_field_phi.fill(0)
    
    def _map_metrics_to_fields(self):
        """Map state metrics to field values"""
        # Apply awareness metrics
        if self.metrics['total_awareness'] > 0 and self.metrics['active_nodes'] > 0:
            # Create a base awareness level
            base_awareness = self.metrics['total_awareness'] / self.metrics['active_nodes']
            
            # Apply to awareness field
            self._awareness_field.fill(base_awareness * 0.5)  # Scale for better visualization
        
        # Apply phase coherence to phase field
        coherence = self.metrics.get('phase_coherence', 0.5)
        # Create phase pattern based on coherence
        phase_pattern = np.sin(self._theta_grid * self._phi_grid * coherence * 4)
        self._phase_field = (phase_pattern + 1) * np.pi  # Map to 0-2π range
        
        # Apply tension metrics to tension field
        tension = self.metrics.get('structural_tension_mean', 0.0)
        self._tension_field.fill(tension)
        
        # Apply void metrics to void field
        void_ratio = self.metrics.get('void_affected_node_ratio', 0.0)
        self._void_field.fill(void_ratio)
        
        # Apply stability metrics to stability field
        stability = self.metrics.get('mean_phase_stability', 0.5)
        self._stability_field.fill(stability)
        
        # Apply backflow metrics to backflow field
        if 'decay_ratio' in self.metrics:
            # Use decay ratio as proxy for backflow potential
            self._backflow_field.fill(self.metrics['decay_ratio'])
    
    def _map_structures_to_fields(self, manifold):
        """
        Map state structures to field patterns.
        
        Args:
            manifold: RelationalManifold for position reference
        """
        # Map vortices to field patterns
        if 'vortices' in self.structures and self.structures['vortices']:
            for vortex in self.structures['vortices']:
                # Skip vortices without position info
                if 'center_id' not in vortex and ('theta' not in vortex or 'phi' not in vortex):
                    continue
                
                # Get vortex position
                if 'center_id' in vortex:
                    # Get position from grain
                    center_id = vortex['center_id']
                    if not hasattr(manifold, 'grains') or center_id not in manifold.grains:
                        continue
                    
                    # Try to get position from toroidal phase
                    theta = phi = None
                    
                    # From grain properties
                    if hasattr(manifold.grains[center_id], 'theta') and hasattr(manifold.grains[center_id], 'phi'):
                        theta = manifold.grains[center_id].theta
                        phi = manifold.grains[center_id].phi
                    # From coordinator if available
                    elif hasattr(manifold, 'toroidal_coordinator'):
                        # Try synchronizing coordinates
                        manifold.toroidal_coordinator.synchronize_coordinates(center_id)
                        
                        # Try fetching again
                        if hasattr(manifold.grains[center_id], 'theta') and hasattr(manifold.grains[center_id], 'phi'):
                            theta = manifold.grains[center_id].theta
                            phi = manifold.grains[center_id].phi
                    
                    if theta is None or phi is None:
                        continue
                else:
                    # Use explicit theta/phi
                    theta = vortex['theta']
                    phi = vortex['phi']
                
                # Get vortex strength and radius
                strength = vortex.get('strength', 0.5)
                radius = vortex.get('radius', 0.5)
                
                # Scale radius based on grid resolution
                radius_scaled = radius * 2 * np.pi / 10
                
                # Apply vortex pattern to phase field
                for i in range(self.viz_resolution):
                    for j in range(self.viz_resolution):
                        grid_theta = self._theta_grid[i, j]
                        grid_phi = self._phi_grid[i, j]
                        
                        # Calculate toroidal distance (accounting for wrapping)
                        d_theta = min(abs(grid_theta - theta), 2*np.pi - abs(grid_theta - theta))
                        d_phi = min(abs(grid_phi - phi), 2*np.pi - abs(grid_phi - phi))
                        dist = np.sqrt(d_theta**2 + d_phi**2)
                        
                        # Skip points outside vortex radius
                        if dist > radius_scaled:
                            continue
                        
                        # Calculate influence based on distance
                        influence = (1 - dist/radius_scaled) * strength
                        
                        # Apply to phase field as a swirl pattern
                        angle = np.arctan2(grid_phi - phi, grid_theta - theta)
                        phase_shift = angle * influence * 2  # Create swirl effect
                        
                        # Apply to phase field
                        self._phase_field[i, j] += phase_shift
                        
                        # Apply to backflow field
                        self._backflow_field[i, j] = max(self._backflow_field[i, j], influence)
    
    def _map_circular_recursion_to_fields(self, manifold):
        """
        Map circular recursion patterns to visualization fields.
        
        Args:
            manifold: RelationalManifold for position reference
        """
        # For each grain with circular recursion factor
        for grain_id, grain in manifold.grains.items():
            # Skip grains without position info
            if not hasattr(grain, 'theta') or not hasattr(grain, 'phi'):
                continue
                
            # Get position
            grain_theta = grain.theta
            grain_phi = grain.phi
            
            # Get circular recursion factor
            circular_factor = 0.0
            
            if hasattr(grain, 'circular_recursion_factor'):
                circular_factor = grain.circular_recursion_factor
            elif grain_id in self.circular_tracker.circular_memory:
                circular_factor = self.circular_tracker.circular_memory[grain_id]['circular_factor']
            
            # Skip grains with little circular recursion
            if circular_factor < 0.3:
                continue
                
            # Get polarity for influence pattern
            polarity = getattr(grain, 'polarity', 0.0)
            
            # Create influence radius based on factor
            radius = 0.3 + circular_factor * 0.4  # Scale with factor
            radius_scaled = radius * 2 * np.pi / 10
            
            # Apply pattern to circular recursion field
            for i in range(self.viz_resolution):
                for j in range(self.viz_resolution):
                    grid_theta = self._theta_grid[i, j]
                    grid_phi = self._phi_grid[i, j]
                    
                    # Calculate toroidal distance (accounting for wrapping)
                    d_theta = min(abs(grid_theta - grain_theta), 2*np.pi - abs(grid_theta - grain_theta))
                    d_phi = min(abs(grid_phi - grain_phi), 2*np.pi - abs(grid_phi - grain_phi))
                    dist = np.sqrt(d_theta**2 + d_phi**2)
                    
                    # Skip points outside influence radius
                    if dist > radius_scaled:
                        continue
                    
                    # Calculate influence based on distance
                    influence = (1 - dist/radius_scaled) * circular_factor
                    
                    # Apply ripple pattern based on polarity
                    phase_ripple = np.sin(dist * 10 * (abs(polarity) + 0.5)) * 0.3 * influence
                    
                    # Apply to fields
                    self._circular_recursion_field[i, j] = max(
                        self._circular_recursion_field[i, j], 
                        influence
                    )
                    
                    # Add ripple effect to phase field
                    self._phase_field[i, j] += phase_ripple
    
    def _map_ancestry_to_fields(self, manifold):
        """
        Map ancestry patterns to visualization fields.
        
        Args:
            manifold: RelationalManifold for position reference
        """
        # For each grain with significant ancestry
        for grain_id, grain in manifold.grains.items():
            # Skip grains without position or ancestry
            if not hasattr(grain, 'theta') or not hasattr(grain, 'phi') or not hasattr(grain, 'ancestry'):
                continue
                
            # Skip grains with little ancestry
            if len(grain.ancestry) < 2:
                continue
                
            # Get position
            grain_theta = grain.theta
            grain_phi = grain.phi
            
            # Get ancestry size
            ancestry_size = len(grain.ancestry)
            
            # Is grain self-referential?
            is_self_ref = grain_id in grain.ancestry
            
            # Create influence radius based on ancestry size
            radius = 0.2 + min(0.6, ancestry_size * 0.05)  # Cap at reasonable size
            radius_scaled = radius * 2 * np.pi / 10
            
            # Calculate influence factor
            influence_factor = 0.3 + min(0.6, ancestry_size * 0.05)
            
            # Boost factor for self-reference
            if is_self_ref:
                influence_factor += 0.2
            
            # Apply pattern to ancestry field
            for i in range(self.viz_resolution):
                for j in range(self.viz_resolution):
                    grid_theta = self._theta_grid[i, j]
                    grid_phi = self._phi_grid[i, j]
                    
                    # Calculate toroidal distance (accounting for wrapping)
                    d_theta = min(abs(grid_theta - grain_theta), 2*np.pi - abs(grid_theta - grain_theta))
                    d_phi = min(abs(grid_phi - grain_phi), 2*np.pi - abs(grid_phi - grain_phi))
                    dist = np.sqrt(d_theta**2 + d_phi**2)
                    
                    # Skip points outside influence radius
                    if dist > radius_scaled:
                        continue
                    
                    # Calculate influence based on distance
                    influence = (1 - dist/radius_scaled) * influence_factor
                    
                    # Apply to ancestry field
                    self._ancestry_field[i, j] = max(
                        self._ancestry_field[i, j], 
                        influence
                    )
                    
                    # Add to stability field - ancestry increases stability
                    self._stability_field[i, j] += influence * 0.2
    
    def _generate_vector_fields(self, manifold):
        """
        Generate vector fields for visualization.
        
        Args:
            manifold: RelationalManifold for position reference
        """
        # Try to extract polarity data from manifold
        if hasattr(manifold, 'polarity_field') and hasattr(manifold.polarity_field, 'calculate_vector_field'):
            # Use polarity field's vector calculation if available
            try:
                theta_vectors, phi_vectors = manifold.polarity_field.calculate_vector_field(
                    self._theta_grid, self._phi_grid
                )
                
                # Apply to vector fields
                self._vector_field_theta = theta_vectors
                self._vector_field_phi = phi_vectors
            except:
                # Fallback to metric-based generation
                self._generate_vector_fields_from_metrics()
        else:
            # Use metric-based generation
            self._generate_vector_fields_from_metrics()
        
        # Update aliases
        self._polarity_field_u = self._vector_field_theta
        self._polarity_field_v = self._vector_field_phi
    
    def _generate_vector_fields_from_metrics(self):
        """Generate vector fields based on metrics for visualization"""
        # Get polarity metrics
        mean_polarity = self.metrics.get('mean_polarity', 0.0)
        structure_ratio = self.metrics.get('structure_ratio', 0.5)
        decay_ratio = self.metrics.get('decay_ratio', 0.5)
        
        # Create a simple gradient field based on polarity metrics
        # Structure bias creates flow in theta direction
        # Decay bias creates flow in phi direction
        theta_bias = (structure_ratio - 0.5) * 2.0  # -1 to 1 range
        phi_bias = (decay_ratio - 0.5) * 2.0        # -1 to 1 range
        
        # Apply to vector fields with some wave variation
        for i in range(self.viz_resolution):
            for j in range(self.viz_resolution):
                theta = self._theta_grid[i, j]
                phi = self._phi_grid[i, j]
                
                # Add wave variation for visual interest
                theta_var = np.sin(theta * 2 + phi) * 0.3
                phi_var = np.cos(phi * 2 + theta) * 0.3
                
                # Apply circular recursion influence
                circular_influence = self._circular_recursion_field[i, j]
                
                # Add circular swirl component based on recursion
                if circular_influence > 0.1:
                    # Create swirl effect - tangential to radius
                    center_theta = np.pi
                    center_phi = np.pi
                    
                    # Calculate angle to center
                    d_theta = theta - center_theta
                    d_phi = phi - center_phi
                    
                    # Create swirl vectors (perpendicular to radius)
                    swirl_theta = -d_phi * circular_influence * 0.5
                    swirl_phi = d_theta * circular_influence * 0.5
                    
                    # Add to variation
                    theta_var += swirl_theta
                    phi_var += swirl_phi
                
                # Combine base direction with variation
                self._vector_field_theta[i, j] = 0.5 * (theta_bias + theta_var)
                self._vector_field_phi[i, j] = 0.5 * (phi_bias + phi_var)
    
    def _post_process_fields(self):
        """Apply post-processing to fields for better visualization"""
        try:
            # Import here to avoid dependencies if scipy is not available
            from scipy.ndimage import gaussian_filter
            
            # Apply smoothing to fields
            self._awareness_field = gaussian_filter(self._awareness_field, sigma=2.0)
            self._phase_field = gaussian_filter(self._phase_field, sigma=1.5)
            self._tension_field = gaussian_filter(self._tension_field, sigma=2.0)
            self._backflow_field = gaussian_filter(self._backflow_field, sigma=2.0)
            self._stability_field = gaussian_filter(self._stability_field, sigma=2.0)
            self._void_field = gaussian_filter(self._void_field, sigma=2.0)
            self._circular_recursion_field = gaussian_filter(self._circular_recursion_field, sigma=1.5)
            self._ancestry_field = gaussian_filter(self._ancestry_field, sigma=1.5)
            
            # Apply smoothing to vector fields
            self._vector_field_theta = gaussian_filter(self._vector_field_theta, sigma=1.0)
            self._vector_field_phi = gaussian_filter(self._vector_field_phi, sigma=1.0)
        except ImportError:
            # Fallback if scipy not available - simple averaging smoothing
            self._simple_smooth(self._awareness_field)
            self._simple_smooth(self._phase_field)
            self._simple_smooth(self._tension_field)
            self._simple_smooth(self._backflow_field)
            self._simple_smooth(self._stability_field)
            self._simple_smooth(self._void_field)
            self._simple_smooth(self._circular_recursion_field)
            self._simple_smooth(self._ancestry_field)
            self._simple_smooth(self._vector_field_theta)
            self._simple_smooth(self._vector_field_phi)
        
        # Normalize fields to 0-1 range
        self._normalize_field(self._awareness_field)
        self._normalize_field(self._tension_field)
        self._normalize_field(self._backflow_field)
        self._normalize_field(self._stability_field)
        self._normalize_field(self._void_field)
        self._normalize_field(self._circular_recursion_field)
        self._normalize_field(self._ancestry_field)
        
        # Ensure phase field is modulo 2π
        self._phase_field = self._phase_field % (2 * np.pi)
        
        # Ensure periodicity at boundaries
        self._ensure_field_periodicity(self._awareness_field)
        self._ensure_field_periodicity(self._tension_field)
        self._ensure_field_periodicity(self._backflow_field)
        self._ensure_field_periodicity(self._stability_field)
        self._ensure_field_periodicity(self._void_field)
        self._ensure_field_periodicity(self._circular_recursion_field)
        self._ensure_field_periodicity(self._ancestry_field)
        self._ensure_field_periodicity(self._vector_field_theta)
        self._ensure_field_periodicity(self._vector_field_phi)
    
    def _simple_smooth(self, field):
        """
        Apply simple averaging smoothing to a field.
        
        Args:
            field: Field array to smooth
        """
        # Create a copy to avoid modifying while iterating
        orig = field.copy()
        resolution = field.shape[0]
        
        for i in range(resolution):
            for j in range(resolution):
                # Get neighbors (with wrapping)
                neighbors = [
                    orig[i, j],  # Center
                    orig[(i-1) % resolution, j],  # Up
                    orig[(i+1) % resolution, j],  # Down
                    orig[i, (j-1) % resolution],  # Left
                    orig[i, (j+1) % resolution],  # Right
                ]
                
                # Average of neighbors
                field[i, j] = sum(neighbors) / len(neighbors)
    
    def _normalize_field(self, field):
        """
        Normalize a field to the 0-1 range.
        
        Args:
            field: Field array to normalize
        """
        field_min = np.min(field)
        field_max = np.max(field)
        
        if field_max > field_min:
            # Normalize to 0-1 range
            field[:] = (field - field_min) / (field_max - field_min)
    
    def _ensure_field_periodicity(self, field):
        """
        Ensure a field is periodic at boundaries.
        
        Args:
            field: Field array to ensure periodicity
        """
        # Smooth field across theta boundary (left-right edges)
        for i in range(self.viz_resolution):
            # Average edge values
            left_edge = np.mean(field[i, 0:3])
            right_edge = np.mean(field[i, -3:])
            avg_edge = (left_edge + right_edge) / 2
            
            # Apply smoothing to boundaries
            for j in range(3):
                weight = (3-j)/3
                field[i, j] = field[i, j] * (1-weight) + avg_edge * weight
                field[i, -(j+1)] = field[i, -(j+1)] * (1-weight) + avg_edge * weight
        
        # Smooth field across phi boundary (top-bottom edges)
        for j in range(self.viz_resolution):
            # Average edge values
            top_edge = np.mean(field[0:3, j])
            bottom_edge = np.mean(field[-3:, j])
            avg_edge = (top_edge + bottom_edge) / 2
            
            # Apply smoothing to boundaries
            for i in range(3):
                weight = (3-i)/3
                field[i, j] = field[i, j] * (1-weight) + avg_edge * weight
                field[-(i+1), j] = field[-(i+1), j] * (1-weight) + avg_edge * weight

# Factory function for easier creation
def create_simulation_state() -> SimulationState:
    """Create a new simulation state projection"""
    return SimulationState()