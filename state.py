"""
SimulationState - Projection field for the Collapse Geometry framework

Acts as a recursive projection of the system's state, capturing emergent patterns
without disturbing the core dynamics. Not an external observer but a reflection
of the system's own self-observation capacity.

This version is enhanced with visualization field support that translates metrics and
structures into visual representations for the visualizers.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
import time
import json
import math


class SimulationState:
    """
    Projection field for the Collapse Geometry framework.
    
    Captures and records the emergent patterns, structural transitions,
    and relational dynamics of the manifold without imposing external
    perspective. Functions as a memory surface that the system itself
    can reference, creating recursive feedback loops of self-reference.
    
    Enhanced with visualization field support to translate metrics and structures
    into visual representations.
    """
    
    def __init__(self):
        """Initialize the projection field with empty metrics and structures"""
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
            'neutral_ratio': 0.0
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
            'decay_pathways': []
        }
        
        # Memory trace: historical record of system evolution
        self.history = {
            'time': [],
            'metrics': {},
            'structures': {},
            'events': []
        }
        
        # Specialized memory traces
        self.void_formation_history = []
        self.decay_emission_history = []
        self.incompatible_structure_history = []
        self.toroidal_vortex_history = []
        self.phase_transition_history = []
        self.toroidal_resonance_history = []
        self.polarity_shift_history = []
        
        # Initialize history metrics
        for key in self.metrics:
            self.history['metrics'][key] = []
        
        for key in self.structures:
            self.history['structures'][key] = []
            
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
        self.time = manifold.time
        self.step_count += 1
    
        # Capture system-level reflections from grains
        self._update_grain_metrics(manifold)
        
        # Capture config space metrics
        self._update_config_space_metrics(manifold)
        
        # Capture toroidal metrics from coordinator
        self._update_toroidal_metrics(manifold)
        
        # Capture polarity metrics
        self._update_polarity_metrics(manifold)
        
        # Calculate system entropy and temperature
        self._calculate_emergent_thermodynamics(manifold)
        
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
        for node in manifold.grains.values():
            # Basic grain properties
            total_awareness += node.awareness
            total_collapse_metric += node.collapse_metric
            total_grain_activation += node.grain_activation
            total_grain_saturation += node.grain_saturation
            
            # Track polarity if available
            node_polarity = getattr(node, 'polarity', 0.0)
            total_polarity += node_polarity
            polarity_values.append(node_polarity)
            
            # Count by polarity category
            if node_polarity > 0.2:
                structure_count += 1
            elif node_polarity < -0.2:
                decay_count += 1
            else:
                neutral_count += 1
            
            # Optional field properties if available
            if hasattr(node, 'field_resonance'):
                total_field_resonance += node.field_resonance
                
            if hasattr(node, 'field_momentum') and isinstance(node.field_momentum, np.ndarray) and np.any(node.field_momentum):
                total_field_momentum += np.linalg.norm(node.field_momentum)
                
            if hasattr(node, 'unresolved_tension'):
                total_unresolved_tension += node.unresolved_tension
        
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
        
        if tension_count > 0:
            self.metrics['structural_tension_mean'] = total_tension / tension_count
        else:
            self.metrics['structural_tension_mean'] = 0.0
        
        # Capture vortex information from config space
        if hasattr(config_space, 'find_collapse_cascade_pathways'):
            # Get pathways from config space
            pathways = config_space.find_collapse_cascade_pathways()
            
            # Update metrics
            self.metrics['structure_pathway_count'] = len(pathways.get('structure', []))
            self.metrics['decay_pathway_count'] = len(pathways.get('decay', []))
            
            # Store structures
            self.structures['structure_pathways'] = pathways.get('structure', [])
            self.structures['decay_pathways'] = pathways.get('decay', [])
        
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
            
            if vortices:
                # Calculate vortex strength
                strengths = [abs(v.get('strength', 0.0)) for v in vortices]
                self.metrics['mean_vortex_strength'] = sum(strengths) / len(strengths)
            else:
                self.metrics['mean_vortex_strength'] = 0.0
        
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
        
        # Get coherence from coordinator
        if hasattr(coordinator, 'calculate_global_coherence'):
            self.metrics['toroidal_coherence'] = coordinator.calculate_global_coherence()
            
        # Phase coherence - delegate to manifold if available
        if hasattr(manifold, 'field_coherence'):
            self.metrics['phase_coherence'] = manifold.field_coherence
        
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
        
        # Get polarity-based structures if available
        if hasattr(polarity_field, 'find_structure_decay_regions'):
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
        
    def _calculate_emergent_thermodynamics(self, manifold):
        """
        Calculate emergent thermodynamic properties like entropy and temperature
        
        Args:
            manifold: RelationalManifold to project
        """
        # Project system entropy (S = -∫ρ(x,t)log(ρ(x,t) + ε)dx)
        entropy = 0.0
        epsilon = 1e-10  # Small constant to prevent log(0)
    
        for node in manifold.grains.values():
            if node.awareness > 0:
                entropy -= node.awareness * np.log(node.awareness + epsilon)
    
        self.metrics['system_entropy'] = entropy
    
        # Project system temperature (average grain activation)
        self.metrics['system_temperature'] = self.metrics['mean_grain_activation']
    
        # Project field coherence (as a simplified version using polarity alignment)
        field_coherence = 0.0
        
        polarity_values = []
        for node in manifold.grains.values():
            if hasattr(node, 'polarity'):
                polarity_values.append(node.polarity)
                
        if len(polarity_values) >= 2:
            # Calculate polarity alignment (values with same sign are aligned)
            aligned_count = 0
            total_pairs = 0
            
            for i, p1 in enumerate(polarity_values):
                for p2 in polarity_values[i+1:]:
                    # Check if same sign or both zero
                    if (p1 == 0 and p2 == 0) or (p1 * p2 > 0):
                        aligned_count += 1
                    total_pairs += 1
                    
            if total_pairs > 0:
                field_coherence = aligned_count / total_pairs
    
        self.metrics['field_coherence'] = field_coherence
        
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
        
        # Log special events to their specific histories
        if event_type == 'void_formation':
            self.void_formation_history.append(event)
        elif event_type == 'decay_emission':
            self.decay_emission_history.append(event)
        elif event_type == 'incompatible_structure':
            self.incompatible_structure_history.append(event)
        elif event_type == 'toroidal_vortex':
            self.toroidal_vortex_history.append(event)
        elif event_type == 'phase_transition':
            self.phase_transition_history.append(event)
        elif event_type == 'toroidal_resonance':
            self.toroidal_resonance_history.append(event)
        elif event_type == 'polarity_shift':
            self.polarity_shift_history.append(event)
            
    def log_phase_transition(self, previous_coherence: float, new_coherence: float,
                            transition_type: str, affected_grains: List[str] = None):
        """
        Log a phase transition event
        
        Args:
            previous_coherence: Previous phase coherence value
            new_coherence: New phase coherence value
            transition_type: Type of transition ('ordering', 'disordering')
            affected_grains: List of affected grain IDs
        """
        event = {
            'type': 'phase_transition',
            'time': self.time,
            'step': self.step_count,
            'data': {
                'previous_coherence': previous_coherence,
                'new_coherence': new_coherence,
                'change': abs(new_coherence - previous_coherence),
                'transition_type': transition_type,
                'affected_grains': affected_grains or []
            }
        }
        
        self.history['events'].append(event)
        self.phase_transition_history.append(event)
        
    def log_toroidal_resonance(self, theta_mode: int, phi_mode: int, 
                              ratio: float, strength: float):
        """
        Log a toroidal resonance event
        
        Args:
            theta_mode: Theta mode number
            phi_mode: Phi mode number
            ratio: Ratio between modes
            strength: Strength of resonance
        """
        event = {
            'type': 'toroidal_resonance',
            'time': self.time,
            'step': self.step_count,
            'data': {
                'theta_mode': theta_mode,
                'phi_mode': phi_mode,
                'ratio': ratio,
                'strength': strength
            }
        }
        
        self.history['events'].append(event)
        self.toroidal_resonance_history.append(event)
        
    def log_polarity_shift(self, previous_polarity: float, new_polarity: float,
                          shift_magnitude: float, affected_grains: List[str] = None):
        """
        Log a significant polarity shift event
        
        Args:
            previous_polarity: Previous average polarity value
            new_polarity: New average polarity value
            shift_magnitude: Magnitude of the shift
            affected_grains: List of affected grain IDs
        """
        event = {
            'type': 'polarity_shift',
            'time': self.time,
            'step': self.step_count,
            'data': {
                'previous_polarity': previous_polarity,
                'new_polarity': new_polarity,
                'shift_magnitude': shift_magnitude,
                'direction': 'structure' if new_polarity > previous_polarity else 'decay',
                'affected_grains': affected_grains or []
            }
        }
        
        self.history['events'].append(event)
        self.polarity_shift_history.append(event)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current simulation state projection"""
        summary = {
            'time': self.time,
            'step_count': self.step_count,
            'elapsed_real_time': time.time() - self.start_time,
            'metrics': self.metrics,
            'structures': {
                'attractors': len(self.structures['attractors']),
                'confinement_zones': len(self.structures['confinement_zones']),
                'recurrence_patterns': len(self.structures['recurrence_patterns']),
                'resonant_regions': len(self.structures.get('resonant_regions', [])),
                'field_emergent_patterns': len(self.structures.get('field_emergent_patterns', [])),
                'vortices': len(self.structures.get('vortices', [])),
                'phase_locked_regions': len(self.structures.get('phase_locked_regions', [])),
                'void_regions': len(self.structures.get('void_regions', [])),
                'void_clusters': len(self.structures.get('void_clusters', [])),
                'structural_alignment_clusters': len(self.structures.get('structural_alignment_clusters', [])),
                'structure_pathways': len(self.structures.get('structure_pathways', [])),
                'decay_pathways': len(self.structures.get('decay_pathways', []))
            }
        }
        
        # Add polarity metrics if available
        if 'mean_polarity' in self.metrics:
            summary['polarity_metrics'] = {
                'mean_polarity': self.metrics['mean_polarity'],
                'polarity_diversity': self.metrics['polarity_diversity'],
                'structure_ratio': self.metrics['structure_ratio'],
                'decay_ratio': self.metrics['decay_ratio'],
                'neutral_ratio': self.metrics['neutral_ratio']
            }
        
        # Add rotational reflections if available
        if 'mean_rotational_curvature' in self.metrics:
            summary['rotational_metrics'] = {
                'mean_rotational_curvature': self.metrics['mean_rotational_curvature'],
                'mean_phase_continuity': self.metrics['mean_phase_continuity'],
                'vortex_count': self.metrics['vortex_count'],
                'mean_vortex_strength': self.metrics['mean_vortex_strength'],
                'dominant_theta_mode': self.metrics['dominant_theta_mode'],
                'dominant_phi_mode': self.metrics['dominant_phi_mode'],
                'toroidal_coherence': self.metrics['toroidal_coherence']
            }
        
        # Add toroidal reflections
        if 'phase_coherence' in self.metrics:
            summary['toroidal_metrics'] = {
                'phase_coherence': self.metrics['phase_coherence'],
                'toroidal_domain_count': self.metrics['toroidal_domain_count'],
                'toroidal_vortex_count': self.metrics['toroidal_vortex_count'],
                'toroidal_cluster_count': self.metrics['toroidal_cluster_count'],
                'major_circle_flow': self.metrics['major_circle_flow'],
                'minor_circle_flow': self.metrics['minor_circle_flow'],
                'toroidal_flux': self.metrics['toroidal_flux'],
                'mean_phase_stability': self.metrics['mean_phase_stability'],
                'cross_phase_structure_count': self.metrics['cross_phase_structure_count']
            }
            
            # Add event counts
            summary['toroidal_metrics']['toroidal_vortex_events'] = len(self.toroidal_vortex_history)
            summary['toroidal_metrics']['phase_transition_events'] = len(self.phase_transition_history)
            summary['toroidal_metrics']['toroidal_resonance_events'] = len(self.toroidal_resonance_history)
        
        # Add Void-Decay reflections if available
        if 'void_region_count' in self.metrics:
            summary['void_decay_metrics'] = {
                'void_region_count': self.metrics['void_region_count'],
                'decay_particle_count': self.metrics['decay_particle_count'],
                'mean_void_strength': self.metrics['mean_void_strength'],
                'mean_void_radius': self.metrics['mean_void_radius'],
                'incompatible_structure_rate': self.metrics['incompatible_structure_rate'],
                'alignment_failure_rate': self.metrics['alignment_failure_rate'],
                'decay_emission_rate': self.metrics['decay_emission_rate'],
                'void_affected_node_ratio': self.metrics['void_affected_node_ratio'],
                'structural_tension_mean': self.metrics['structural_tension_mean']
            }
            
            # Add void formation and decay emission counts
            summary['void_decay_metrics']['void_formation_count'] = len(self.void_formation_history)
            summary['void_decay_metrics']['decay_emission_count'] = len(self.decay_emission_history)
            summary['void_decay_metrics']['incompatible_structure_count'] = len(self.incompatible_structure_history)
        
        return summary
    
    def export_history(self, filename: str = None, include_void_decay: bool = True, 
                      include_toroidal: bool = True, include_polarity: bool = True) -> Dict[str, Any]:
        """
        Export the simulation history to a file or return as dictionary
        
        Args:
            filename: Output filename (if None, just returns the data)
            include_void_decay: Whether to include Void-Decay histories
            include_toroidal: Whether to include toroidal dynamics histories
            include_polarity: Whether to include polarity dynamics histories
            
        Returns:
            Dictionary of history data
        """
        export_data = {
            'time': self.history['time'],
            'metrics': self.history['metrics'],
            'structures': self.history['structures'],
            'events': self.history['events'][:100]  # Limit event export to prevent huge files
        }
        
        # Add Void-Decay histories if requested
        if include_void_decay:
            export_data['void_formation_history'] = self.void_formation_history
            export_data['decay_emission_history'] = self.decay_emission_history
            export_data['incompatible_structure_history'] = self.incompatible_structure_history
        
        # Add toroidal histories if requested
        if include_toroidal:
            export_data['toroidal_vortex_history'] = self.toroidal_vortex_history
            export_data['phase_transition_history'] = self.phase_transition_history
            export_data['toroidal_resonance_history'] = self.toroidal_resonance_history
            
        # Add polarity histories if requested
        if include_polarity:
            export_data['polarity_shift_history'] = self.polarity_shift_history
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(export_data, f)
        
        return export_data
    
    def get_toroidal_analysis(self) -> Dict[str, Any]:
        """
        Get enhanced toroidal analysis data for visualization or further processing
        
        Returns:
            Dictionary with comprehensive toroidal analysis results
        """
        # Basic toroidal metrics
        toroidal_metrics = {
            'phase_coherence': self.metrics.get('phase_coherence', 0.0),
            'major_circle_flow': self.metrics.get('major_circle_flow', 0.0),
            'minor_circle_flow': self.metrics.get('minor_circle_flow', 0.0),
            'toroidal_flux': self.metrics.get('toroidal_flux', 0.0),
            'toroidal_domain_count': self.metrics.get('toroidal_domain_count', 0),
            'toroidal_vortex_count': self.metrics.get('toroidal_vortex_count', 0),
            'toroidal_cluster_count': self.metrics.get('toroidal_cluster_count', 0),
            'mean_phase_stability': self.metrics.get('mean_phase_stability', 0.0),
            'cross_phase_structure_count': self.metrics.get('cross_phase_structure_count', 0),
            'theta_mode_strength': self.metrics.get('theta_mode_strength', 0.0),
            'phi_mode_strength': self.metrics.get('phi_mode_strength', 0.0),
            'dominant_theta_mode': self.metrics.get('dominant_theta_mode', 0),
            'dominant_phi_mode': self.metrics.get('dominant_phi_mode', 0),
            'toroidal_coherence': self.metrics.get('toroidal_coherence', 0.0)
        }
        
        # Get toroidal structures
        toroidal_structures = {
            'theta_slices': self.structures.get('theta_slices', []),
            'phi_slices': self.structures.get('phi_slices', []),
            'phase_locked_regions': self.structures.get('phase_locked_regions', []),
            'toroidal_domains': self.structures.get('toroidal_domains', []),
            'toroidal_vortices': self.structures.get('toroidal_vortices', []),
            'toroidal_clusters': self.structures.get('toroidal_clusters', []),
            'phase_domains': self.structures.get('phase_domains', []),
            'cross_phase_structures': self.structures.get('cross_phase_structures', []),
            'structure_pathways': self.structures.get('structure_pathways', []),
            'decay_pathways': self.structures.get('decay_pathways', [])
        }
        
        # Get recent toroidal events
        recent_events = {
            'toroidal_vortex_events': self.toroidal_vortex_history[-min(10, len(self.toroidal_vortex_history)):],
            'phase_transition_events': self.phase_transition_history[-min(10, len(self.phase_transition_history)):],
            'toroidal_resonance_events': self.toroidal_resonance_history[-min(10, len(self.toroidal_resonance_history)):]
        }
        
        # Phase stability map (snapshot of current values)
        phase_stability = self.structures.get('phase_stability_map', {})
        
        return {
            'metrics': toroidal_metrics,
            'structures': toroidal_structures,
            'recent_events': recent_events,
            'phase_stability': phase_stability,
            'event_counts': {
                'toroidal_vortex_count': len(self.toroidal_vortex_history),
                'phase_transition_count': len(self.phase_transition_history),
                'toroidal_resonance_count': len(self.toroidal_resonance_history)
            }
        }
    
    def get_void_decay_analysis(self) -> Dict[str, Any]:
        """
        Get Void-Decay analysis data for visualization or further processing
        
        Returns:
            Dictionary with Void-Decay analysis results
        """
        # Basic void metrics
        void_metrics = {
            'void_region_count': self.metrics.get('void_region_count', 0),
            'decay_particle_count': self.metrics.get('decay_particle_count', 0),
            'mean_void_strength': self.metrics.get('mean_void_strength', 0.0),
            'mean_void_radius': self.metrics.get('mean_void_radius', 0.0),
            'void_affected_node_ratio': self.metrics.get('void_affected_node_ratio', 0.0),
            'structural_tension_mean': self.metrics.get('structural_tension_mean', 0.0),
            'incompatible_structure_rate': self.metrics.get('incompatible_structure_rate', 0.0),
            'alignment_failure_rate': self.metrics.get('alignment_failure_rate', 0.0),
            'decay_emission_rate': self.metrics.get('decay_emission_rate', 0.0)
        }
        
        # Get void and alignment structures
        void_structures = {
            'void_regions': self.structures.get('void_regions', []),
            'void_clusters': self.structures.get('void_clusters', []),
            'structural_alignment_clusters': self.structures.get('structural_alignment_clusters', [])
        }
        
        # Get recent void and decay events
        recent_void_formations = self.void_formation_history[-min(10, len(self.void_formation_history)):]
        recent_decay_emissions = self.decay_emission_history[-min(10, len(self.decay_emission_history)):]
        
        return {
            'metrics': void_metrics,
            'structures': void_structures,
            'recent_void_formations': recent_void_formations,
            'recent_decay_emissions': recent_decay_emissions,
            'void_formation_count': len(self.void_formation_history),
            'decay_emission_count': len(self.decay_emission_history),
            'incompatible_structure_count': len(self.incompatible_structure_history)
        }
    
    def get_polarity_analysis(self) -> Dict[str, Any]:
        """
        Get polarity analysis data for visualization or further processing
        
        Returns:
            Dictionary with polarity analysis results
        """
        # Basic polarity metrics
        polarity_metrics = {
            'mean_polarity': self.metrics.get('mean_polarity', 0.0),
            'polarity_diversity': self.metrics.get('polarity_diversity', 0.0),
            'structure_ratio': self.metrics.get('structure_ratio', 0.0),
            'decay_ratio': self.metrics.get('decay_ratio', 0.0),
            'neutral_ratio': self.metrics.get('neutral_ratio', 0.0)
        }
        
        # Get polarity structures
        polarity_structures = {
            'structure_regions': self.structures.get('structure_regions', []),
            'decay_regions': self.structures.get('decay_regions', []),
            'polarity_interface_regions': self.structures.get('polarity_interface_regions', []),
            'structure_pathways': self.structures.get('structure_pathways', []),
            'decay_pathways': self.structures.get('decay_pathways', [])
        }
        
        # Get recent polarity events
        recent_polarity_shifts = self.polarity_shift_history[-min(10, len(self.polarity_shift_history)):]
        
        return {
            'metrics': polarity_metrics,
            'structures': polarity_structures,
            'recent_polarity_shifts': recent_polarity_shifts,
            'polarity_shift_count': len(self.polarity_shift_history)
        }

    #---------------------------------------------------------------------------
    # Visualization Field Support - Maps metrics to visual fields
    #---------------------------------------------------------------------------
    
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
        if 'phase_coherence' in self.metrics:
            coherence = self.metrics['phase_coherence']
            # Create phase pattern based on coherence
            phase_pattern = np.sin(self._theta_grid * self._phi_grid * coherence * 4)
            self._phase_field = (phase_pattern + 1) * np.pi  # Map to 0-2π range
        
        # Apply tension metrics to tension field
        if 'structural_tension_mean' in self.metrics:
            self._tension_field.fill(self.metrics['structural_tension_mean'])
        elif 'mean_unresolved_tension' in self.metrics:
            self._tension_field.fill(self.metrics['mean_unresolved_tension'])
        
        # Apply void metrics to void field
        if 'void_affected_node_ratio' in self.metrics:
            self._void_field.fill(self.metrics['void_affected_node_ratio'])
        
        # Apply stability metrics to stability field
        if 'mean_phase_stability' in self.metrics:
            self._stability_field.fill(self.metrics['mean_phase_stability'])
        
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
                if 'center_node' not in vortex and ('theta' not in vortex or 'phi' not in vortex):
                    continue
                
                # Get vortex position
                if 'center_node' in vortex:
                    # Get position from grain
                    center_id = vortex['center_node']
                    if not hasattr(manifold, 'grains') or center_id not in manifold.grains:
                        continue
                    
                    # Try to get position from toroidal phase
                    theta = phi = None
                    
                    # From toroidal phase
                    if hasattr(manifold, 'toroidal_phase') and center_id in manifold.toroidal_phase:
                        theta, phi = manifold.toroidal_phase[center_id]
                    # From grain properties
                    elif hasattr(manifold.grains[center_id], 'theta') and hasattr(manifold.grains[center_id], 'phi'):
                        theta = manifold.grains[center_id].theta
                        phi = manifold.grains[center_id].phi
                    # From coordinator if available
                    elif hasattr(manifold, 'toroidal_coordinator') and hasattr(manifold.toroidal_coordinator, 'grain_positions'):
                        theta, phi = manifold.toroidal_coordinator.grain_positions.get(center_id, (None, None))
                    
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
        
        # Map void regions to void field
        if 'void_regions' in self.structures and self.structures['void_regions']:
            for region in self.structures['void_regions']:
                # Skip regions without position info
                if 'center' not in region and 'affected_nodes' not in region:
                    continue
                
                # Get region position
                center_theta = center_phi = None
                
                if 'center' in region:
                    # Use explicit center
                    if isinstance(region['center'], tuple) and len(region['center']) == 2:
                        center_theta, center_phi = region['center']
                elif 'affected_nodes' in region and hasattr(manifold, 'grains'):
                    # Calculate center from affected nodes
                    thetas = []
                    phis = []
                    
                    for node_id in region['affected_nodes']:
                        if node_id not in manifold.grains:
                            continue
                        
                        # Try to get position
                        grain_theta = grain_phi = None
                        
                        # From toroidal phase
                        if hasattr(manifold, 'toroidal_phase') and node_id in manifold.toroidal_phase:
                            grain_theta, grain_phi = manifold.toroidal_phase[node_id]
                        # From grain properties
                        elif hasattr(manifold.grains[node_id], 'theta') and hasattr(manifold.grains[node_id], 'phi'):
                            grain_theta = manifold.grains[node_id].theta
                            grain_phi = manifold.grains[node_id].phi
                        # From coordinator if available
                        elif hasattr(manifold, 'toroidal_coordinator') and hasattr(manifold.toroidal_coordinator, 'grain_positions'):
                            grain_theta, grain_phi = manifold.toroidal_coordinator.grain_positions.get(node_id, (None, None))
                        
                        if grain_theta is not None and grain_phi is not None:
                            thetas.append(grain_theta)
                            phis.append(grain_phi)
                    
                    if thetas and phis:
                        # Calculate circular mean to handle wrapping
                        sin_sum_theta = sum(np.sin(t) for t in thetas)
                        cos_sum_theta = sum(np.cos(t) for t in thetas)
                        center_theta = np.arctan2(sin_sum_theta, cos_sum_theta) % (2*np.pi)
                        
                        sin_sum_phi = sum(np.sin(p) for p in phis)
                        cos_sum_phi = sum(np.cos(p) for p in phis)
                        center_phi = np.arctan2(sin_sum_phi, cos_sum_phi) % (2*np.pi)
                
                if center_theta is None or center_phi is None:
                    continue
                
                # Get void strength and radius
                strength = region.get('strength', 0.5)
                radius = region.get('radius', 0.5)
                
                # Scale radius based on grid resolution
                radius_scaled = radius * 2 * np.pi / 10
                
                # Apply void pattern to void field
                for i in range(self.viz_resolution):
                    for j in range(self.viz_resolution):
                        grid_theta = self._theta_grid[i, j]
                        grid_phi = self._phi_grid[i, j]
                        
                        # Calculate toroidal distance (accounting for wrapping)
                        d_theta = min(abs(grid_theta - center_theta), 2*np.pi - abs(grid_theta - center_theta))
                        d_phi = min(abs(grid_phi - center_phi), 2*np.pi - abs(grid_phi - center_phi))
                        dist = np.sqrt(d_theta**2 + d_phi**2)
                        
                        # Skip points outside void radius
                        if dist > radius_scaled:
                            continue
                        
                        # Calculate influence based on distance
                        influence = (1 - dist/radius_scaled) * strength
                        
                        # Apply to void field
                        self._void_field[i, j] = max(self._void_field[i, j], influence)
    
    def _generate_vector_fields(self, manifold):
        """
        Generate vector fields for visualization.
        
        Args:
            manifold: RelationalManifold for position reference
        """
        # Try to extract polarity data from manifold
        if hasattr(manifold, 'polarity_field') and hasattr(manifold.polarity_field, 'calculate_vector_field'):
            # Use polarity field's vector calculation if available
            theta_vectors, phi_vectors = manifold.polarity_field.calculate_vector_field(
                self._theta_grid, self._phi_grid
            )
            
            # Apply to vector fields
            self._vector_field_theta = theta_vectors
            self._vector_field_phi = phi_vectors
        else:
            # Generate vector field from metrics
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
                    
                    # Combine base direction with variation
                    self._vector_field_theta[i, j] = 0.5 * (theta_bias + theta_var)
                    self._vector_field_phi[i, j] = 0.5 * (phi_bias + phi_var)
        
        # Update aliases
        self._polarity_field_u = self._vector_field_theta
        self._polarity_field_v = self._vector_field_phi
    
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
            self._simple_smooth(self._vector_field_theta)
            self._simple_smooth(self._vector_field_phi)
        
        # Normalize fields to 0-1 range
        self._normalize_field(self._awareness_field)
        self._normalize_field(self._tension_field)
        self._normalize_field(self._backflow_field)
        self._normalize_field(self._stability_field)
        self._normalize_field(self._void_field)
        
        # Ensure phase field is modulo 2π
        self._phase_field = self._phase_field % (2 * np.pi)
        
        # Ensure periodicity at boundaries
        self._ensure_field_periodicity(self._awareness_field)
        self._ensure_field_periodicity(self._tension_field)
        self._ensure_field_periodicity(self._backflow_field)
        self._ensure_field_periodicity(self._stability_field)
        self._ensure_field_periodicity(self._void_field)
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