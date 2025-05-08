"""
Emergent Duality Engine - Enhanced Observer for Collapse Geometry

This engine implements a purely observational approach to collapse dynamics with
improved communication pathways between the engine and manifold.
It observes what naturally emerges from relational dynamics without imposing changes.
"""

import random
import time as system_time
import math
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Callable, TYPE_CHECKING
from collections import defaultdict, deque

# Import specific functions from collapse rules to avoid circular imports
from axiom8.collapse_rules.config_space import angular_difference
from axiom8.collapse_rules.polarity_space import circular_mean

# Forward references for TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from axiom8.core.relational_manifold import RelationalManifold
    from axiom8.core.state import SimulationState

# Forward references for circular imports
if TYPE_CHECKING:
    from axiom8.core.relational_manifold import RelationalManifold


class EmergentDualityEngine:
    """
    Pure observer engine for the Collapse Geometry framework.
    
    This engine doesn't drive the dynamics but simply observes what naturally
    emerges from the relational manifold. It acknowledges that all structure,
    including time, emerges from the relational dynamics themselves.
    
    Enhanced with proper delegation to manifold methods for accessing data,
    improving observation of toroidal structures and ancestry patterns.
    """
    
    def __init__(self, manifold: 'RelationalManifold', state=None, config=None):
        """
        Initialize the observer engine with a manifold and optional configuration.
        
        Args:
            manifold: RelationalManifold to observe
            state: Optional state object (for backward compatibility)
            config: Optional configuration parameters
        """
        self.manifold = manifold
        self.state = state
        
        # Default configuration - minimal for pure observation
        self.config = {
            'observation_detail': 0.8,   # How detailed observations should be
            'observation_rate': 1.0,     # How frequently to observe
            'memory_depth': 50,          # How many observations to keep in memory
            'trace_causal_relations': True,  # Whether to trace causal relations
            'record_metrics': True,      # Whether to record metrics
            'observe_ancestry': True,    # Whether to track ancestry patterns
            'observe_backflow': True,    # Whether to track backflow dynamics
            'track_absence_patterns': True,  # Whether to track absence patterns
            'safe_interface_access': True  # Whether to use safe interface access
        }
        
        # Update with custom config if provided
        if config:
            self.config.update(config)
        
        # Observation memory
        self.collapse_observations = deque(maxlen=self.config['memory_depth'])
        self.structure_observations = deque(maxlen=self.config['memory_depth'])
        self.field_observations = deque(maxlen=self.config['memory_depth'])
        self.backflow_observations = deque(maxlen=self.config['memory_depth'])
        self.absence_observations = deque(maxlen=self.config['memory_depth'])
        self.ancestry_observations = deque(maxlen=self.config['memory_depth'])
        
        # Metrics and statistics
        self.metrics = {
            'observation_count': 0,
            'last_observation_time': system_time.time(),
            'observed_emergent_time': 0.0,
            'observed_collapses': 0,
            'observed_voids': 0,
            'observed_coherence': 0.0,
            'observed_field_tension': 0.0,
            'observed_emergence_events': 0,
            'observed_vortices': 0,
            'observed_pathways': {'structure': 0, 'decay': 0},
            
            # Enhanced metrics for bidirectional dynamics
            'lightlike_ratio': 0.0,      # Ratio of lightlike to total grains
            'backflow_intensity': 0.0,   # Strength of recursive collapse patterns
            'curvature_metric': 0.0,     # Overall system curvature
            'ancestry_depth': 0.0,       # Average ancestry tree depth
            'absence_pattern_count': 0,  # Number of detected absence patterns
            'recursive_collapse_count': 0,  # Count of recursive collapses
            'phase_reversal_count': 0,   # Count of phase reversals from backflow
            'toroidal_structure_metrics': {},  # Detailed toroidal structure metrics
            'self_referential_count': 0,  # Count of self-referential grains
            'field_genesis_count': 0      # Count of field-genesis events
        }
        
        # Trace observed causal relations without imposing them
        if self.config['trace_causal_relations']:
            self.causal_observations = defaultdict(list)
            self.recursive_causal_paths = []  # Track recursive causal loops
        
        # Track backflow patterns
        if self.config['observe_backflow']:
            self.backflow_regions = []  # Regions with high backflow activity
            self.curvature_centers = []  # Centers of high curvature
            self.recursive_pathways = []  # Pathways with recursive collapse
        
        # Track ancestry patterns
        if self.config['observe_ancestry']:
            self.ancestry_trees = {}  # Maps grain_id -> ancestry tree
            self.inheritance_patterns = []  # Detected inheritance patterns
            self.self_referential_grains = set()  # Grains that reference themselves
            self.field_genesis_events = []  # Field-level genesis events
        
        # Track absence patterns
        if self.config['track_absence_patterns']:
            self.void_formations = []  # Void formation events
            self.decay_cascades = []  # Decay cascade patterns
            self.tensile_boundaries = []  # Boundaries between structure and void
        
        # Register with manifold if it supports it
        if hasattr(manifold, 'integrated_engine'):
            manifold.integrated_engine = self
            
        # Add necessary accessor methods to manifold if they don't exist
        self._enhance_manifold_interface()
        
        # Verify interface compatibility if safe access is enabled
        if self.config['safe_interface_access']:
            self._verify_interface_compatibility()
    
    def _safe_get(self, data_dict: Dict[str, Any], primary_key: str, 
                 fallback_keys: Optional[List[str]] = None, 
                 default_value: Any = None) -> Any:
        """
        Safely get a value from a dictionary with fallback keys and default value.
        Helps make the engine more resilient to interface changes.
        
        Args:
            data_dict: Dictionary to get value from
            primary_key: Primary key to try
            fallback_keys: List of fallback keys to try if primary_key is not found
            default_value: Default value to return if no keys are found
            
        Returns:
            Value from dictionary or default value
        """
        # Try primary key first
        if data_dict and primary_key in data_dict:
            return data_dict[primary_key]
        
        # Try fallback keys if provided
        if fallback_keys:
            for key in fallback_keys:
                if data_dict and key in data_dict:
                    return data_dict[key]
        
        # Return default value if not found
        return default_value
    
    def _verify_interface_compatibility(self):
        """
        Verify that the manifold interface is compatible with the engine.
        Logs warnings for potential issues but doesn't block execution.
        """
        issues = []
        
        # Check manifold methods expected by engine
        expected_methods = [
            'get_ancestry_distribution',
            'get_relation_memory_metrics',
            'get_all_emergent_structures',
            'get_grain_ancestry',
            'get_recursive_collapse_metrics',
            'get_field_coherence_metrics'
        ]
        
        for method in expected_methods:
            if not hasattr(self.manifold, method):
                issues.append(f"Missing expected manifold method: {method}")
        
        # Check data structure compatibility (when methods exist)
        if hasattr(self.manifold, 'get_ancestry_distribution'):
            try:
                ancestry_data = self.manifold.get_ancestry_distribution()
                expected_keys = [
                    'distribution', 'recursive_indices', 'curvature_metrics',
                    'average_ancestry_size', 'average_recursive_index', 
                    'average_curvature', 'self_referential_count'
                ]
                
                for key in expected_keys:
                    if key not in ancestry_data:
                        issues.append(f"Missing expected key in ancestry data: {key}")
                        
                # Check for key name mismatch that originally caused the error
                if 'average_depth' in ancestry_data and ancestry_data['average_depth'] != ancestry_data.get('average_ancestry_size'):
                    issues.append(f"Key mismatch: 'average_depth' doesn't match 'average_ancestry_size'")
            except Exception as e:
                issues.append(f"Error accessing ancestry distribution: {e}")
        
        # Log issues if any
        if issues:
            print("WARNING: Interface compatibility issues detected:")
            for issue in issues:
                print(f"  - {issue}")
            print("Engine will attempt to handle these issues gracefully.")
    
    def _enhance_manifold_interface(self):
        """
        Add necessary accessor methods to the manifold for better communication.
        This ensures the engine can access data through proper abstractions.
        """
        # Add method to get ancestry distribution if it doesn't exist
        if not hasattr(self.manifold, 'get_ancestry_distribution'):
            def get_ancestry_distribution():
                ancestry_counts = defaultdict(int)
                self_referential_count = 0
                
                for grain_id, grain in self.manifold.grains.items():
                    ancestry = getattr(grain, 'ancestry', set())
                    ancestry_size = len(ancestry)
                    ancestry_counts[ancestry_size] += 1
                    
                    # Track self-referential grains
                    if grain_id in ancestry:
                        self_referential_count += 1
                
                # Calculate average ancestry size
                total_ancestry = sum(size * count for size, count in ancestry_counts.items())
                total_grains = sum(ancestry_counts.values())
                average_ancestry_size = total_ancestry / total_grains if total_grains else 0
                
                return {
                    'distribution': dict(ancestry_counts),
                    'self_referential_count': self_referential_count,
                    'average_depth': average_ancestry_size,  # For backward compatibility
                    'average_ancestry_size': average_ancestry_size  # New consistent name
                }
            
            # Bind method to manifold
            self.manifold.get_ancestry_distribution = get_ancestry_distribution.__get__(self.manifold)
        
        # Add method to get relation memory metrics if it doesn't exist
        if not hasattr(self.manifold, 'get_relation_memory_metrics'):
            def get_relation_memory_metrics():
                if not hasattr(self.manifold, 'relation_memory'):
                    return {}
                    
                memory_counts = {
                    'collapse': 0,
                    'tension': 0,
                    'collapse_echo': 0,
                    'total_relations': len(self.manifold.relation_memory)
                }
                
                for relation_key, interactions in self.manifold.relation_memory.items():
                    for interaction in interactions:
                        memory_type = interaction.get('type', 'unknown')
                        memory_counts[memory_type] = memory_counts.get(memory_type, 0) + 1
                
                return memory_counts
            
            # Bind method to manifold
            self.manifold.get_relation_memory_metrics = get_relation_memory_metrics.__get__(self.manifold)
        
        # Add method to get all emergent toroidal structures if it doesn't exist
        if not hasattr(self.manifold, 'get_all_emergent_structures'):
            def get_all_emergent_structures():
                """
                Returns all detected emergent structures in the toroidal manifold.
                This provides a single access point that can be expanded as new
                structure types emerge without requiring engine changes.
                """
                if not hasattr(self.manifold, 'toroidal_coordinator'):
                    return {}
                
                coordinator = self.manifold.toroidal_coordinator
                
                return {
                    'vortices': getattr(coordinator, 'vortices', []),
                    'lightlike_pathways': getattr(coordinator, 'lightlike_pathways', {'structure': [], 'decay': []}),
                    'phase_boundaries': getattr(coordinator, 'phase_boundaries', []),
                    'recursive_circuits': getattr(coordinator, 'recursive_circuits', []),
                    'field_singularities': getattr(coordinator, 'field_singularities', []),
                    'emergent_loops': getattr(coordinator, 'emergent_loops', [])
                }
            
            # Bind method to manifold
            self.manifold.get_all_emergent_structures = get_all_emergent_structures.__get__(self.manifold)
        
        # Add method to get grain ancestry if it doesn't exist
        if not hasattr(self.manifold, 'get_grain_ancestry'):
            def get_grain_ancestry(grain_id):
                grain = self.manifold.get_grain(grain_id)
                if not grain:
                    return set()
                return getattr(grain, 'ancestry', set())
            
            # Bind method to manifold
            self.manifold.get_grain_ancestry = get_grain_ancestry.__get__(self.manifold)
        
        # Add method to get recursive collapse metrics if it doesn't exist
        if not hasattr(self.manifold, 'get_recursive_collapse_metrics'):
            def get_recursive_collapse_metrics():
                if not hasattr(self.manifold, 'collapse_history'):
                    return {}
                
                # Count recursive collapses (where target has source in ancestry)
                recursive_count = 0
                self_collapse_count = 0
                field_genesis_count = 0
                
                for event in self.manifold.collapse_history:
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
                    target_grain = self.manifold.get_grain(target_id)
                    if target_grain and hasattr(target_grain, 'ancestry'):
                        if source_id in target_grain.ancestry:
                            recursive_count += 1
                
                return {
                    'recursive_collapse_count': recursive_count,
                    'self_collapse_count': self_collapse_count,
                    'field_genesis_count': field_genesis_count,
                    'total_collapses': len(self.manifold.collapse_history)
                }
            
            # Bind method to manifold
            self.manifold.get_recursive_collapse_metrics = get_recursive_collapse_metrics.__get__(self.manifold)
        
        # Add method to calculate field coherence metrics if it doesn't exist
        if not hasattr(self.manifold, 'get_field_coherence_metrics'):
            def get_field_coherence_metrics():
                """Get detailed coherence metrics across the field"""
                metrics = {
                    'global_coherence': getattr(self.manifold, 'field_coherence', 0.5),
                    'system_tension': getattr(self.manifold, 'system_tension', 0.0)
                }
                
                if hasattr(self.manifold, 'toroidal_coordinator'):
                    coordinator = self.manifold.toroidal_coordinator
                    
                    # Get toroidal coherence if available
                    if hasattr(coordinator, 'calculate_global_coherence'):
                        metrics['toroidal_coherence'] = coordinator.calculate_global_coherence()
                    
                    # Get coherence field data if available
                    if hasattr(coordinator, 'coherence_field'):
                        coherence_values = list(coordinator.coherence_field.values())
                        if coherence_values:
                            metrics['mean_coherence'] = sum(coherence_values) / len(coherence_values)
                            
                            # Calculate variance
                            mean = metrics['mean_coherence']
                            variance = sum((c - mean) ** 2 for c in coherence_values) / len(coherence_values)
                            metrics['coherence_variance'] = variance
                
                return metrics
            
            # Bind method to manifold
            self.manifold.get_field_coherence_metrics = get_field_coherence_metrics.__get__(self.manifold)
    
    def observe(self, duration=1.0) -> Dict[str, Any]:
        """
        Observe the manifold for a specified duration.
        This doesn't drive dynamics but simply watches what happens naturally.
        
        Args:
            duration: Real-time seconds to observe
            
        Returns:
            Dictionary with observation results
        """
        start_time = system_time.time()
        observations = []
        
        try:
            # Record initial state
            initial_state = self._observe_current_state()
            
            # Let manifold dynamics naturally manifest
            manifest_result = self.manifold.manifest_dynamics(time_step=1.0)
            
            # Observe what naturally manifested
            manifestation_observation = self._observe_manifestation(manifest_result)
            observations.append(manifestation_observation)
            
            # Record final state
            final_state = self._observe_current_state()
            
            # Calculate differential observations
            differential = self._observe_differential(initial_state, final_state)
            
            # Observe backflow and absence patterns
            backflow_observation = self._observe_backflow_dynamics()
            absence_observation = self._observe_absence_patterns()
            ancestry_observation = self._observe_ancestry_patterns()
            
            # Update metrics
            self._update_metrics(
                manifestation_observation, 
                differential, 
                backflow_observation,
                absence_observation,
                ancestry_observation
            )
        except Exception as e:
            # Graceful error handling
            print(f"Error during observation: {e}")
            # Create minimal observation data to return
            if not 'initial_state' in locals():
                initial_state = {'error': f"Failed to observe initial state: {e}"}
            if not 'final_state' in locals():
                final_state = {'error': f"Failed to observe final state: {e}"}
            if not 'differential' in locals():
                differential = {'error': f"Failed to calculate differential: {e}"}
            if not 'backflow_observation' in locals():
                backflow_observation = {'error': f"Failed to observe backflow dynamics: {e}"}
            if not 'absence_observation' in locals():
                absence_observation = {'error': f"Failed to observe absence patterns: {e}"}
            if not 'ancestry_observation' in locals():
                ancestry_observation = {'error': f"Failed to observe ancestry patterns: {e}"}
        
        # Increment observation count
        self.metrics['observation_count'] += 1
        self.metrics['last_observation_time'] = system_time.time()
        self.metrics['observed_emergent_time'] = getattr(self.manifold, 'time', 0.0)
        
        # Return combined observations
        return {
            'initial_state': initial_state,
            'manifestations': observations,
            'final_state': final_state,
            'differential': differential,
            'backflow': backflow_observation,
            'absence_patterns': absence_observation,
            'ancestry_patterns': ancestry_observation,
            'elapsed_time': system_time.time() - start_time,
            'metrics': self.metrics
        }
    
    def _observe_current_state(self) -> Dict[str, Any]:
        """
        Observe the current state of the manifold without changing it.
        Uses delegation to read from manifold through proper abstractions.
        
        Returns:
            Dictionary with current state observation
        """
        # Basic metrics
        grain_count = len(self.manifold.grains)
        
        # Observe field structure
        field_structure = {
            'grain_count': grain_count,
            'emergent_time': self.manifold.time,
            'field_coherence': self._safe_get(self.manifold.__dict__, 'field_coherence', default_value=0.0),
            'system_tension': self._safe_get(self.manifold.__dict__, 'system_tension', default_value=0.0),
        }
        
        # Observe grain distribution if there are grains
        if grain_count > 0:
            # Calculate lightlike ratio
            lightlike_count = sum(1 for grain in self.manifold.grains.values() 
                              if grain.grain_saturation < 0.2)
            field_structure['lightlike_ratio'] = lightlike_count / grain_count
            
            # Calculate high-saturation (potential backflow) ratio
            high_saturation_count = sum(1 for grain in self.manifold.grains.values()
                                     if grain.grain_saturation > 0.7)
            field_structure['high_saturation_ratio'] = high_saturation_count / grain_count
            
            # Get ancestry distribution using proper delegation
            if hasattr(self.manifold, 'get_ancestry_distribution'):
                try:
                    ancestry_data = self.manifold.get_ancestry_distribution()
                    
                    # Use safe getter for all fields
                    field_structure['ancestry_distribution'] = self._safe_get(
                        ancestry_data, 'distribution', default_value={})
                    
                    field_structure['self_referential_count'] = self._safe_get(
                        ancestry_data, 'self_referential_count', 
                        ['self_reference_count'], 0)
                    
                    # Use multiple possible keys for ancestry depth
                    field_structure['avg_ancestry_depth'] = self._safe_get(
                        ancestry_data, 'average_depth', 
                        ['average_ancestry_size', 'avg_ancestry_depth'], 0.0)
                except Exception as e:
                    print(f"Error getting ancestry distribution: {e}")
                    # Provide fallback values
                    field_structure['ancestry_distribution'] = {}
                    field_structure['self_referential_count'] = 0
                    field_structure['avg_ancestry_depth'] = 0.0
            else:
                # Fallback to direct property access if method doesn't exist
                ancestry_counts = defaultdict(int)
                self_referential_count = 0
                for grain_id, grain in self.manifold.grains.items():
                    ancestry = getattr(grain, 'ancestry', set())
                    ancestry_depth = len(ancestry)
                    ancestry_counts[ancestry_depth] += 1
                    
                    # Count self-referential grains
                    if grain_id in ancestry:
                        self_referential_count += 1
                
                field_structure['ancestry_distribution'] = dict(ancestry_counts)
                field_structure['self_referential_count'] = self_referential_count
                
                if ancestry_counts:
                    # Calculate average ancestry depth
                    total_depth = sum(depth * count for depth, count in ancestry_counts.items())
                    field_structure['avg_ancestry_depth'] = total_depth / grain_count
            
            # Observe awareness distribution
            awareness_values = [grain.awareness for grain in self.manifold.grains.values()]
            field_structure['mean_awareness'] = sum(awareness_values) / grain_count
            field_structure['awareness_variance'] = self._calculate_variance(awareness_values)
            
            # Observe activation patterns
            activation_values = [grain.grain_activation for grain in self.manifold.grains.values()]
            field_structure['mean_activation'] = sum(activation_values) / grain_count
            
            # Observe saturation patterns
            saturation_values = [grain.grain_saturation for grain in self.manifold.grains.values()]
            field_structure['mean_saturation'] = sum(saturation_values) / grain_count
            field_structure['saturation_variance'] = self._calculate_variance(saturation_values)
            
            # Observe polarity patterns
            # Read polarity using proper accessor method if available
            if hasattr(self.manifold, 'get_polarity_values'):
                try:
                    polarity_values = self.manifold.get_polarity_values()
                except Exception as e:
                    print(f"Error getting polarity values: {e}")
                    # Fallback to direct property access
                    polarity_values = [getattr(grain, 'polarity', 0.0) for grain in self.manifold.grains.values()]
            else:
                # Fallback to direct property access
                polarity_values = [getattr(grain, 'polarity', 0.0) for grain in self.manifold.grains.values()]
            
            field_structure['mean_polarity'] = sum(polarity_values) / grain_count
            field_structure['polarity_variance'] = self._calculate_variance(polarity_values)
            
            # Calculate structure vs decay bias
            structure_count = sum(1 for p in polarity_values if p > 0.2)
            decay_count = sum(1 for p in polarity_values if p < -0.2)
            neutral_count = grain_count - structure_count - decay_count
            
            field_structure['structure_ratio'] = structure_count / grain_count
            field_structure['decay_ratio'] = decay_count / grain_count
            field_structure['neutral_ratio'] = neutral_count / grain_count
            
            # Calculate collapse metric statistics
            collapse_values = [grain.collapse_metric for grain in self.manifold.grains.values()]
            field_structure['mean_collapse_metric'] = sum(collapse_values) / grain_count
            field_structure['max_collapse_metric'] = max(collapse_values, default=0.0)
            
            # Get field genesis count using proper delegation
            if hasattr(self.manifold, 'get_recursive_collapse_metrics'):
                try:
                    metrics = self.manifold.get_recursive_collapse_metrics()
                    field_structure['field_genesis_count'] = self._safe_get(
                        metrics, 'field_genesis_count', default_value=0)
                except Exception as e:
                    print(f"Error getting recursive collapse metrics: {e}")
                    field_structure['field_genesis_count'] = 0
            elif hasattr(self.manifold, 'collapse_history'):
                # Fallback to direct property access
                field_genesis_count = sum(1 for event in self.manifold.collapse_history 
                                       if event.get('field_genesis', False))
                field_structure['field_genesis_count'] = field_genesis_count
        
        # Observe toroidal structure using delegation
        toroidal_data = self._observe_toroidal_structure()
        if toroidal_data:
            field_structure.update(toroidal_data)
        
        # Observe curvature metrics using delegation if available
        if hasattr(self.manifold, 'calculate_curvature_metrics'):
            try:
                curvature_data = self.manifold.calculate_curvature_metrics()
            except Exception as e:
                print(f"Error calculating curvature metrics: {e}")
                curvature_data = None
        else:
            # Fallback to internal calculation
            curvature_data = self._observe_curvature_metrics()
            
        if curvature_data:
            field_structure.update(curvature_data)
        
        # Get coherence metrics using proper delegation
        if hasattr(self.manifold, 'get_field_coherence_metrics'):
            try:
                coherence_metrics = self.manifold.get_field_coherence_metrics()
                for key, value in coherence_metrics.items():
                    field_structure[key] = value
            except Exception as e:
                print(f"Error getting field coherence metrics: {e}")
        
        # Construct final observation
        observation = {
            'type': 'state_observation',
            'time': self.manifold.time,
            'observation_time': system_time.time(),
            'field_structure': field_structure,
        }
        
        # Add collapse history count
        observation['collapse_count'] = (
            len(self.manifold.collapse_history) if hasattr(self.manifold, 'collapse_history') else 0)
        
        # Add void formation count
        observation['void_count'] = (
            len(self.manifold.void_formation_events) if hasattr(self.manifold, 'void_formation_events') else 0)
        
        # Add emergent structure counts
        if hasattr(self.manifold, 'toroidal_coordinator'):
            coordinator = self.manifold.toroidal_coordinator
            
            # Get vortex count
            observation['vortex_count'] = (
                len(coordinator.vortices) if hasattr(coordinator, 'vortices') else 0)
            
            # Get pathway counts
            if hasattr(coordinator, 'lightlike_pathways'):
                structure_pathways = len(coordinator.lightlike_pathways.get('structure', []))
                decay_pathways = len(coordinator.lightlike_pathways.get('decay', []))
                observation['pathway_counts'] = {
                    'structure': structure_pathways,
                    'decay': decay_pathways
                }
            else:
                observation['pathway_counts'] = {'structure': 0, 'decay': 0}
        else:
            observation['vortex_count'] = 0
            observation['pathway_counts'] = {'structure': 0, 'decay': 0}
        
        # Add superposition count
        observation['superposition_count'] = self._safe_get(
            self.manifold.__dict__, 'superposition_count', default_value=0)
        
        # Observe field metrics
        observation['observed_fields'] = {
            'coherence_field': self._observe_coherence_field(),
            'tension_field': self._observe_tension_field()
        }
        
        return observation
    
    def _observe_toroidal_structure(self) -> Dict[str, Any]:
        """
        Observe the toroidal structure via the coordinator.
        Uses delegation to access all emergent structures in a unified way.
        
        Returns:
            Dictionary with toroidal structure observations
        """
        # Initialize empty result
        result = {}
        
        # Use proper delegation if available
        if hasattr(self.manifold, 'get_all_emergent_structures'):
            try:
                # Get all emergent structures using the unified method
                structures = self.manifold.get_all_emergent_structures()
                
                # Extract vortex data
                vortices = self._safe_get(structures, 'vortices', default_value=[])
                if vortices:
                    result['vortex_count'] = len(vortices)
                    
                    # Calculate vortex metrics
                    circulation_values = [abs(self._safe_get(v, 'circulation', default_value=0.0)) for v in vortices]
                    if circulation_values:
                        result['mean_circulation'] = sum(circulation_values) / len(circulation_values)
                        result['max_circulation'] = max(circulation_values)
                    
                    # Count lightlike vortices
                    lightlike_vortices = sum(1 for v in vortices if self._safe_get(v, 'is_lightlike', default_value=False))
                    result['lightlike_vortex_count'] = lightlike_vortices
                    result['lightlike_vortex_ratio'] = lightlike_vortices / len(vortices)
                    
                    # Analyze vortex polarities
                    vortex_polarities = [self._safe_get(v, 'polarity', default_value=0.0) for v in vortices if 'polarity' in v]
                    if vortex_polarities:
                        result['mean_vortex_polarity'] = sum(vortex_polarities) / len(vortex_polarities)
                        
                        structure_vortices = sum(1 for p in vortex_polarities if p > 0.2)
                        decay_vortices = sum(1 for p in vortex_polarities if p < -0.2)
                        result['structure_vortex_ratio'] = structure_vortices / len(vortex_polarities)
                        result['decay_vortex_ratio'] = decay_vortices / len(vortex_polarities)
                    
                    # Calculate vortex coherence
                    coherence = self._calculate_vortex_coherence(vortices)
                    result['vortex_coherence'] = coherence
                
                # Extract pathways data
                pathways = self._safe_get(structures, 'lightlike_pathways', default_value={'structure': [], 'decay': []})
                result['structure_pathways'] = len(self._safe_get(pathways, 'structure', default_value=[]))
                result['decay_pathways'] = len(self._safe_get(pathways, 'decay', default_value=[]))
                
                # Calculate pathway metrics
                structure_pathways = self._safe_get(pathways, 'structure', default_value=[])
                decay_pathways = self._safe_get(pathways, 'decay', default_value=[])
                
                if structure_pathways:
                    structure_lengths = [len(self._safe_get(p, 'nodes', default_value=[])) for p in structure_pathways]
                    result['mean_structure_pathway_length'] = sum(structure_lengths) / len(structure_lengths)
                    result['max_structure_pathway_length'] = max(structure_lengths)
                    
                    # Calculate lightlike ratio along structure pathways
                    lightlike_ratios = [self._safe_get(p, 'lightlike_ratio', default_value=0.0) for p in structure_pathways]
                    result['structure_pathway_lightlike_ratio'] = sum(lightlike_ratios) / len(lightlike_ratios)
                
                if decay_pathways:
                    decay_lengths = [len(self._safe_get(p, 'nodes', default_value=[])) for p in decay_pathways]
                    result['mean_decay_pathway_length'] = sum(decay_lengths) / len(decay_lengths)
                    result['max_decay_pathway_length'] = max(decay_lengths)
                    
                    # Calculate lightlike ratio along decay pathways
                    lightlike_ratios = [self._safe_get(p, 'lightlike_ratio', default_value=0.0) for p in decay_pathways]
                    result['decay_pathway_lightlike_ratio'] = sum(lightlike_ratios) / len(lightlike_ratios)
                
                # Check for new emergent structures
                for structure_type in ['phase_boundaries', 'recursive_circuits', 'field_singularities', 'emergent_loops']:
                    structure_list = self._safe_get(structures, structure_type, default_value=[])
                    if structure_list:
                        result[f'{structure_type}_count'] = len(structure_list)
            except Exception as e:
                print(f"Error accessing emergent structures: {e}")
        else:
            # Fallback to direct coordinator access if delegation not available
            if not hasattr(self.manifold, 'toroidal_coordinator'):
                return result
                
            coordinator = self.manifold.toroidal_coordinator
            
            # Observe vortices
            if hasattr(coordinator, 'vortices') and coordinator.vortices:
                vortices = coordinator.vortices
                result['vortex_count'] = len(vortices)
                
                if vortices:
                    # Calculate average circulation strength
                    circulation_values = [abs(self._safe_get(v, 'circulation', default_value=0.0)) for v in vortices]
                    result['mean_circulation'] = sum(circulation_values) / len(circulation_values)
                    result['max_circulation'] = max(circulation_values)
                    
                    # Count lightlike vortices (special significance in theory)
                    lightlike_vortices = sum(1 for v in vortices if self._safe_get(v, 'is_lightlike', default_value=False))
                    result['lightlike_vortex_count'] = lightlike_vortices
                    result['lightlike_vortex_ratio'] = lightlike_vortices / len(vortices)
                    
                    # Track polarity distribution in vortices
                    vortex_polarities = [self._safe_get(v, 'polarity', default_value=0.0) for v in vortices if 'polarity' in v]
                    if vortex_polarities:
                        result['mean_vortex_polarity'] = sum(vortex_polarities) / len(vortex_polarities)
                        
                        # Count structure and decay vortices
                        structure_vortices = sum(1 for p in vortex_polarities if p > 0.2)
                        decay_vortices = sum(1 for p in vortex_polarities if p < -0.2)
                        result['structure_vortex_ratio'] = structure_vortices / len(vortex_polarities)
                        result['decay_vortex_ratio'] = decay_vortices / len(vortex_polarities)
                    
                    # Calculate vortex coherence
                    coherence = self._calculate_vortex_coherence(vortices)
                    result['vortex_coherence'] = coherence
            
            # Observe lightlike pathways
            if hasattr(coordinator, 'lightlike_pathways'):
                pathways = coordinator.lightlike_pathways
                result['structure_pathways'] = len(self._safe_get(pathways, 'structure', default_value=[]))
                result['decay_pathways'] = len(self._safe_get(pathways, 'decay', default_value=[]))
                
                # Calculate pathway metrics
                structure_lengths = [len(self._safe_get(p, 'nodes', default_value=[])) 
                                     for p in self._safe_get(pathways, 'structure', default_value=[])]
                decay_lengths = [len(self._safe_get(p, 'nodes', default_value=[])) 
                                 for p in self._safe_get(pathways, 'decay', default_value=[])]
                
                if structure_lengths:
                    result['mean_structure_pathway_length'] = sum(structure_lengths) / len(structure_lengths)
                    result['max_structure_pathway_length'] = max(structure_lengths)
                    
                    # Calculate lightlike ratio along structure pathways
                    if self._safe_get(pathways, 'structure', default_value=[]):
                        lightlike_ratios = [self._safe_get(p, 'lightlike_ratio', default_value=0.0) 
                                            for p in pathways['structure']]
                        result['structure_pathway_lightlike_ratio'] = sum(lightlike_ratios) / len(lightlike_ratios)
                
                if decay_lengths:
                    result['mean_decay_pathway_length'] = sum(decay_lengths) / len(decay_lengths)
                    result['max_decay_pathway_length'] = max(decay_lengths)
                    
                    # Calculate lightlike ratio along decay pathways
                    if self._safe_get(pathways, 'decay', default_value=[]):
                        lightlike_ratios = [self._safe_get(p, 'lightlike_ratio', default_value=0.0) 
                                            for p in pathways['decay']]
                        result['decay_pathway_lightlike_ratio'] = sum(lightlike_ratios) / len(lightlike_ratios)
        
        # Calculate overall coherence using coordinator's method
        if hasattr(self.manifold, 'toroidal_coordinator'):
            coordinator = self.manifold.toroidal_coordinator
            if hasattr(coordinator, 'calculate_global_coherence'):
                try:
                    result['global_coherence'] = coordinator.calculate_global_coherence()
                except Exception as e:
                    print(f"Error calculating global coherence: {e}")
                    result['global_coherence'] = 0.5
        
        # Store in metrics for consistent tracking
        self.metrics['toroidal_structure_metrics'] = result
        
        return result
    
    def _observe_curvature_metrics(self) -> Dict[str, Any]:
        """
        Observe curvature-related metrics that indicate recursive collapse patterns.
        These are indicators of backflow dynamics in the system.
        
        Returns:
            Dictionary with curvature metrics
        """
        result = {}
        grain_count = len(self.manifold.grains)
        
        if grain_count == 0:
            return result
        
        # Calculate ancestry-based curvature using proper delegation if available
        if hasattr(self.manifold, 'get_grain_ancestry'):
            # Use proper accessor methods
            ancestry_depths = []
            recursive_indices = []
            
            for grain_id in self.manifold.grains:
                try:
                    # Get ancestry using accessor method
                    ancestry = self.manifold.get_grain_ancestry(grain_id)
                    ancestry_depths.append(len(ancestry))
                    
                    # Calculate recursive index (how much a grain references its own ancestry)
                    recursive_index = 0.0
                    
                    # Check for self-reference (self-genesis)
                    if grain_id in ancestry:
                        recursive_index += 0.5  # Strong recursive signal when grain references itself
                    
                    # Check for ancestry sharing (transitive recursion)
                    ancestor_pairs = [(a1, a2) for a1 in ancestry for a2 in ancestry if a1 != a2]
                    for a1, a2 in ancestor_pairs:
                        # Check if ancestors share relations
                        relation_key = (a1, a2)
                        if hasattr(self.manifold, 'relation_memory') and relation_key in self.manifold.relation_memory:
                            recursive_index += 0.05  # Small increment for each ancestor relationship
                    
                    # Get grain for additional metrics
                    grain = self.manifold.get_grain(grain_id)
                    
                    # Check for direct parent relationship
                    if hasattr(grain, 'parent_id') and grain.parent_id:
                        # Direct parent relationship increases recursion
                        recursive_index += 0.3
                        
                        # Check for recursive ancestry patterns
                        if hasattr(grain, 'ancestry_polarity') and grain.parent_id in grain.ancestry_polarity:
                            # Parent's polarity influence on grain indicates recursion
                            recursive_index += 0.5
                    
                    # High saturation with continued collapse activity indicates recursion
                    if grain.grain_saturation > 0.7 and grain.grain_activation > 0.5:
                        recursive_index += 0.4
                    
                    recursive_indices.append(recursive_index)
                except Exception as e:
                    print(f"Error calculating curvature metrics for grain {grain_id}: {e}")
                    # Add default values to prevent calculation errors
                    ancestry_depths.append(0)
                    recursive_indices.append(0.0)
        else:
            # Fallback to direct property access
            ancestry_depths = []
            recursive_indices = []
            
            for grain_id, grain in self.manifold.grains.items():
                # Get ancestry depth
                ancestry = getattr(grain, 'ancestry', set())
                ancestry_depths.append(len(ancestry))
                
                # Calculate recursive index
                recursive_index = 0.0
                
                # Check for self-reference (self-genesis)
                if grain_id in ancestry:
                    recursive_index += 0.5
                
                # Check for ancestry sharing (transitive recursion)
                ancestor_pairs = [(a1, a2) for a1 in ancestry for a2 in ancestry if a1 != a2]
                for a1, a2 in ancestor_pairs:
                    # Check if ancestors share relations
                    relation_key = (a1, a2)
                    if hasattr(self.manifold, 'relation_memory') and relation_key in self.manifold.relation_memory:
                        recursive_index += 0.05
                
                # Check for direct parent relationship
                if hasattr(grain, 'parent_id') and grain.parent_id:
                    recursive_index += 0.3
                    
                    if hasattr(grain, 'ancestry_polarity') and grain.parent_id in grain.ancestry_polarity:
                        recursive_index += 0.5
                
                # High saturation with continued collapse activity indicates recursion
                if grain.grain_saturation > 0.7 and grain.grain_activation > 0.5:
                    recursive_index += 0.4
                
                recursive_indices.append(recursive_index)
        
        # Calculate backflow metrics
        if ancestry_depths:
            result['ancestry_depth_mean'] = sum(ancestry_depths) / len(ancestry_depths)
            result['ancestry_depth_max'] = max(ancestry_depths, default=0)
        else:
            result['ancestry_depth_mean'] = 0
            result['ancestry_depth_max'] = 0
            
        if recursive_indices:
            result['recursive_index_mean'] = sum(recursive_indices) / len(recursive_indices)
        else:
            result['recursive_index_mean'] = 0
        
        # Calculate system-wide curvature metric
        # Higher values indicate more overall recursive dynamics
        ancestry_contribution = min(1.0, result.get('ancestry_depth_mean', 0) / 10.0) * 0.4
        recursive_contribution = result.get('recursive_index_mean', 0) * 0.6
        
        result['system_curvature'] = ancestry_contribution + recursive_contribution
        
        # Track high-curvature grain count
        if recursive_indices:
            high_curvature_count = sum(1 for r in recursive_indices if r > 0.7)
            result['high_curvature_ratio'] = high_curvature_count / len(recursive_indices)
        else:
            result['high_curvature_ratio'] = 0
        
        return result
    
    def _observe_coherence_field(self) -> Dict[str, float]:
        """
        Observe the coherence field across the manifold.
        This tracks quantum coherence in different regions.
        
        Returns:
            Dictionary with coherence metrics
        """
        # Use delegation if available
        if hasattr(self.manifold, 'get_field_coherence_metrics'):
            try:
                coherence_metrics = self.manifold.get_field_coherence_metrics()
                
                # Extract coherence field subset
                coherence_data = {
                    'mean_coherence': self._safe_get(coherence_metrics, 'mean_coherence', default_value=0.5),
                    'coherence_variance': self._safe_get(coherence_metrics, 'coherence_variance', default_value=0.0),
                    'high_coherence_ratio': self._safe_get(coherence_metrics, 'high_coherence_ratio', default_value=0.0),
                    'low_coherence_ratio': self._safe_get(coherence_metrics, 'low_coherence_ratio', default_value=0.0)
                }
                return coherence_data
            except Exception as e:
                print(f"Error getting field coherence metrics: {e}")
        
        # Fallback to direct observation
        coherence_data = {}
        
        # Check if the manifold has toroidal coordinator
        if not hasattr(self.manifold, 'toroidal_coordinator'):
            return coherence_data
            
        coordinator = self.manifold.toroidal_coordinator
        
        # Access coherence field if available
        if hasattr(coordinator, 'coherence_field'):
            try:
                # Calculate average coherence
                coherence_values = list(coordinator.coherence_field.values())
                if coherence_values:
                    coherence_data['mean_coherence'] = sum(coherence_values) / len(coherence_values)
                    coherence_data['coherence_variance'] = self._calculate_variance(coherence_values)
                    
                    # Count high and low coherence ratios
                    high_coherence = sum(1 for c in coherence_values if c > 0.8)
                    low_coherence = sum(1 for c in coherence_values if c < 0.4)
                    
                    coherence_data['high_coherence_ratio'] = high_coherence / len(coherence_values)
                    coherence_data['low_coherence_ratio'] = low_coherence / len(coherence_values)
            except Exception as e:
                print(f"Error calculating coherence field metrics: {e}")
        
        return coherence_data
    
    def _observe_tension_field(self) -> Dict[str, float]:
        """
        Observe the tension field across the manifold.
        This tracks where structural tension accumulates.
        
        Returns:
            Dictionary with tension metrics
        """
        # Use delegation if available
        if hasattr(self.manifold, 'get_tension_field_metrics'):
            try:
                return self.manifold.get_tension_field_metrics()
            except Exception as e:
                print(f"Error getting tension field metrics: {e}")
        
        # Fallback to direct observation
        tension_data = {}
        
        # Check if the manifold has toroidal coordinator
        if not hasattr(self.manifold, 'toroidal_coordinator'):
            return tension_data
            
        coordinator = self.manifold.toroidal_coordinator
        
        # Access tension field if available
        if hasattr(coordinator, 'tension_field'):
            try:
                # Calculate average tension
                tension_values = list(coordinator.tension_field.values())
                if tension_values:
                    tension_data['mean_tension'] = sum(tension_values) / len(tension_values)
                    tension_data['tension_variance'] = self._calculate_variance(tension_values)
                    tension_data['max_tension'] = max(tension_values)
                    
                    # Count high tension points (potential void formation areas)
                    high_tension = sum(1 for t in tension_values if t > 0.8)
                    tension_data['high_tension_ratio'] = high_tension / len(tension_values)
            except Exception as e:
                print(f"Error calculating tension field metrics: {e}")
        
        return tension_data
    
    def _calculate_variance(self, values: List[float]) -> float:
        """
        Calculate variance of a list of values.
        
        Args:
            values: List of values to calculate variance for
            
        Returns:
            Variance value
        """
        if not values:
            return 0.0
            
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)
    
    def _calculate_circular_coherence(self, angles: List[float]) -> float:
        """
        Calculate coherence of circular/angular values.
        
        Args:
            angles: List of angular values
            
        Returns:
            Coherence value from 0.0 to 1.0
        """
        if not angles or len(angles) < 2:
            return 1.0
            
        # Convert to radians if needed (assume angles are in radians)
        x_sum = sum(math.cos(angle) for angle in angles)
        y_sum = sum(math.sin(angle) for angle in angles)
        
        # Calculate mean resultant length
        r = math.sqrt(x_sum**2 + y_sum**2) / len(angles)
        
        # Convert to coherence (0 = completely incoherent, 1 = perfectly coherent)
        return r
    
    def _calculate_vortex_coherence(self, vortices: List[Dict[str, Any]]) -> float:
        """
        Calculate the coherence of vortices - how aligned they are.
        
        Args:
            vortices: List of vortex dictionaries
            
        Returns:
            Coherence value from 0.0 to 1.0
        """
        if not vortices or len(vortices) < 2:
            return 1.0  # Single vortex is perfectly coherent with itself
            
        # Calculate circulation direction coherence
        clockwise = sum(1 for v in vortices if self._safe_get(v, 'direction') == 'clockwise')
        counterclockwise = len(vortices) - clockwise
        
        # Normalized to [0,1] where 1 = all vortices rotate in same direction
        direction_coherence = max(clockwise, counterclockwise) / len(vortices)
        
        # Calculate spatial coherence based on theta/phi positions
        # Extract theta/phi positions
        thetas = [self._safe_get(v, 'theta', default_value=0.0) for v in vortices if 'theta' in v]
        phis = [self._safe_get(v, 'phi', default_value=0.0) for v in vortices if 'phi' in v]
        
        # Calculate circular variance for theta and phi
        theta_coherence = self._calculate_circular_coherence(thetas) if thetas else 0.5
        phi_coherence = self._calculate_circular_coherence(phis) if phis else 0.5
        
        # Calculate polarity alignment if available
        polarities = [self._safe_get(v, 'polarity', default_value=0.0) for v in vortices if 'polarity' in v]
        polarity_coherence = 0.5  # Default neutral value
        
        if polarities and len(polarities) >= 2:
            # Calculate variance in polarities
            polarity_mean = sum(polarities) / len(polarities)
            polarity_variance = sum((p - polarity_mean) ** 2 for p in polarities) / len(polarities)
            
            # Convert to coherence (1 - normalized variance)
            polarity_coherence = 1.0 - min(1.0, polarity_variance)
        
        # Combined coherence with weights
        combined_coherence = (
            direction_coherence * 0.4 +
            theta_coherence * 0.2 +
            phi_coherence * 0.2 +
            polarity_coherence * 0.2
        )
        
        return combined_coherence
    
    def _count_repetitions(self, items):
        """Count repetitions in a list and return a dictionary of counts"""
        counts = defaultdict(int)
        for item in items:
            counts[item] += 1
        return {k: v for k, v in counts.items() if v > 1}  # Only return repeated items
    
    def _observe_manifestation(self, manifest_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Observe what naturally manifested in the manifold.
        Uses proper delegation to observe collapse events and other manifestations.
        
        Args:
            manifest_result: Result from manifold's manifest_dynamics method
            
        Returns:
            Dictionary with manifestation observation
        """
        # Safety check
        if not isinstance(manifest_result, dict):
            return {
                'type': 'manifestation_observation',
                'time': getattr(self.manifold, 'time', 0.0),
                'observation_time': system_time.time(),
                'events_count': 0,
                'collapses': 0,
                'voids_formed': 0,
                'coherence': 0.0,
                'tension': 0.0,
                'vortices': 0,
                'superposition_count': 0,
                'pathways': {'structure': 0, 'decay': 0},
                'error': 'Invalid manifest_result'
            }
            
        observation = {
            'type': 'manifestation_observation',
            'time': self.manifold.time,
            'observation_time': system_time.time(),
            'events_count': len(self._safe_get(manifest_result, 'events', default_value=[])),
            'collapses': self._safe_get(manifest_result, 'collapses', default_value=0),
            'voids_formed': self._safe_get(manifest_result, 'voids_formed', default_value=0),
            'coherence': self._safe_get(manifest_result, 'coherence', default_value=0.0),
            'tension': self._safe_get(manifest_result, 'tension', default_value=0.0),
            'vortices': self._safe_get(manifest_result, 'vortices', default_value=0),
            'superposition_count': self._safe_get(manifest_result, 'superposition_count', default_value=0),
            'pathways': {
                'structure': self._safe_get(manifest_result, 'structure_pathways', default_value=0),
                'decay': self._safe_get(manifest_result, 'decay_pathways', default_value=0)
            }
        }
        
        # Observe collapse events in more detail if there are any
        if observation['collapses'] > 0:
            # Get recent collapses using proper delegation if available
            if hasattr(self.manifold, 'get_recent_collapses'):
                try:
                    recent_collapses = self.manifold.get_recent_collapses(observation['collapses'])
                except Exception as e:
                    print(f"Error getting recent collapses: {e}")
                    recent_collapses = []
            elif hasattr(self.manifold, 'collapse_history'):
                # Fallback to direct property access
                try:
                    recent_collapses = self.manifold.collapse_history[-observation['collapses']:]
                except Exception as e:
                    print(f"Error accessing collapse history: {e}")
                    recent_collapses = []
            else:
                recent_collapses = []
                
            if recent_collapses:
                # Observe collapse patterns
                source_grains = [
                    self._safe_get(event, 'source', 
                                  fallback_keys=['grain_id'], 
                                  default_value='unknown') 
                    for event in recent_collapses
                ]
                target_grains = [
                    self._safe_get(event, 'target', 
                                  fallback_keys=['grain_id'], 
                                  default_value='unknown') 
                    for event in recent_collapses
                ]
                
                # Look for repeated patterns
                repeated_sources = self._count_repetitions(source_grains)
                repeated_targets = self._count_repetitions(target_grains)
                
                # Record most active grains
                observation['active_sources'] = [item for item in repeated_sources.items()]
                observation['active_targets'] = [item for item in repeated_targets.items()]
                
                # Use proper delegation for collapse metrics if available
                if hasattr(self.manifold, 'analyze_recent_collapses'):
                    try:
                        collapse_metrics = self.manifold.analyze_recent_collapses(recent_collapses)
                        observation.update(collapse_metrics)
                    except Exception as e:
                        print(f"Error analyzing recent collapses: {e}")
                else:
                    # Fallback to direct analysis
                    positive_collapses = 0
                    negative_collapses = 0
                    from_superposition = 0
                    recursive_collapses = 0
                    field_genesis_collapses = 0
                    
                    for e in recent_collapses:
                        # Count polarity distribution
                        if self._safe_get(e, 'polarity', default_value=0) > 0:
                            positive_collapses += 1
                        elif self._safe_get(e, 'polarity', default_value=0) < 0:
                            negative_collapses += 1
                            
                        # Count collapses from superposition
                        if self._safe_get(e, 'from_superposition', default_value=False):
                            from_superposition += 1
                        
                        # Count field-genesis events
                        if self._safe_get(e, 'field_genesis', default_value=False):
                            field_genesis_collapses += 1
                            
                        # Detect recursive collapses (backflow)
                        source_id = self._safe_get(e, 'source')
                        target_id = self._safe_get(e, 'target')
                        
                        if source_id and target_id:
                            target_grain = self.manifold.get_grain(target_id)
                            if target_grain and hasattr(target_grain, 'ancestry'):
                                if source_id in target_grain.ancestry:
                                    recursive_collapses += 1
                    
                    # Add to observation
                    observation['polarity_distribution'] = {
                        'positive_collapses': positive_collapses,
                        'negative_collapses': negative_collapses
                    }
                    observation['from_superposition'] = from_superposition
                    observation['recursive_collapses'] = recursive_collapses
                    observation['field_genesis_collapses'] = field_genesis_collapses
                    
                    # Calculate ratios for easier analysis
                    total_collapses = len(recent_collapses)
                    if total_collapses > 0:  # Prevent division by zero
                        observation['positive_ratio'] = positive_collapses / total_collapses
                        observation['negative_ratio'] = negative_collapses / total_collapses
                        observation['superposition_collapse_ratio'] = from_superposition / total_collapses
                        observation['recursive_collapse_ratio'] = recursive_collapses / total_collapses
                        observation['field_genesis_ratio'] = field_genesis_collapses / total_collapses
                
                # Observe for causal relations if enabled
                if self.config['trace_causal_relations']:
                    for event in recent_collapses:
                        self._observe_causal_relation(event)
        
        # Observe void formation events
        if observation['voids_formed'] > 0:
            # Use proper delegation if available
            if hasattr(self.manifold, 'get_recent_voids'):
                try:
                    recent_voids = self.manifold.get_recent_voids(observation['voids_formed'])
                except Exception as e:
                    print(f"Error getting recent voids: {e}")
                    recent_voids = []
            elif hasattr(self.manifold, 'void_formation_events'):
                # Fallback to direct property access
                try:
                    recent_voids = self.manifold.void_formation_events[-observation['voids_formed']:]
                except Exception as e:
                    print(f"Error accessing void formation events: {e}")
                    recent_voids = []
            else:
                recent_voids = []
                
            # Calculate average void metrics
            if recent_voids:
                try:
                    observation['void_metrics'] = {
                        'avg_tension': sum(self._safe_get(v, 'tension', default_value=0.0) for v in recent_voids) / len(recent_voids),
                        'avg_strength': sum(self._safe_get(v, 'void_strength', default_value=0.0) for v in recent_voids) / len(recent_voids)
                    }
                    
                    # Track most common void centers
                    void_centers = [self._safe_get(v, 'center_point') for v in recent_voids]
                    repeated_centers = self._count_repetitions([c for c in void_centers if c is not None])
                    observation['void_centers'] = [item for item in repeated_centers.items()]
                except Exception as e:
                    print(f"Error calculating void metrics: {e}")
                    observation['void_metrics'] = {'avg_tension': 0.0, 'avg_strength': 0.0}
                    observation['void_centers'] = []
        
        # Store in memory
        self.collapse_observations.append(observation)
        
        return observation
    
    def _observe_causal_relation(self, event: Dict[str, Any]):
        """
        Observe a causal relation from a collapse event.
        
        Args:
            event: Collapse event dictionary
        """
        # Skip if not a collapse event or missing required fields
        if self._safe_get(event, 'type') != 'collapse' or 'source' not in event or 'target' not in event:
            return
            
        source_id = event['source']
        target_id = event['target']
        
        # Record causal relation
        self.causal_observations[(source_id, target_id)].append({
            'time': self._safe_get(event, 'time', default_value=self.manifold.time),
            'strength': self._safe_get(event, 'strength', default_value=0.5),
            'polarity': self._safe_get(event, 'polarity', default_value=0.0)
        })
        
        # Check for recursive causation (loops)
        for other_target, relations in self.causal_observations.items():
            if isinstance(other_target, tuple) and len(other_target) == 2:
                causal_source, causal_target = other_target
                if causal_source == target_id and causal_target == source_id:
                    # Found a causal loop
                    self.recursive_causal_paths.append({
                        'path': [source_id, target_id, source_id],
                        'last_event_time': self._safe_get(event, 'time', default_value=self.manifold.time),
                        'strength': self._safe_get(event, 'strength', default_value=0.5) * 
                                   self._safe_get(relations[-1], 'strength', default_value=0.5) if relations else 0.0
                    })
    
    def _observe_differential(self, initial_state: Dict[str, Any], 
                            final_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Observe changes between initial and final states.
        Enhanced to track differential patterns related to backflow and absence.
        
        Args:
            initial_state: Initial state observation
            final_state: Final state observation
            
        Returns:
            Dictionary with differential observation
        """
        # Handle potential missing data with safe defaults
        initial_field = self._safe_get(initial_state, 'field_structure', default_value={})
        final_field = self._safe_get(final_state, 'field_structure', default_value={})
        
        # Calculate observed changes
        differential = {
            'type': 'differential_observation',
            'time_delta': self._safe_get(final_state, 'time', default_value=0.0) - 
                         self._safe_get(initial_state, 'time', default_value=0.0),
            'real_time_delta': self._safe_get(final_state, 'observation_time', default_value=0.0) - 
                              self._safe_get(initial_state, 'observation_time', default_value=0.0),
            'collapse_delta': self._safe_get(final_state, 'collapse_count', default_value=0) - 
                             self._safe_get(initial_state, 'collapse_count', default_value=0),
            'void_delta': self._safe_get(final_state, 'void_count', default_value=0) - 
                         self._safe_get(initial_state, 'void_count', default_value=0),
            'vortex_delta': self._safe_get(final_state, 'vortex_count', default_value=0) - 
                           self._safe_get(initial_state, 'vortex_count', default_value=0),
            'superposition_delta': self._safe_get(final_state, 'superposition_count', default_value=0) - 
                                  self._safe_get(initial_state, 'superposition_count', default_value=0),
            'pathway_delta': {
                'structure': self._safe_get(final_state, 'pathway_counts', {}).get('structure', 0) - 
                            self._safe_get(initial_state, 'pathway_counts', {}).get('structure', 0),
                'decay': self._safe_get(final_state, 'pathway_counts', {}).get('decay', 0) - 
                        self._safe_get(initial_state, 'pathway_counts', {}).get('decay', 0)
            }
        }
        
        # Calculate field structure changes
        self._observe_field_structure_changes(differential, initial_field, final_field)
        
        # Calculate specialty metric changes
        self._observe_specialty_metric_changes(differential, initial_field, final_field)
        
        # Track significant structure changes
        self._track_significant_structure_changes(differential)
        
        # Track standard emergence events
        self._track_standard_emergence_events(differential)
        
        return differential
    
    def _observe_field_structure_changes(self, differential, initial_field, final_field):
        """
        Observe changes in basic field structure metrics.
        
        Args:
            differential: Differential dictionary to update
            initial_field: Initial field structure dictionary
            final_field: Final field structure dictionary
        """
        # Track awareness changes
        if 'mean_awareness' in initial_field and 'mean_awareness' in final_field:
            differential['awareness_delta'] = final_field['mean_awareness'] - initial_field['mean_awareness']
            
        # Track activation changes
        if 'mean_activation' in initial_field and 'mean_activation' in final_field:
            differential['activation_delta'] = final_field['mean_activation'] - initial_field['mean_activation']
            
        # Track saturation changes
        if 'mean_saturation' in initial_field and 'mean_saturation' in final_field:
            differential['saturation_delta'] = final_field['mean_saturation'] - initial_field['mean_saturation']
            
        # Track saturation variance changes
        if 'saturation_variance' in initial_field and 'saturation_variance' in final_field:
            differential['saturation_variance_delta'] = final_field['saturation_variance'] - initial_field['saturation_variance']
            
        # Track coherence changes
        if 'field_coherence' in initial_field and 'field_coherence' in final_field:
            differential['coherence_delta'] = final_field['field_coherence'] - initial_field['field_coherence']
            
        # Track tension changes
        if 'system_tension' in initial_field and 'system_tension' in final_field:
            differential['tension_delta'] = final_field['system_tension'] - initial_field['system_tension']
            
        # Track polarity changes
        if 'mean_polarity' in initial_field and 'mean_polarity' in final_field:
            differential['polarity_delta'] = final_field['mean_polarity'] - initial_field['mean_polarity']
    
    def _observe_specialty_metric_changes(self, differential, initial_field, final_field):
        """
        Observe changes in specialty metrics like ancestry, curvature, etc.
        
        Args:
            differential: Differential dictionary to update
            initial_field: Initial field structure dictionary
            final_field: Final field structure dictionary
        """
        # Track ancestry changes
        if 'self_referential_count' in initial_field and 'self_referential_count' in final_field:
            differential['self_ref_delta'] = final_field['self_referential_count'] - initial_field['self_referential_count']
            
            # Check for significant self-reference changes
            if self._safe_get(differential, 'self_ref_delta', default_value=0) > 3:
                self._record_emergence_event(differential, {
                    'type': 'self_reference_surge',
                    'magnitude': differential['self_ref_delta'],
                    'description': 'Significant increase in self-referential grains'
                })
                
        # Track field genesis changes
        if 'field_genesis_count' in initial_field and 'field_genesis_count' in final_field:
            differential['field_genesis_delta'] = final_field['field_genesis_count'] - initial_field['field_genesis_count']
            
            # Check for significant field genesis changes
            if self._safe_get(differential, 'field_genesis_delta', default_value=0) > 3:
                self._record_emergence_event(differential, {
                    'type': 'field_genesis_burst',
                    'magnitude': differential['field_genesis_delta'],
                    'description': 'Significant burst of field-genesis events'
                })
            
        # Track major polarity shifts
        if 'polarity_delta' in differential and abs(self._safe_get(differential, 'polarity_delta', default_value=0)) > 0.2:
            # Significant shift in system bias (structure vs decay)
            self._record_emergence_event(differential, {
                'type': 'polarity_shift',
                'magnitude': differential['polarity_delta'],
                'description': 'Significant shift in system polarity bias'
            })
        
        # Track lightlike ratio changes
        if 'lightlike_ratio' in initial_field and 'lightlike_ratio' in final_field:
            differential['lightlike_ratio_delta'] = final_field['lightlike_ratio'] - initial_field['lightlike_ratio']
            
            # Check for significant lightlike ratio changes
            if abs(self._safe_get(differential, 'lightlike_ratio_delta', default_value=0)) > 0.1:
                # Significant change in lightlike propagation capability
                event_type = 'lightlike_increase' if differential['lightlike_ratio_delta'] > 0 else 'lightlike_decrease'
                self._record_emergence_event(differential, {
                    'type': event_type,
                    'magnitude': abs(differential['lightlike_ratio_delta']),
                    'description': f'Significant {"increase" if event_type == "lightlike_increase" else "decrease"} in lightlike grain ratio'
                })
        
        # Track backflow metric changes
        if 'high_saturation_ratio' in initial_field and 'high_saturation_ratio' in final_field:
            differential['high_saturation_delta'] = final_field['high_saturation_ratio'] - initial_field['high_saturation_ratio']
            
            # Check for significant high saturation changes
            if self._safe_get(differential, 'high_saturation_delta', default_value=0) > 0.1:
                # Significant increase in high-saturation grains (potential backflow regions)
                self._record_emergence_event(differential, {
                    'type': 'backflow_potential_increase',
                    'magnitude': differential['high_saturation_delta'],
                    'description': 'Significant increase in high-saturation grain ratio (backflow potential)'
                })
        
        # Track ancestry depth changes
        if 'avg_ancestry_depth' in initial_field and 'avg_ancestry_depth' in final_field:
            differential['ancestry_depth_delta'] = final_field['avg_ancestry_depth'] - initial_field['avg_ancestry_depth']
            
            # Check for significant ancestry depth changes
            if self._safe_get(differential, 'ancestry_depth_delta', default_value=0) > 1.0:
                # Significant increase in ancestry depth (memory complexity)
                self._record_emergence_event(differential, {
                    'type': 'ancestry_complexity_increase',
                    'magnitude': differential['ancestry_depth_delta'],
                    'description': 'Significant increase in ancestry tree depth'
                })
        
        # Track system curvature changes
        if 'system_curvature' in initial_field and 'system_curvature' in final_field:
            differential['curvature_delta'] = final_field['system_curvature'] - initial_field['system_curvature']
            
            # Significant curvature changes indicate shifts in recursive dynamics
            if abs(self._safe_get(differential, 'curvature_delta', default_value=0)) > 0.2:
                event_type = 'curvature_increase' if differential['curvature_delta'] > 0 else 'curvature_decrease'
                self._record_emergence_event(differential, {
                    'type': event_type,
                    'magnitude': abs(differential['curvature_delta']),
                    'description': f'Significant {"increase" if event_type == "curvature_increase" else "decrease"} in system curvature'
                })
                
        # Track changes in toroidal structure coherence
        if ('global_coherence' in initial_field and 'global_coherence' in final_field and
            abs(final_field['global_coherence'] - initial_field['global_coherence']) > 0.2):
            
            # Significant coherence change
            coherence_delta = final_field['global_coherence'] - initial_field['global_coherence']
            self._record_emergence_event(differential, {
                'type': 'coherence_transition',
                'magnitude': coherence_delta,
                'description': 'Phase transition in global coherence'
            })
    
    def _track_significant_structure_changes(self, differential: Dict[str, Any]):
        """
        Track significant structural changes for future reference.
        Called by observe_differential to analyze structural transitions.
        
        Args:
            differential: Differential observation dictionary
        """
        # Add emergence events based on significant changes
        events = []
        
        # Check for vortex formation
        if self._safe_get(differential, 'vortex_delta', default_value=0) > 0:
            events.append({
                'type': 'vortex_formation',
                'magnitude': differential['vortex_delta'],
                'description': f"Formation of {differential['vortex_delta']} new vortices"
            })
        
        # Check for superposition collapse
        if self._safe_get(differential, 'superposition_delta', default_value=0) < 0:
            events.append({
                'type': 'superposition_collapse',
                'magnitude': abs(differential['superposition_delta']),
                'description': f"Collapse of {abs(differential['superposition_delta'])} superposition states"
            })
        
        # Check for pathway formation
        structure_pathway_delta = self._safe_get(differential, 'pathway_delta', {}).get('structure', 0)
        decay_pathway_delta = self._safe_get(differential, 'pathway_delta', {}).get('decay', 0)
        
        if structure_pathway_delta > 0:
            events.append({
                'type': 'structure_pathway_formation',
                'magnitude': structure_pathway_delta,
                'description': f"Formation of {structure_pathway_delta} new structure-building pathways"
            })
            
        if decay_pathway_delta > 0:
            events.append({
                'type': 'decay_pathway_formation',
                'magnitude': decay_pathway_delta,
                'description': f"Formation of {decay_pathway_delta} new decay pathways"
            })
        
        # Add events to differential
        differential['emergence_events'] = events
        
        # Update metrics
        self.metrics['observed_emergence_events'] += len(events)
    
    def _track_standard_emergence_events(self, differential: Dict[str, Any]):
        """
        Track standard emergence events based on differential observation.
        Similar to _track_significant_structure_changes but focuses on
        standard metrics rather than specialized structures.
        
        Args:
            differential: Differential observation dictionary
        """
        # Check for void formation bursts
        if self._safe_get(differential, 'void_delta', default_value=0) > 3:
            self._record_emergence_event(differential, {
                'type': 'void_formation_burst',
                'magnitude': differential['void_delta'],
                'description': f"Burst of {differential['void_delta']} void formations"
            })
        
        # Check for collapse bursts
        if self._safe_get(differential, 'collapse_delta', default_value=0) > 5:
            self._record_emergence_event(differential, {
                'type': 'collapse_burst',
                'magnitude': differential['collapse_delta'],
                'description': f"Burst of {differential['collapse_delta']} collapse events"
            })
        
        # Check for significant awareness increases
        if 'awareness_delta' in differential and differential['awareness_delta'] > 0.3:
            self._record_emergence_event(differential, {
                'type': 'awareness_surge',
                'magnitude': differential['awareness_delta'],
                'description': 'Significant increase in field awareness'
            })
    
    def _record_emergence_event(self, differential: Dict[str, Any], event: Dict[str, Any]):
        """
        Record an emergence event in the differential observation.
        
        Args:
            differential: Differential observation dictionary
            event: Emergence event dictionary
        """
        # Ensure emergence_events exists
        if 'emergence_events' not in differential:
            differential['emergence_events'] = []
            
        # Add event
        differential['emergence_events'].append(event)
        
        # Update metrics
        self.metrics['observed_emergence_events'] += 1
    
    def _observe_backflow_dynamics(self) -> Dict[str, Any]:
        """
        Observe backflow dynamics in the manifold.
        Backflow is when collapse patterns create recursive structures.
        
        Returns:
            Dictionary with backflow observations
        """
        if not self.config['observe_backflow']:
            return {}
            
        # Initialize observation
        observation = {
            'type': 'backflow_observation',
            'time': self.manifold.time,
            'observation_time': system_time.time(),
            'detected_backflow': 0,
            'backflow_regions': [],
            'recursive_indices': []
        }
        
        # Check for recursive causal paths (indicating backflow)
        if hasattr(self, 'recursive_causal_paths') and self.recursive_causal_paths:
            # Focus on recent recursive paths
            recent_paths = [path for path in self.recursive_causal_paths 
                           if path['last_event_time'] > self.manifold.time - 5.0]
            
            observation['detected_backflow'] = len(recent_paths)
            
            # Calculate backflow metrics
            if recent_paths:
                observation['backflow_metrics'] = {
                    'mean_strength': sum(path['strength'] for path in recent_paths) / len(recent_paths),
                    'path_count': len(recent_paths)
                }
                
                # Extract backflow regions (grains involved in recursive paths)
                region_grains = set()
                for path in recent_paths:
                    for grain_id in path['path']:
                        region_grains.add(grain_id)
                
                # Get metrics for each region grain
                for grain_id in region_grains:
                    grain = self.manifold.get_grain(grain_id)
                    if grain:
                        region = {
                            'grain_id': grain_id,
                            'saturation': grain.grain_saturation,
                            'activation': grain.grain_activation,
                            'awareness': grain.awareness,
                            'polarity': getattr(grain, 'polarity', 0.0),
                            'collapse_metric': grain.collapse_metric
                        }
                        
                        observation['backflow_regions'].append(region)
        
        # Get recursive indices from ancestry data
        if hasattr(self.manifold, 'get_ancestry_distribution'):
            try:
                ancestry_data = self.manifold.get_ancestry_distribution()
                recursive_indices = self._safe_get(ancestry_data, 'recursive_indices', default_value={})
                
                # Extract high recursive index grains (strong backflow potential)
                high_recursive = {grain_id: index for grain_id, index in recursive_indices.items() if index > 0.5}
                
                observation['recursive_indices'] = [
                    {'grain_id': grain_id, 'recursive_index': index}
                    for grain_id, index in high_recursive.items()
                ]
                
                # Calculate backflow intensity from recursive indices
                if recursive_indices:
                    avg_recursive = sum(recursive_indices.values()) / len(recursive_indices)
                    observation['backflow_intensity'] = avg_recursive
                    self.metrics['backflow_intensity'] = avg_recursive
            except Exception as e:
                print(f"Error getting ancestry distribution for backflow observation: {e}")
        
        # Store in memory
        self.backflow_observations.append(observation)
        
        return observation
    
    def _observe_absence_patterns(self) -> Dict[str, Any]:
        """
        Observe absence patterns in the manifold.
        Absence patterns are where structure fails to form or decays.
        
        Returns:
            Dictionary with absence pattern observations
        """
        if not self.config['track_absence_patterns']:
            return {}
            
        # Initialize observation
        observation = {
            'type': 'absence_observation',
            'time': self.manifold.time,
            'observation_time': system_time.time(),
            'void_count': 0,
            'tension_regions': [],
            'decay_rates': []
        }
        
        # Count void formation events
        if hasattr(self.manifold, 'void_formation_events'):
            observation['void_count'] = len(self.manifold.void_formation_events)
            
            # Get recent void events
            recent_voids = [event for event in self.manifold.void_formation_events 
                          if event.get('time', 0) > self.manifold.time - 5.0]
            
            observation['recent_void_count'] = len(recent_voids)
            
            if recent_voids:
                # Extract void centers
                void_centers = [event.get('center_point') for event in recent_voids]
                
                # Track void metrics
                void_tensions = [event.get('tension', 0.0) for event in recent_voids]
                void_strengths = [event.get('void_strength', 0.0) for event in recent_voids]
                
                observation['void_metrics'] = {
                    'mean_tension': sum(void_tensions) / len(void_tensions) if void_tensions else 0.0,
                    'mean_strength': sum(void_strengths) / len(void_strengths) if void_strengths else 0.0
                }
        
        # Find tension regions by looking at points with high structural tension
        tension_regions = []
        
        for point_id, point in self.config_space.points.items():
            if hasattr(point, 'structural_tension') and point.structural_tension > 0.5:
                region = {
                    'point_id': point_id,
                    'tension': point.structural_tension
                }
                
                # Add grain data if available
                if point_id in self.manifold.grains:
                    grain = self.manifold.grains[point_id]
                    region.update({
                        'awareness': grain.awareness,
                        'saturation': grain.grain_saturation,
                        'polarity': getattr(grain, 'polarity', 0.0)
                    })
                
                tension_regions.append(region)
        
        observation['tension_regions'] = tension_regions
        observation['tension_region_count'] = len(tension_regions)
        
        # Calculate decay rates by looking at negative polarity regions
        decay_regions = []
        
        for grain_id, grain in self.manifold.grains.items():
            if hasattr(grain, 'polarity') and grain.polarity < -0.3:
                region = {
                    'grain_id': grain_id,
                    'polarity': grain.polarity,
                    'saturation': grain.grain_saturation,
                    'activation': grain.grain_activation
                }
                
                decay_regions.append(region)
        
        observation['decay_regions'] = decay_regions
        observation['decay_region_count'] = len(decay_regions)
        
        # Store in memory
        self.absence_observations.append(observation)
        
        # Update metrics
        self.metrics['absence_pattern_count'] = len(tension_regions) + len(decay_regions)
        
        return observation
    
    def _observe_ancestry_patterns(self) -> Dict[str, Any]:
        """
        Observe ancestry patterns in the manifold.
        Ancestry patterns show how memory creates structure across time.
        
        Returns:
            Dictionary with ancestry pattern observations
        """
        if not self.config['observe_ancestry']:
            return {}
            
        # Initialize observation
        observation = {
            'type': 'ancestry_observation',
            'time': self.manifold.time,
            'observation_time': system_time.time(),
            'self_referential_count': 0,
            'ancestry_depths': {},
            'ancestry_trees': []
        }
        
        # Get ancestry distribution
        if hasattr(self.manifold, 'get_ancestry_distribution'):
            try:
                ancestry_data = self.manifold.get_ancestry_distribution()
                
                # Extract key metrics
                observation['self_referential_count'] = self._safe_get(
                    ancestry_data, 'self_referential_count', 
                    ['self_reference_count'], 0)
                
                observation['average_ancestry_size'] = self._safe_get(
                    ancestry_data, 'average_ancestry_size', 
                    ['average_depth'], 0.0)
                
                observation['ancestry_sizes'] = self._safe_get(
                    ancestry_data, 'distribution', 
                    ['ancestry_sizes'], {})
                
                # Extract recursive indices
                observation['recursive_indices'] = self._safe_get(
                    ancestry_data, 'recursive_indices', default_value={})
                
                # Extract curvature metrics
                observation['curvature_metrics'] = self._safe_get(
                    ancestry_data, 'curvature_metrics', default_value={})
                
                # Update metrics
                self.metrics['self_referential_count'] = observation['self_referential_count']
                self.metrics['ancestry_depth'] = observation['average_ancestry_size']
                
                # Extract high-curvature grains
                if 'curvature_metrics' in ancestry_data:
                    curvature_metrics = ancestry_data['curvature_metrics']
                    high_curvature = {grain_id: curve for grain_id, curve in curvature_metrics.items() 
                                    if curve > 0.5}
                    observation['high_curvature_grains'] = high_curvature
                    
                    # Calculate curvature metric
                    if curvature_metrics:
                        avg_curvature = sum(curvature_metrics.values()) / len(curvature_metrics)
                        observation['average_curvature'] = avg_curvature
                        self.metrics['curvature_metric'] = avg_curvature
            except Exception as e:
                print(f"Error getting ancestry distribution for ancestry observation: {e}")
        
        # Sample some ancestry trees for detailed observation
        if len(self.manifold.grains) > 0:
            # Select a few random grains with non-empty ancestry
            sample_size = min(5, len(self.manifold.grains))
            sample_grains = random.sample(list(self.manifold.grains.keys()), sample_size)
            
            for grain_id in sample_grains:
                grain = self.manifold.get_grain(grain_id)
                if not grain or not hasattr(grain, 'ancestry') or not grain.ancestry:
                    continue
                    
                # Create ancestry tree
                tree = {
                    'grain_id': grain_id,
                    'ancestry': list(grain.ancestry),
                    'self_referential': grain_id in grain.ancestry,
                    'collapse_metric': grain.collapse_metric,
                    'ancestry_size': len(grain.ancestry)
                }
                
                # Add recursive ancestry to track cycles
                recursive_ancestry = []
                for ancestor_id in grain.ancestry:
                    if ancestor_id in self.manifold.grains:
                        ancestor = self.manifold.get_grain(ancestor_id)
                        if hasattr(ancestor, 'ancestry') and grain_id in ancestor.ancestry:
                            recursive_ancestry.append(ancestor_id)
                
                tree['recursive_ancestry'] = recursive_ancestry
                observation['ancestry_trees'].append(tree)
        
        # Store in memory
        self.ancestry_observations.append(observation)
        
        return observation
    
    def _update_metrics(self, manifestation_observation, differential_observation,
                      backflow_observation, absence_observation, ancestry_observation):
        """
        Update internal metrics based on observations.
        
        Args:
            manifestation_observation: Observation of natural manifestation
            differential_observation: Observation of differential changes
            backflow_observation: Observation of backflow dynamics
            absence_observation: Observation of absence patterns
            ancestry_observation: Observation of ancestry patterns
        """
        # Update basic metrics
        self.metrics['observed_collapses'] += self._safe_get(manifestation_observation, 'collapses', default_value=0)
        self.metrics['observed_voids'] += self._safe_get(manifestation_observation, 'voids_formed', default_value=0)
        self.metrics['observed_coherence'] = self._safe_get(manifestation_observation, 'coherence', default_value=0.0)
        self.metrics['observed_field_tension'] = self._safe_get(manifestation_observation, 'tension', default_value=0.0)
        self.metrics['observed_vortices'] = self._safe_get(manifestation_observation, 'vortices', default_value=0)
        
        # Update pathway counts
        pathway_counts = self._safe_get(manifestation_observation, 'pathways', default_value={'structure': 0, 'decay': 0})
        self.metrics['observed_pathways']['structure'] = pathway_counts.get('structure', 0)
        self.metrics['observed_pathways']['decay'] = pathway_counts.get('decay', 0)
        
        # Update lightlike ratio
        field_structure = self._safe_get(differential_observation, 'field_structure', default_value={})
        self.metrics['lightlike_ratio'] = self._safe_get(field_structure, 'lightlike_ratio', default_value=0.0)
        
        # Update recursive collapse count from ancestry
        if ancestry_observation:
            self.metrics['recursive_collapse_count'] = len(self._safe_get(
                ancestry_observation, 'recursive_indices', default_value={}))
        
        # Update field genesis count
        if 'field_genesis_collapses' in manifestation_observation:
            self.metrics['field_genesis_count'] += manifestation_observation['field_genesis_collapses']
        
        # Update phase reversal count
        if 'phase_transition' in differential_observation.get('emergence_events', []):
            self.metrics['phase_reversal_count'] += 1
            
        # Update backflow intensity
        if backflow_observation and 'backflow_intensity' in backflow_observation:
            self.metrics['backflow_intensity'] = backflow_observation['backflow_intensity']