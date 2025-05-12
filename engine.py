"""
EmergentDualityEngine - Pure Observer for Collapse Geometry

This engine implements a purely observational approach to collapse dynamics
aligned with the updated RelationalManifold implementation. It observes what
naturally emerges from relational dynamics without imposing changes.

Key principles:
- The engine observes; the field *is* the system
- All structure, including time, emerges from the relational dynamics
- Circular recursion patterns emerge naturally from the manifold's constraint dynamics
"""

import math
import numpy as np
import time as system_time
from typing import Dict, List, Tuple, Set, Optional, Any, Callable
from collections import defaultdict, deque

class EmergentDualityEngine:
    """
    Pure observer engine for the Collapse Geometry framework.
    
    This engine doesn't drive the dynamics but simply observes what naturally
    emerges from the relational manifold. It acknowledges that all structure,
    including time, emerges from the relational dynamics themselves.
    
    Features:
    - Focuses on observing natural emergence rather than imposing structure
    - Enhanced detection of circular recursion patterns
    - Improved observation of ancestry and phase dynamics
    - Adaptive interface that works with the manifold's natural evolution
    """
    
    def __init__(self, manifold, state=None, config=None):
        """
        Initialize the observer engine with a manifold and optional configuration.
        
        Args:
            manifold: RelationalManifold to observe
            state: Optional state object for tracking
            config: Optional configuration parameters
        """
        self.manifold = manifold
        self.state = state
        
        # Default configuration for pure observation
        self.config = {
            'observation_detail': 0.8,      # How detailed observations should be
            'observation_rate': 1.0,        # How frequently to observe
            'memory_depth': 50,             # How many observations to keep
            'trace_causal_relations': True, # Whether to trace causal relations
            'record_metrics': True,         # Whether to record metrics
            'observe_ancestry': True,       # Whether to track ancestry patterns
            'observe_backflow': True,       # Whether to track backflow dynamics
            'track_absence_patterns': True, # Whether to track absence patterns
            'observe_circular_patterns': True, # Whether to track circular recursion
        }
        
        # Update with custom config if provided
        if config:
            self.config.update(config)
        
        # Observation memory
        self.collapse_observations = deque(maxlen=self.config['memory_depth'])
        self.structure_observations = deque(maxlen=self.config['memory_depth'])
        self.field_observations = deque(maxlen=self.config['memory_depth'])
        self.circular_observations = deque(maxlen=self.config['memory_depth'])
        self.ancestry_observations = deque(maxlen=self.config['memory_depth'])
        self.absence_observations = deque(maxlen=self.config['memory_depth'])
        
        # Metrics and statistics
        self.metrics = {
            'observation_count': 0,
            'last_observation_time': system_time.time(),
            'observed_emergent_time': 0.0,
            'observed_collapses': 0,
            'observed_voids': 0,
            'observed_coherence': 0.0,
            'observed_field_tension': 0.0,
            'observed_vortices': 0,
            'observed_pathways': {'structure': 0, 'decay': 0},
            
            # Enhanced metrics for circular dynamics
            'lightlike_ratio': 0.0,         # Ratio of lightlike to total grains
            'ancestry_depth': 0.0,          # Average ancestry tree depth
            'self_referential_count': 0,    # Count of self-referential grains
            'field_genesis_count': 0,       # Count of field-genesis events
            'circular_recursion_count': 0,  # Count of grains with circular recursion
            'circular_ancestry_count': 0,   # Count of circular ancestry connections
            'polarity_wraparound_events': 0, # Count of polarity wraparound events
            'circular_coherence': 0.0,      # Measure of phase coherence on circle
        }
        
        # Track causal relations
        if self.config['trace_causal_relations']:
            self.causal_observations = defaultdict(list)
            self.recursive_causal_paths = []
            self.circular_causal_paths = []
        
        # Track special patterns
        if self.config['observe_circular_patterns']:
            self.circular_ancestry_pairs = []
            self.polarity_wraparound_events = []
            self.circular_recursion_grains = set()
        
        if self.config['observe_ancestry']:
            self.ancestry_trees = {}
            self.self_referential_grains = set()
            self.field_genesis_events = []
        
        if self.config['observe_backflow']:
            self.recursive_pathways = []
        
        if self.config['track_absence_patterns']:
            self.void_formations = []
            self.decay_cascades = []
            self.tensile_boundaries = []
    
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
            
            # Observe emergent patterns
            circular_observation = self._observe_circular_recursion()
            ancestry_observation = self._observe_ancestry_patterns()
            absence_observation = self._observe_absence_patterns()
            
            # Update metrics
            self._update_metrics(
                manifestation_observation, 
                differential,
                circular_observation,
                ancestry_observation, 
                absence_observation
            )
        except Exception as e:
            # Graceful error handling
            print(f"Error during observation: {e}")
            return {
                'error': f"Observation failed: {e}",
                'elapsed_time': system_time.time() - start_time
            }
        
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
            'circular_recursion': circular_observation,
            'ancestry_patterns': ancestry_observation,
            'absence_patterns': absence_observation,
            'elapsed_time': system_time.time() - start_time,
            'metrics': self.metrics
        }
    
    def _observe_current_state(self) -> Dict[str, Any]:
        """
        Observe the current state of the manifold without changing it.
        
        Returns:
            Dictionary with current state observation
        """
        # Basic metrics
        grain_count = len(self.manifold.grains)
        
        # Observe field structure
        field_structure = {
            'grain_count': grain_count,
            'emergent_time': self.manifold.time,
            'field_coherence': getattr(self.manifold, 'field_coherence', 0.5),
            'system_tension': getattr(self.manifold, 'system_tension', 0.0),
            'system_stability': getattr(self.manifold, 'system_stability', 0.0),
        }
        
        # Observe grain distribution if there are grains
        if grain_count > 0:
            # Calculate lightlike ratio
            lightlike_count = sum(1 for grain in self.manifold.grains.values() 
                              if grain.grain_saturation < 0.2)
            field_structure['lightlike_ratio'] = lightlike_count / grain_count
            
            # Calculate states ratio
            activity_states = defaultdict(int)
            for grain_id, grain in self.manifold.grains.items():
                activity_state = getattr(grain, 'activity_state', 'active')
                activity_states[activity_state] += 1
                
            for state, count in activity_states.items():
                field_structure[f'{state}_ratio'] = count / grain_count
            
            # Get ancestry metrics
            ancestry_data = self._get_ancestry_metrics()
            field_structure.update(ancestry_data)
            
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
            polarity_values = [getattr(grain, 'polarity', 0.0) for grain in self.manifold.grains.values()]
            field_structure['mean_polarity'] = sum(polarity_values) / grain_count
            field_structure['polarity_variance'] = self._calculate_variance(polarity_values)
            
            # Circular polarity metrics - observe distribution on the circle
            positive_extreme = sum(1 for p in polarity_values if p > 0.9)
            negative_extreme = sum(1 for p in polarity_values if p < -0.9)
            positive_moderate = sum(1 for p in polarity_values if 0.2 < p <= 0.9)
            negative_moderate = sum(1 for p in polarity_values if -0.9 <= p < -0.2)
            neutral = grain_count - positive_extreme - negative_extreme - positive_moderate - negative_moderate
            
            field_structure['positive_extreme_ratio'] = positive_extreme / grain_count
            field_structure['negative_extreme_ratio'] = negative_extreme / grain_count
            field_structure['positive_moderate_ratio'] = positive_moderate / grain_count
            field_structure['negative_moderate_ratio'] = negative_moderate / grain_count
            field_structure['neutral_ratio'] = neutral / grain_count
            
            # Calculate polarity circular coherence
            polarity_angles = [(p + 1) * math.pi for p in polarity_values]  # Map [-1,1] to [0,2π]
            field_structure['polarity_circular_coherence'] = self._calculate_circular_coherence(polarity_angles)
            self.metrics['circular_coherence'] = field_structure['polarity_circular_coherence']
            
            # Calculate collapse metric statistics
            collapse_values = [grain.collapse_metric for grain in self.manifold.grains.values()]
            field_structure['mean_collapse_metric'] = sum(collapse_values) / grain_count
            field_structure['max_collapse_metric'] = max(collapse_values, default=0.0)
        
        # Observe toroidal structure
        field_structure.update(self._observe_toroidal_structure())
        
        # Observe circular recursion
        field_structure.update(self._get_circular_recursion_metrics())
        
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
        
        # Add superposition count
        observation['superposition_count'] = getattr(self.manifold, 'superposition_count', 0)
        
        # Add stability metrics
        if hasattr(self.manifold, 'get_stability_metrics'):
            observation['stability_metrics'] = self.manifold.get_stability_metrics()
        else:
            observation['stability_metrics'] = {
                'system_stability': getattr(self.manifold, 'system_stability', 0.0),
                'stable_duration': getattr(self.manifold, 'stable_duration', 0.0),
                'is_stable': getattr(self.manifold, 'is_system_stable', lambda: False)()
            }
        
        return observation
    
    def _observe_toroidal_structure(self) -> Dict[str, Any]:
        """
        Observe the toroidal structure of the manifold.
        
        Returns:
            Dictionary with toroidal structure observations
        """
        result = {}
        
        # Get vortices if available
        vortices = getattr(self.manifold, 'vortices', [])
        result['vortex_count'] = len(vortices)
        
        if vortices:
            # Calculate vortex metrics
            circulation_values = [abs(v.get('circulation', 0.0)) for v in vortices]
            if circulation_values:
                result['mean_circulation'] = sum(circulation_values) / len(circulation_values)
                result['max_circulation'] = max(circulation_values)
            
            # Analyze vortex polarities
            vortex_polarities = [v.get('polarity', 0.0) for v in vortices if 'polarity' in v]
            if vortex_polarities:
                result['mean_vortex_polarity'] = sum(vortex_polarities) / len(vortex_polarities)
                
                # Analyze vortices with extreme polarities
                extreme_polarity_vortices = sum(1 for p in vortex_polarities if abs(p) > 0.9)
                result['extreme_polarity_vortex_ratio'] = extreme_polarity_vortices / len(vortex_polarities)
                
                # Calculate circular coherence of vortex polarities
                vortex_polarity_angles = [(p + 1) * math.pi for p in vortex_polarities]
                result['vortex_polarity_coherence'] = self._calculate_circular_coherence(vortex_polarity_angles)
        
        # Get lightlike pathways if available
        pathways = getattr(self.manifold, 'lightlike_pathways', {'structure': [], 'decay': []})
        result['structure_pathways'] = len(pathways.get('structure', []))
        result['decay_pathways'] = len(pathways.get('decay', []))
        
        # Calculate pathway metrics if available
        structure_pathways = pathways.get('structure', [])
        decay_pathways = pathways.get('decay', [])
        
        if structure_pathways:
            structure_lengths = [len(p.get('nodes', [])) for p in structure_pathways]
            result['mean_structure_pathway_length'] = sum(structure_lengths) / len(structure_lengths)
            result['max_structure_pathway_length'] = max(structure_lengths)
            
            # Calculate lightlike ratio along structure pathways
            lightlike_ratios = [p.get('lightlike_ratio', 0.0) for p in structure_pathways]
            result['structure_pathway_lightlike_ratio'] = sum(lightlike_ratios) / len(lightlike_ratios)
        
        if decay_pathways:
            decay_lengths = [len(p.get('nodes', [])) for p in decay_pathways]
            result['mean_decay_pathway_length'] = sum(decay_lengths) / len(decay_lengths)
            result['max_decay_pathway_length'] = max(decay_lengths)
            
            # Calculate lightlike ratio along decay pathways
            lightlike_ratios = [p.get('lightlike_ratio', 0.0) for p in decay_pathways]
            result['decay_pathway_lightlike_ratio'] = sum(lightlike_ratios) / len(lightlike_ratios)
        
        return result
    
    def _get_ancestry_metrics(self) -> Dict[str, Any]:
        """
        Get ancestry metrics from the manifold.
        
        Returns:
            Dictionary with ancestry metrics
        """
        result = {}
        
        # Calculate ancestry metrics manually
        ancestry_counts = defaultdict(int)
        self_referential_count = 0
        
        for grain_id, grain in self.manifold.grains.items():
            ancestry = getattr(grain, 'ancestry', set())
            ancestry_size = len(ancestry)
            ancestry_counts[ancestry_size] += 1
            
            # Count self-referential grains
            if grain_id in ancestry:
                self_referential_count += 1
        
        result['ancestry_distribution'] = dict(ancestry_counts)
        result['self_referential_count'] = self_referential_count
        
        # Calculate average ancestry depth
        total_grains = len(self.manifold.grains)
        if total_grains > 0:
            total_ancestry = sum(size * count for size, count in ancestry_counts.items())
            result['avg_ancestry_depth'] = total_ancestry / total_grains
        else:
            result['avg_ancestry_depth'] = 0.0
        
        # Get field genesis count from collapse history
        if hasattr(self.manifold, 'collapse_history'):
            field_genesis_count = sum(1 for event in self.manifold.collapse_history 
                                   if event.get('field_genesis', False))
            result['field_genesis_count'] = field_genesis_count
        else:
            result['field_genesis_count'] = 0
        
        return result
    
    def _get_circular_recursion_metrics(self) -> Dict[str, Any]:
        """
        Get circular recursion metrics from the manifold.
        
        Returns:
            Dictionary with circular recursion metrics
        """
        result = {}
        
        # Count grains with circular recursion patterns
        circular_recursion_count = 0
        circular_ancestry_pairs = 0
        circular_factors = []
        polarity_extremes_count = 0
        
        for grain_id, grain in self.manifold.grains.items():
            # Check for circular recursion factor
            if hasattr(grain, 'circular_recursion_factor'):
                circular_recursion_count += 1
                circular_factors.append(grain.circular_recursion_factor)
            
            # Check for circular ancestry
            if hasattr(grain, 'circular_ancestry'):
                circular_ancestry_pairs += len(grain.circular_ancestry)
            
            # Count grains with extreme polarity (near +1 or -1)
            if hasattr(grain, 'polarity') and abs(grain.polarity) > 0.9:
                polarity_extremes_count += 1
        
        result['circular_recursion_count'] = circular_recursion_count
        result['circular_ancestry_pairs'] = circular_ancestry_pairs
        result['polarity_extremes_count'] = polarity_extremes_count
        
        # Calculate average circular recursion factor
        if circular_factors:
            result['circular_recursion_factor_avg'] = sum(circular_factors) / len(circular_factors)
        else:
            result['circular_recursion_factor_avg'] = 0.0
        
        return result
    
    def _observe_manifestation(self, manifest_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Observe what naturally manifested in the manifold.
        
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
                'error': 'Invalid manifest_result'
            }
            
        observation = {
            'type': 'manifestation_observation',
            'time': self.manifold.time,
            'observation_time': system_time.time(),
            'events_count': len(manifest_result.get('events', [])),
            'collapses': manifest_result.get('collapses', 0),
            'voids_formed': manifest_result.get('voids_formed', 0),
            'coherence': manifest_result.get('coherence', 0.0),
            'tension': manifest_result.get('tension', 0.0),
            'vortices': manifest_result.get('vortices', 0),
            'superposition_count': manifest_result.get('superposition_count', 0),
            'pathways': {
                'structure': manifest_result.get('structure_pathways', 0),
                'decay': manifest_result.get('decay_pathways', 0)
            },
            'is_stable': manifest_result.get('is_stable', False)
        }
        
        # Observe circular recursion events
        if 'events' in manifest_result:
            # Look for polarity wraparound events in the events list
            polarity_wraparound_events = []
            
            for event in manifest_result['events']:
                # Check if this is a collapse event with polarity info
                if event.get('type') == 'collapse' and 'polarity' in event:
                    # Check if source and target have opposite extreme polarities
                    source_id = event.get('source')
                    target_id = event.get('target')
                    
                    if not source_id or not target_id:
                        continue
                    
                    source_grain = self.manifold.get_grain(source_id)
                    target_grain = self.manifold.get_grain(target_id)
                    
                    if not source_grain or not target_grain:
                        continue
                    
                    # Check for opposite extreme polarities (circular wraparound)
                    if (hasattr(source_grain, 'polarity') and hasattr(target_grain, 'polarity') and
                        abs(source_grain.polarity) > 0.9 and abs(target_grain.polarity) > 0.9 and
                        source_grain.polarity * target_grain.polarity < 0):
                        # Opposite extreme polarities - potential wraparound
                        polarity_wraparound_events.append({
                            'source_id': source_id,
                            'target_id': target_id,
                            'source_polarity': source_grain.polarity,
                            'target_polarity': target_grain.polarity,
                            'time': event.get('time', self.manifold.time)
                        })
            
            if polarity_wraparound_events:
                observation['polarity_wraparound_events'] = polarity_wraparound_events
                observation['polarity_wraparound_count'] = len(polarity_wraparound_events)
                
                # Update the metrics
                self.metrics['polarity_wraparound_events'] += len(polarity_wraparound_events)
                
                # Store in memory
                self.polarity_wraparound_events.extend(polarity_wraparound_events)
        
        # Observe collapse events in more detail if there are any
        if observation['collapses'] > 0 and hasattr(self.manifold, 'collapse_history'):
            # Get recent collapses
            recent_collapses = self.manifold.collapse_history[-observation['collapses']:]
                
            # Calculate average collapse metrics
            if recent_collapses:
                observation['collapse_metrics'] = {
                    'avg_strength': sum(c.get('strength', 0.0) for c in recent_collapses) / len(recent_collapses),
                    'avg_polarity': sum(c.get('polarity', 0.0) for c in recent_collapses) / len(recent_collapses),
                    'superposition_collapses': sum(1 for c in recent_collapses if c.get('from_superposition', False))
                }
                
                # Observe circular patterns in collapses
                circular_collapse_count = 0
                for collapse in recent_collapses:
                    source_id = collapse.get('source')
                    target_id = collapse.get('target')
                    
                    if not source_id or not target_id:
                        continue
                    
                    source_grain = self.manifold.get_grain(source_id)
                    target_grain = self.manifold.get_grain(target_id)
                    
                    if not source_grain or not target_grain:
                        continue
                    
                    # Check for collapse between opposite extreme polarities
                    if (hasattr(source_grain, 'polarity') and hasattr(target_grain, 'polarity') and
                        abs(source_grain.polarity) > 0.8 and abs(target_grain.polarity) > 0.8 and
                        source_grain.polarity * target_grain.polarity < 0):
                        circular_collapse_count += 1
                
                if circular_collapse_count > 0:
                    observation['collapse_metrics']['circular_collapses'] = circular_collapse_count
        
        # Observe void events
        if observation['voids_formed'] > 0 and hasattr(self.manifold, 'void_formation_events'):
            # Get recent voids
            recent_voids = self.manifold.void_formation_events[-observation['voids_formed']:]
                
            # Calculate average void metrics
            if recent_voids:
                observation['void_metrics'] = {
                    'avg_tension': sum(v.get('tension', 0.0) for v in recent_voids) / len(recent_voids),
                    'avg_strength': sum(v.get('void_strength', 0.0) for v in recent_voids) / len(recent_voids)
                }
        
        # Store in memory
        self.collapse_observations.append(observation)
        
        return observation
    
    def _observe_differential(self, initial_state: Dict[str, Any], 
                            final_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Observe changes between initial and final states.
        
        Args:
            initial_state: Initial state observation
            final_state: Final state observation
            
        Returns:
            Dictionary with differential observation
        """
        # Handle potential missing data with safe defaults
        initial_field = initial_state.get('field_structure', {})
        final_field = final_state.get('field_structure', {})
        
        # Calculate observed changes
        differential = {
            'type': 'differential_observation',
            'time_delta': final_state.get('time', 0.0) - initial_state.get('time', 0.0),
            'real_time_delta': final_state.get('observation_time', 0.0) - initial_state.get('observation_time', 0.0),
            'collapse_delta': final_state.get('collapse_count', 0) - initial_state.get('collapse_count', 0),
            'void_delta': final_state.get('void_count', 0) - initial_state.get('void_count', 0),
            'superposition_delta': final_state.get('superposition_count', 0) - initial_state.get('superposition_count', 0),
        }
        
        # Calculate grain count delta
        if 'grain_count' in initial_field and 'grain_count' in final_field:
            differential['grain_count_delta'] = final_field['grain_count'] - initial_field['grain_count']
        
        # Calculate field structure changes
        field_metrics = [
            ('mean_awareness', 'mean_awareness_delta'),
            ('mean_activation', 'mean_activation_delta'),
            ('mean_saturation', 'mean_saturation_delta'),
            ('mean_polarity', 'mean_polarity_delta'),
            ('field_coherence', 'field_coherence_delta'),
            ('system_tension', 'system_tension_delta'),
            ('circular_recursion_count', 'circular_recursion_delta'),
            ('circular_ancestry_pairs', 'circular_ancestry_delta'),
            ('polarity_circular_coherence', 'polarity_circular_coherence_delta'),
            ('vortex_count', 'vortex_delta')
        ]
        
        for src_key, dest_key in field_metrics:
            if src_key in initial_field and src_key in final_field:
                differential[dest_key] = final_field[src_key] - initial_field[src_key]
        
        # Calculate pathway deltas
        if ('structure_pathways' in initial_field and 'structure_pathways' in final_field and
            'decay_pathways' in initial_field and 'decay_pathways' in final_field):
            differential['pathway_delta'] = {
                'structure': final_field['structure_pathways'] - initial_field['structure_pathways'],
                'decay': final_field['decay_pathways'] - initial_field['decay_pathways']
            }
        
        # Calculate significant changes
        significant_changes = []
        
        # Track significant metric changes
        for src_key, dest_key in field_metrics:
            if dest_key in differential:
                change = differential[dest_key]
                
                # Determine threshold based on metric type
                threshold = 0.2  # Default threshold
                if 'count' in dest_key:
                    # For counts, use absolute change of 2 or more
                    if abs(change) >= 2:
                        direction = 'increase' if change > 0 else 'decrease'
                        significant_changes.append({
                            'type': src_key.replace('_', ' ') + ' change',
                            'metric': src_key,
                            'change': change,
                            'direction': direction,
                            'description': f"Significant {direction} in {src_key.replace('_', ' ')}"
                        })
                else:
                    # For ratios and other metrics, use relative change
                    if abs(change) >= threshold:
                        direction = 'increase' if change > 0 else 'decrease'
                        significant_changes.append({
                            'type': src_key.replace('_', ' ') + ' shift',
                            'metric': src_key,
                            'change': change,
                            'direction': direction,
                            'description': f"Significant {direction} in {src_key.replace('_', ' ')}"
                        })
        
        # Track structure changes
        structure_changes = []
        
        # Check for vortex formation or destruction
        if 'vortex_delta' in differential and differential['vortex_delta'] != 0:
            structure_changes.append({
                'type': 'vortex_change',
                'count': differential['vortex_delta'],
                'direction': 'increase' if differential['vortex_delta'] > 0 else 'decrease',
                'description': f"{'Formation' if differential['vortex_delta'] > 0 else 'Dissolution'} of {abs(differential['vortex_delta'])} vortices"
            })
        
        # Check for pathway formation or destruction
        if 'pathway_delta' in differential:
            structure_pathway_delta = differential['pathway_delta'].get('structure', 0)
            decay_pathway_delta = differential['pathway_delta'].get('decay', 0)
            
            if structure_pathway_delta != 0:
                structure_changes.append({
                    'type': 'structure_pathway_change',
                    'count': structure_pathway_delta,
                    'direction': 'increase' if structure_pathway_delta > 0 else 'decrease',
                    'description': f"{'Formation' if structure_pathway_delta > 0 else 'Dissolution'} of {abs(structure_pathway_delta)} structure pathways"
                })
            
            if decay_pathway_delta != 0:
                structure_changes.append({
                    'type': 'decay_pathway_change',
                    'count': decay_pathway_delta,
                    'direction': 'increase' if decay_pathway_delta > 0 else 'decrease',
                    'description': f"{'Formation' if decay_pathway_delta > 0 else 'Dissolution'} of {abs(decay_pathway_delta)} decay pathways"
                })
        
        # Add changes to differential
        differential['significant_changes'] = significant_changes
        differential['structure_changes'] = structure_changes
        
        return differential
    
    def _observe_circular_recursion(self) -> Dict[str, Any]:
        """
        Observe circular recursion patterns in the manifold.
        
        Returns:
            Dictionary with circular recursion observation
        """
        # Skip if not configured to observe circular patterns
        if not self.config['observe_circular_patterns']:
            return {
                'type': 'circular_recursion_observation',
                'time': self.manifold.time,
                'observation_time': system_time.time(),
                'enabled': False
            }
        
        # Initialize counters
        circular_recursion_count = 0
        circular_ancestry_count = 0
        circular_factors = []
        polarity_extremes_count = 0
        
        # Collect circular recursion grains
        circular_recursion_grains = []
        
        for grain_id, grain in self.manifold.grains.items():
            # Check for circular recursion factor
            has_circular_recursion = False
            circular_factor = 0.0
            
            if hasattr(grain, 'circular_recursion_factor'):
                circular_factor = grain.circular_recursion_factor
                if circular_factor > 0.3:  # Only count significant circular recursion
                    has_circular_recursion = True
                    circular_recursion_count += 1
                    circular_factors.append(circular_factor)
            
            # Check for circular ancestry
            has_circular_ancestry = False
            circular_ancestry_pairs = []
            
            if hasattr(grain, 'circular_ancestry'):
                has_circular_ancestry = len(grain.circular_ancestry) > 0
                circular_ancestry_count += len(grain.circular_ancestry)
                circular_ancestry_pairs = list(grain.circular_ancestry)
            
            # Check for extreme polarity values (near +1 or -1)
            has_extreme_polarity = False
            
            if hasattr(grain, 'polarity') and abs(grain.polarity) > 0.9:
                polarity_extremes_count += 1
                has_extreme_polarity = True
            
            # Only add grains with circular properties
            if has_circular_recursion or has_circular_ancestry or has_extreme_polarity:
                # Get position
                position = {
                    'theta': getattr(grain, 'theta', None),
                    'phi': getattr(grain, 'phi', None)
                }
                
                # Create circular recursion grain entry
                grain_entry = {
                    'grain_id': grain_id,
                    'circular_recursion_factor': circular_factor,
                    'has_circular_ancestry': has_circular_ancestry,
                    'circular_ancestry_pairs': circular_ancestry_pairs,
                    'has_extreme_polarity': has_extreme_polarity,
                    'position': position,
                    'properties': {
                        'awareness': getattr(grain, 'awareness', 0.0),
                        'polarity': getattr(grain, 'polarity', 0.0),
                        'saturation': getattr(grain, 'grain_saturation', 0.0),
                        'activation': getattr(grain, 'grain_activation', 0.0),
                        'is_self_recursive': grain_id in getattr(grain, 'ancestry', set())
                    }
                }
                
                circular_recursion_grains.append(grain_entry)
        
        # Store in memory
        self.circular_recursion_grains = set(g['grain_id'] for g in circular_recursion_grains)
        
        # Collect polarity wraparound paths
        polarity_wraparound_paths = []
        
        # Check collapse history for polarity wraparound events
        if hasattr(self.manifold, 'collapse_history'):
            # Look at recent collapses (last 20)
            recent_collapses = self.manifold.collapse_history[-min(20, len(self.manifold.collapse_history)):]
            
            for collapse in recent_collapses:
                source_id = collapse.get('source')
                target_id = collapse.get('target')
                
                if not source_id or not target_id:
                    continue
                    
                source_grain = self.manifold.get_grain(source_id)
                target_grain = self.manifold.get_grain(target_id)
                
                if not source_grain or not target_grain:
                    continue
                    
                # Check for opposite extreme polarities
                if (hasattr(source_grain, 'polarity') and hasattr(target_grain, 'polarity') and
                    abs(source_grain.polarity) > 0.8 and abs(target_grain.polarity) > 0.8 and
                    source_grain.polarity * target_grain.polarity < 0):
                    
                    # Create path
                    path = {
                        'source_id': source_id,
                        'target_id': target_id,
                        'source_polarity': source_grain.polarity,
                        'target_polarity': target_grain.polarity,
                        'time': collapse.get('time', self.manifold.time),
                        'polarity_product': source_grain.polarity * target_grain.polarity
                    }
                    
                    polarity_wraparound_paths.append(path)
        
        # Store in memory
        self.polarity_wraparound_events = polarity_wraparound_paths
        
        # Calculate circular coherence if there are enough grains
        if len(self.manifold.grains) >= 2:
            # Get polarity values
            polarity_values = [getattr(grain, 'polarity', 0.0) for grain in self.manifold.grains.values()]
            
            # Map polarities to angles on the circle [-1,1] → [0,2π]
            polarity_angles = [(p + 1) * math.pi for p in polarity_values]
            
            # Calculate circular coherence
            circular_coherence = self._calculate_circular_coherence(polarity_angles)
            
            # Update metric
            self.metrics['circular_coherence'] = circular_coherence
        else:
            circular_coherence = 0.5  # Default for too few grains
            
        # Update metrics
        self.metrics['circular_recursion_count'] = circular_recursion_count
        self.metrics['circular_ancestry_count'] = circular_ancestry_count
        self.metrics['polarity_wraparound_events'] = len(polarity_wraparound_paths)
        
        # Calculate average circular recursion factor
        avg_circular_factor = sum(circular_factors) / len(circular_factors) if circular_factors else 0.0
        
        # Calculate circular polarity metrics
        polar_extremes_ratio = polarity_extremes_count / len(self.manifold.grains) if self.manifold.grains else 0.0
        
        # Build observation
        observation = {
            'type': 'circular_recursion_observation',
            'time': self.manifold.time,
            'observation_time': system_time.time(),
            'circular_recursion_count': circular_recursion_count,
            'circular_ancestry_count': circular_ancestry_count,
            'polarity_wraparound_count': len(polarity_wraparound_paths),
            'circular_recursion_grains': circular_recursion_grains,
            'polarity_wraparound_paths': polarity_wraparound_paths,
            'circular_coherence': circular_coherence,
            'avg_circular_factor': avg_circular_factor,
            'polar_extremes_count': polarity_extremes_count,
            'polar_extremes_ratio': polar_extremes_ratio,
            'enabled': True
        }
        
        # Store in memory
        self.circular_observations.append(observation)
        
        return observation
    
    def _observe_ancestry_patterns(self) -> Dict[str, Any]:
        """
        Observe ancestry patterns in the manifold.
        
        Returns:
            Dictionary with ancestry observation
        """
        # Skip if not configured to observe ancestry
        if not self.config['observe_ancestry']:
            return {
                'type': 'ancestry_observation',
                'time': self.manifold.time,
                'observation_time': system_time.time(),
                'enabled': False
            }
        
        # Calculate ancestry distribution
        ancestry_counts = defaultdict(int)
        self_referential_count = 0
        
        for grain_id, grain in self.manifold.grains.items():
            ancestry = getattr(grain, 'ancestry', set())
            ancestry_size = len(ancestry)
            ancestry_counts[ancestry_size] += 1
            
            # Count self-referential grains
            if grain_id in ancestry:
                self_referential_count += 1
        
        # Calculate average ancestry depth
        total_grains = len(self.manifold.grains)
        if total_grains > 0:
            total_ancestry = sum(size * count for size, count in ancestry_counts.items())
            average_ancestry_size = total_ancestry / total_grains
        else:
            average_ancestry_size = 0.0
        
        # Find self-referential grains
        self_referential_grains = []
        
        for grain_id, grain in self.manifold.grains.items():
            ancestry = getattr(grain, 'ancestry', set())
            
            if grain_id in ancestry:
                # Get position and properties
                position = {
                    'theta': getattr(grain, 'theta', None),
                    'phi': getattr(grain, 'phi', None)
                }
                
                # Create self-referential grain entry
                grain_entry = {
                    'grain_id': grain_id,
                    'ancestry_size': len(ancestry),
                    'position': position,
                    'properties': {
                        'awareness': getattr(grain, 'awareness', 0.0),
                        'polarity': getattr(grain, 'polarity', 0.0),
                        'saturation': getattr(grain, 'grain_saturation', 0.0),
                        'activation': getattr(grain, 'grain_activation', 0.0)
                    }
                }
                
                self_referential_grains.append(grain_entry)
        
        # Store in memory
        self.self_referential_grains = set(g['grain_id'] for g in self_referential_grains)
        
        # Calculate shared ancestry patterns
        shared_ancestry_patterns = {}
        
        # Create pairs of grains and calculate shared ancestry
        grain_ids = list(self.manifold.grains.keys())
        for i in range(len(grain_ids)):
            for j in range(i+1, len(grain_ids)):
                grain1_id = grain_ids[i]
                grain2_id = grain_ids[j]
                
                grain1 = self.manifold.get_grain(grain1_id)
                grain2 = self.manifold.get_grain(grain2_id)
                
                if not grain1 or not grain2:
                    continue
                    
                ancestry1 = getattr(grain1, 'ancestry', set())
                ancestry2 = getattr(grain2, 'ancestry', set())
                
                # Calculate shared ancestry
                shared = ancestry1.intersection(ancestry2)
                
                # Only record significant shared ancestry
                if len(shared) >= 2:
                    shared_ancestry_patterns[(grain1_id, grain2_id)] = {
                        'shared_count': len(shared),
                        'shared_ids': list(shared)
                    }
        
        # Update metrics
        self.metrics['ancestry_depth'] = average_ancestry_size
        self.metrics['self_referential_count'] = self_referential_count
        
        # Build observation
        observation = {
            'type': 'ancestry_observation',
            'time': self.manifold.time,
            'observation_time': system_time.time(),
            'self_referential_count': self_referential_count,
            'ancestry_depth': average_ancestry_size,
            'ancestry_distribution': dict(ancestry_counts),
            'self_referential_grains': self_referential_grains,
            'shared_ancestry_patterns': shared_ancestry_patterns,
            'enabled': True
        }
        
        # Store in memory
        self.ancestry_observations.append(observation)
        
        return observation
    
    def _observe_absence_patterns(self) -> Dict[str, Any]:
        """
        Observe absence patterns in the manifold (void formations, decay patterns).
        
        Returns:
            Dictionary with absence pattern observation
        """
        # Skip if not configured to track absence patterns
        if not self.config['track_absence_patterns']:
            return {
                'type': 'absence_observation',
                'time': self.manifold.time,
                'observation_time': system_time.time(),
                'enabled': False
            }
        
        # Initialize counters
        void_formations = 0
        decay_cascades = 0
        tensile_boundary_count = 0
        
        # Collect void formation events
        void_formation_events = []
        
        if hasattr(self.manifold, 'void_formation_events'):
            # Count recent void formations (since last observation)
            last_observation_time = self.metrics['last_observation_time']
            
            for event in self.manifold.void_formation_events:
                if event.get('time', 0.0) > last_observation_time:
                    void_formations += 1
                    
                    # Add to events list
                    void_formation_events.append(event)
        
        # Find tensile boundaries (grains with high tension at boundaries between structure and void)
        tensile_boundaries = []
        
        for grain_id, grain in self.manifold.grains.items():
            # Skip grains with low tension
            tension = 0.0
            if hasattr(grain, 'structural_tension'):
                tension = grain.structural_tension
            elif hasattr(grain, 'constraint_tension'):
                tension = grain.constraint_tension
                
            if tension < 0.7:
                continue
                
            # Check for boundary condition (connection to both structure and void)
            if not hasattr(grain, 'relations'):
                continue
                
            # Count structure and void connections
            structure_connections = 0
            void_connections = 0
            
            for related_id, relation_strength in grain.relations.items():
                related_grain = self.manifold.get_grain(related_id)
                if not related_grain:
                    continue
                    
                # Check polarity to determine structure or void
                if hasattr(related_grain, 'polarity'):
                    if related_grain.polarity > 0.5:
                        structure_connections += 1
                    elif related_grain.polarity < -0.5:
                        void_connections += 1
            
            # Check if this is a boundary grain (has both structure and void connections)
            if structure_connections > 0 and void_connections > 0:
                # Get position and properties
                position = {
                    'theta': getattr(grain, 'theta', None),
                    'phi': getattr(grain, 'phi', None)
                }
                
                # Create tensile boundary entry
                boundary = {
                    'grain_id': grain_id,
                    'tension': tension,
                    'structure_connections': structure_connections,
                    'void_connections': void_connections,
                    'position': position,
                    'properties': {
                        'awareness': getattr(grain, 'awareness', 0.0),
                        'polarity': getattr(grain, 'polarity', 0.0),
                        'saturation': getattr(grain, 'grain_saturation', 0.0),
                        'activation': getattr(grain, 'grain_activation', 0.0)
                    }
                }
                
                tensile_boundaries.append(boundary)
                tensile_boundary_count += 1
        
        # Store in memory
        self.void_formations.extend(void_formation_events)
        self.tensile_boundaries = tensile_boundaries
        
        # Update metrics
        absence_count = void_formations + decay_cascades + tensile_boundary_count
        
        # Build observation
        observation = {
            'type': 'absence_observation',
            'time': self.manifold.time,
            'observation_time': system_time.time(),
            'void_formations': void_formations,
            'decay_cascades': decay_cascades,
            'tensile_boundary_count': tensile_boundary_count,
            'void_formation_events': void_formation_events,
            'tensile_boundaries': tensile_boundaries,
            'absence_count': absence_count,
            'enabled': True
        }
        
        # Store in memory
        self.absence_observations.append(observation)
        
        return observation
    
    def _update_metrics(self, manifestation_observation: Dict[str, Any],
                      differential: Dict[str, Any],
                      circular_observation: Dict[str, Any],
                      ancestry_observation: Dict[str, Any],
                      absence_observation: Dict[str, Any]):
        """
        Update metrics based on observations.
        
        Args:
            manifestation_observation: Manifestation observation
            differential: Differential observation
            circular_observation: Circular recursion observation
            ancestry_observation: Ancestry pattern observation
            absence_observation: Absence pattern observation
        """
        # Update from manifestation
        self.metrics['observed_collapses'] += manifestation_observation.get('collapses', 0)
        self.metrics['observed_voids'] += manifestation_observation.get('voids_formed', 0)
        self.metrics['observed_coherence'] = manifestation_observation.get('coherence', self.metrics['observed_coherence'])
        self.metrics['observed_field_tension'] = manifestation_observation.get('tension', self.metrics['observed_field_tension'])
        self.metrics['observed_vortices'] += manifestation_observation.get('vortices', 0)
        
        # Update pathway counts
        pathway_counts = manifestation_observation.get('pathways', {})
        self.metrics['observed_pathways']['structure'] += pathway_counts.get('structure', 0)
        self.metrics['observed_pathways']['decay'] += pathway_counts.get('decay', 0)
        
        # Update from circular observation
        self.metrics['circular_recursion_count'] = circular_observation.get('circular_recursion_count', self.metrics['circular_recursion_count'])
        self.metrics['circular_ancestry_count'] = circular_observation.get('circular_ancestry_count', self.metrics['circular_ancestry_count'])
        self.metrics['polarity_wraparound_events'] = circular_observation.get('polarity_wraparound_count', self.metrics['polarity_wraparound_events'])
        self.metrics['circular_coherence'] = circular_observation.get('circular_coherence', self.metrics['circular_coherence'])
        
        # Update from ancestry observation
        self.metrics['self_referential_count'] = ancestry_observation.get('self_referential_count', self.metrics['self_referential_count'])
        self.metrics['ancestry_depth'] = ancestry_observation.get('ancestry_depth', self.metrics['ancestry_depth'])
        
        # Update from absence observation if applicable
        if absence_observation.get('enabled', False):
            self.metrics['observed_voids'] += absence_observation.get('void_formations', 0)
        
        # Update from differential if applicable
        field_structure = manifestation_observation.get('field_structure', {})
        if 'lightlike_ratio' in field_structure:
            self.metrics['lightlike_ratio'] = field_structure['lightlike_ratio']
    
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