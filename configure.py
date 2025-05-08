"""
Configure - Configuration system for Collapse Geometry framework

This module provides a unified configuration interface for the Collapse Geometry
framework, allowing easy modification of parameters across various components.
It uses a hierarchical structure to organize settings for different aspects
of the simulation.
"""

import json
import os
from typing import Dict, Any, Optional, List, Union, Set
import dataclasses
from dataclasses import dataclass, field


@dataclass
class CollapseGeometryConfig:
    """
    Configuration class for the Collapse Geometry framework.
    Contains settings for all framework components.
    """
    
    # Manifold configuration
    class Manifold:
        @dataclass
        class FieldConfig:
            """Field dynamics parameters"""
            field_diffusion_rate: float = 0.15        # Controls awareness propagation
            field_gradient_sensitivity: float = 0.25  # Sensitivity to field gradients
            activation_threshold: float = 0.5         # Threshold for grain activation
            
        @dataclass
        class VoidDecayConfig:
            """Void and decay mechanism settings"""
            alignment_threshold: float = 0.7          # Threshold for successful alignment
            void_formation_threshold: float = 0.8     # Tension threshold for void formation
            decay_emission_rate: float = 0.2          # Base rate of decay particle emission
            void_propagation_rate: float = 0.1        # How quickly voids spread
            check_structural_alignment: bool = True   # Whether to enforce structural alignment
            decay_impact_factor: float = 0.3          # How strongly decay affects the system
        
        @dataclass
        class ToroidalConfig:
            """Toroidal system settings"""
            neighborhood_radius: float = 0.5          # Radius for toroidal neighborhood detection
            phase_stability_decay: float = 0.05       # Natural decay rate of phase stability
            coherence_threshold: float = 0.7          # Threshold for phase coherence events
            toroidal_flow_weight: float = 0.2         # Weight for toroidal flow effects
        
        field: FieldConfig = field(default_factory=FieldConfig)
        void_decay: VoidDecayConfig = field(default_factory=VoidDecayConfig)
        toroidal: ToroidalConfig = field(default_factory=ToroidalConfig)
    
    # Engine configuration
    class Engine:
        @dataclass
        class ObservationConfig:
            """Engine observation settings"""
            observation_detail: float = 0.8           # How detailed observations should be
            awareness_sensitivity: float = 0.7        # Sensitivity to awareness patterns
            resonance_sensitivity: float = 0.6        # Sensitivity to resonance patterns
            causal_trace_depth: int = 5               # How many causal steps to track
            ancestry_depth: int = 3                   # How deep to track ancestry
            structure_detection: float = 0.7          # Detection level for emergent structure
            symmetry_detection: float = 0.6           # Sensitivity to symmetry breaking
        
        @dataclass
        class ProcessingConfig:
            """Event processing settings"""
            max_events_per_step: int = 10             # Maximum events to process in one step
            natural_emergence_weight: float = 0.8     # How much to prioritize natural emergence
            distributed_processing: bool = True       # Whether to use distributed processing
            process_void_decay: bool = True           # Whether to process void-decay effects
        
        observation: ObservationConfig = field(default_factory=ObservationConfig)
        processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    # Visualization configuration
    class Visualizer:
        @dataclass
        class TorusConfig:
            """3D torus visualization settings"""
            major_radius: float = 3.0                 # Major radius of the torus
            minor_radius: float = 1.0                 # Minor radius of the torus 
            resolution: int = 36                      # Resolution of the torus surface
            field_smoothing: float = 1.5              # Smoothing factor for field visualization
            wave_complexity: int = 4                  # Number of waves in the field pattern
            symmetry_factor: int = 4                  # Controls symmetry in field pattern
            
        @dataclass
        class UnwrappingConfig:
            """2D unwrapping visualization settings"""
            field_resolution: int = 100               # Resolution of field grid
            vector_density: int = 15                  # Density of vector field visualization
            field_alpha: float = 0.7                  # Transparency of field visualization
            show_relations: bool = True               # Whether to show relations between grains
            show_vortices: bool = True                # Whether to show vortices
            show_void_regions: bool = True            # Whether to show void regions
        
        torus_3d: TorusConfig = field(default_factory=TorusConfig)
        unwrapping: UnwrappingConfig = field(default_factory=UnwrappingConfig)
    
    # State tracking configuration
    class StateTracker:
        @dataclass
        class MetricsConfig:
            """Metrics collection settings"""
            track_toroidal_metrics: bool = True       # Whether to track toroidal metrics
            track_void_decay: bool = True             # Whether to track void-decay metrics
            global_metrics: List[str] = field(default_factory=lambda: [
                'awareness', 'collapse', 'activation', 'saturation', 
                'resonance', 'tension', 'coherence', 'stability'
            ])
            
        @dataclass
        class StructuresConfig:
            """Structure tracking settings"""
            track_vortices: bool = True               # Whether to track vortex structures
            track_phase_domains: bool = True          # Whether to track phase domains
            track_void_regions: bool = True           # Whether to track void regions
            track_attractors: bool = True             # Whether to track attractor states
            
        @dataclass
        class EventsConfig:
            """Event tracking settings"""
            track_collapse_events: bool = True        # Whether to track collapse events
            track_void_formations: bool = True        # Whether to track void formations
            track_decay_emissions: bool = True        # Whether to track decay emissions
            track_phase_transitions: bool = True      # Whether to track phase transitions
            max_event_history: int = 1000             # Maximum number of events to store
            
        metrics: MetricsConfig = field(default_factory=MetricsConfig)
        structures: StructuresConfig = field(default_factory=StructuresConfig)
        events: EventsConfig = field(default_factory=EventsConfig)
        
    # Main configuration fields
    manifold: Manifold = field(default_factory=Manifold)
    engine: Engine = field(default_factory=Engine)
    visualizer: Visualizer = field(default_factory=Visualizer)
    state_tracker: StateTracker = field(default_factory=StateTracker)
    
    # Additional global configuration options
    use_fixed_seed: bool = False
    random_seed: int = 42
    base_directory: str = os.path.expanduser("~/axiom8_data")
    logging_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary"""
        result = {}
        
        # Helper function to convert a dataclass to a dictionary
        def _process_dataclass(obj):
            if dataclasses.is_dataclass(obj):
                return {f.name: _process_dataclass(getattr(obj, f.name)) 
                        for f in dataclasses.fields(obj)}
            elif isinstance(obj, list):
                return [_process_dataclass(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return {k: _process_dataclass(v) for k, v in obj.__dict__.items() 
                        if not k.startswith('_')}
            else:
                return obj
                
        # Process each main section
        result['manifold'] = _process_dataclass(self.manifold)
        result['engine'] = _process_dataclass(self.engine)
        result['visualizer'] = _process_dataclass(self.visualizer)
        result['state_tracker'] = _process_dataclass(self.state_tracker)
        
        # Add global options
        result['use_fixed_seed'] = self.use_fixed_seed
        result['random_seed'] = self.random_seed
        result['base_directory'] = self.base_directory
        result['logging_level'] = self.logging_level
        
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CollapseGeometryConfig':
        """Create a configuration from a dictionary"""
        config = cls()
        
        # Helper function to update a dataclass from a dictionary
        def _update_dataclass(obj, data):
            if dataclasses.is_dataclass(obj):
                for f in dataclasses.fields(obj):
                    if f.name in data:
                        field_value = getattr(obj, f.name)
                        setattr(obj, f.name, _update_dataclass(field_value, data[f.name]))
                return obj
            elif hasattr(obj, '__dict__') and isinstance(data, dict):
                for k, v in data.items():
                    if hasattr(obj, k):
                        attr_value = getattr(obj, k)
                        setattr(obj, k, _update_dataclass(attr_value, v))
                return obj
            elif isinstance(data, dict) and not hasattr(obj, '__dict__'):
                # For nested dataclasses that aren't initialized yet
                return data
            else:
                return data
        
        # Update each main section
        if 'manifold' in config_dict:
            config.manifold = _update_dataclass(config.manifold, config_dict['manifold'])
        if 'engine' in config_dict:
            config.engine = _update_dataclass(config.engine, config_dict['engine'])
        if 'visualizer' in config_dict:
            config.visualizer = _update_dataclass(config.visualizer, config_dict['visualizer'])
        if 'state_tracker' in config_dict:
            config.state_tracker = _update_dataclass(config.state_tracker, config_dict['state_tracker'])
        
        # Update global options
        if 'use_fixed_seed' in config_dict:
            config.use_fixed_seed = config_dict['use_fixed_seed']
        if 'random_seed' in config_dict:
            config.random_seed = config_dict['random_seed']
        if 'base_directory' in config_dict:
            config.base_directory = config_dict['base_directory']
        if 'logging_level' in config_dict:
            config.logging_level = config_dict['logging_level']
        
        return config
    
    def set_config(self, other: 'CollapseGeometryConfig'):
        """Copy configuration from another CollapseGeometryConfig instance"""
        self.__dict__.update(other.__dict__)
    
    def save_to_file(self, filepath: str):
        """Save the configuration to a JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'CollapseGeometryConfig':
        """Load configuration from a JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Global config instance
_CONFIG = CollapseGeometryConfig()


def get_config() -> CollapseGeometryConfig:
    """Get the global configuration instance"""
    return _CONFIG


def update_config(**kwargs):
    """
    Update the global configuration with the provided values.
    
    Args:
        **kwargs: Config values as nested dictionaries
    """
    config_dict = _CONFIG.to_dict()
    
    # Helper function to update nested dictionaries
    def update_nested_dict(d, path, value):
        if len(path) == 1:
            d[path[0]] = value
        else:
            if path[0] not in d:
                d[path[0]] = {}
            update_nested_dict(d[path[0]], path[1:], value)
    
    # Process each config path
    for key, value in kwargs.items():
        if isinstance(value, dict):
            # Process nested dictionary
            for subkey, subvalue in _flatten_dict(value).items():
                path = [key] + subkey.split('.')
                update_nested_dict(config_dict, path, subvalue)
        else:
            # Direct update
            config_dict[key] = value
    
    # Update the global config
    global _CONFIG
    _CONFIG = CollapseGeometryConfig.from_dict(config_dict)


def _flatten_dict(d, parent_key='', sep='.'):
    """Flatten a nested dictionary, joining keys with sep"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_config(filepath):
    """
    Load configuration from a JSON file.
    
    Args:
        filepath: Path to the configuration file
    """
    global _CONFIG
    _CONFIG = CollapseGeometryConfig.load_from_file(filepath)


def save_config(filepath):
    """
    Save the current configuration to a JSON file.
    
    Args:
        filepath: Path to save the configuration file
    """
    _CONFIG.save_to_file(filepath)


def get_predefined_config(preset_name):
    """
    Get a predefined configuration preset.
    
    Args:
        preset_name: Name of the configuration preset
        
    Returns:
        CollapseGeometryConfig instance
    """
    presets = {
        'default': CollapseGeometryConfig(),
        
        'high_resolution': CollapseGeometryConfig(
            visualizer=CollapseGeometryConfig.Visualizer(
                torus_3d=CollapseGeometryConfig.Visualizer.TorusConfig(
                    resolution=100
                ),
                unwrapping=CollapseGeometryConfig.Visualizer.UnwrappingConfig(
                    field_resolution=200,
                    vector_density=30
                )
            )
        ),
        
        'high_dynamics': CollapseGeometryConfig(
            manifold=CollapseGeometryConfig.Manifold(
                field=CollapseGeometryConfig.Manifold.FieldConfig(
                    field_diffusion_rate=0.25,
                    field_gradient_sensitivity=0.35,
                    activation_threshold=0.4
                ),
                toroidal=CollapseGeometryConfig.Manifold.ToroidalConfig(
                    toroidal_flow_weight=0.35,
                    phase_stability_decay=0.08
                )
            ),
            engine=CollapseGeometryConfig.Engine(
                processing=CollapseGeometryConfig.Engine.ProcessingConfig(
                    max_events_per_step=20,
                    natural_emergence_weight=0.9
                )
            )
        ),
        
        'stable_structures': CollapseGeometryConfig(
            manifold=CollapseGeometryConfig.Manifold(
                field=CollapseGeometryConfig.Manifold.FieldConfig(
                    field_diffusion_rate=0.1,
                    field_gradient_sensitivity=0.2,
                    activation_threshold=0.6
                ),
                toroidal=CollapseGeometryConfig.Manifold.ToroidalConfig(
                    phase_stability_decay=0.02,
                    coherence_threshold=0.8
                )
            ),
            engine=CollapseGeometryConfig.Engine(
                observation=CollapseGeometryConfig.Engine.ObservationConfig(
                    structure_detection=0.85,
                    symmetry_detection=0.8
                )
            )
        ),
        
        'void_decay_focus': CollapseGeometryConfig(
            manifold=CollapseGeometryConfig.Manifold(
                void_decay=CollapseGeometryConfig.Manifold.VoidDecayConfig(
                    alignment_threshold=0.65,
                    void_formation_threshold=0.7,
                    decay_emission_rate=0.3,
                    void_propagation_rate=0.15,
                    decay_impact_factor=0.4
                )
            ),
            state_tracker=CollapseGeometryConfig.StateTracker(
                events=CollapseGeometryConfig.StateTracker.EventsConfig(
                    track_void_formations=True,
                    track_decay_emissions=True,
                    max_event_history=2000
                )
            )
        )
    }
    
    return presets.get(preset_name, presets['default'])


def initialize_with_config(obj):
    """
    Initialize an object with the current configuration.
    This applies the relevant configuration sections to the object.
    
    Args:
        obj: Object to initialize
    """
    # Get object type to determine which config section to use
    obj_type = type(obj).__name__
    config_dict = _CONFIG.to_dict()
    
    # Apply configuration based on object type
    if obj_type == 'RelationalManifold':
        _apply_manifold_config(obj, config_dict.get('manifold', {}))
    elif obj_type in ('EmergentDualityEngine', 'EnhancedContinuousDualityEngine'):
        _apply_engine_config(obj, config_dict.get('engine', {}))
    elif obj_type == 'SimulationState':
        _apply_state_config(obj, config_dict.get('state_tracker', {}))
    elif obj_type == 'TorusSimulationVisualizer':
        _apply_torus_visualizer_config(obj, config_dict.get('visualizer', {}).get('torus_3d', {}))
    elif obj_type == 'TorusUnwrappingVisualizer':
        _apply_unwrapping_visualizer_config(obj, config_dict.get('visualizer', {}).get('unwrapping', {}))
    
    # Apply global config options when relevant
    if hasattr(obj, 'random_seed') and _CONFIG.use_fixed_seed:
        obj.random_seed = _CONFIG.random_seed


def _apply_manifold_config(manifold, config):
    """Apply configuration to a RelationalManifold instance"""
    # Apply field settings
    if 'field' in config:
        field_config = config['field']
        if hasattr(manifold, 'field_diffusion_rate'):
            manifold.field_diffusion_rate = field_config.get('field_diffusion_rate', manifold.field_diffusion_rate)
        if hasattr(manifold, 'field_gradient_sensitivity'):
            manifold.field_gradient_sensitivity = field_config.get('field_gradient_sensitivity', manifold.field_gradient_sensitivity)
        if hasattr(manifold, 'activation_threshold'):
            manifold.activation_threshold = field_config.get('activation_threshold', manifold.activation_threshold)
    
    # Apply void-decay settings
    if 'void_decay' in config and hasattr(manifold, 'void_decay_config'):
        void_config = config['void_decay']
        for key, value in void_config.items():
            if key in manifold.void_decay_config:
                manifold.void_decay_config[key] = value
    
    # Apply toroidal settings
    if 'toroidal' in config:
        toroidal_config = config['toroidal']
        if hasattr(manifold, 'toroidal_system') and hasattr(manifold.toroidal_system, 'neighborhood_radius'):
            manifold.toroidal_system.neighborhood_radius = toroidal_config.get('neighborhood_radius', 
                                                                            manifold.toroidal_system.neighborhood_radius)


def _apply_engine_config(engine, config):
    """Apply configuration to an EmergentDualityEngine instance"""
    # Apply observation settings
    if 'observation' in config:
        obs_config = config['observation']
        if hasattr(engine, 'config'):
            for key, value in obs_config.items():
                if key in engine.config:
                    engine.config[key] = value
    
    # Apply processing settings
    if 'processing' in config:
        proc_config = config['processing']
        if hasattr(engine, 'process_void_decay'):
            engine.process_void_decay = proc_config.get('process_void_decay', True)
        if hasattr(engine, 'max_events_per_step'):
            engine.max_events_per_step = proc_config.get('max_events_per_step', 10)
        if hasattr(engine, 'distributed_processing'):
            engine.distributed_processing = proc_config.get('distributed_processing', True)


def _apply_state_config(state, config):
    """Apply configuration to a SimulationState instance"""
    # Apply metrics settings
    if 'metrics' in config and hasattr(state, 'metrics'):
        metrics_config = config['metrics']
        if 'global_metrics' in metrics_config:
            # Initialize metrics if they don't exist
            for metric in metrics_config['global_metrics']:
                if metric not in state.metrics:
                    state.metrics[metric] = 0.0
    
    # Apply events settings
    if 'events' in config and hasattr(state, 'history'):
        events_config = config['events']
        max_events = events_config.get('max_event_history', 1000)
        # Initialize event arrays with specified capacity
        if hasattr(state, 'void_formation_history'):
            state.void_formation_history = state.void_formation_history[:max_events]
        if hasattr(state, 'decay_emission_history'):
            state.decay_emission_history = state.decay_emission_history[:max_events]


def _apply_torus_visualizer_config(visualizer, config):
    """Apply configuration to a TorusSimulationVisualizer instance"""
    if hasattr(visualizer, 'major_radius'):
        visualizer.major_radius = config.get('major_radius', visualizer.major_radius)
    if hasattr(visualizer, 'minor_radius'):
        visualizer.minor_radius = config.get('minor_radius', visualizer.minor_radius)
    if hasattr(visualizer, 'resolution'):
        visualizer.resolution = config.get('resolution', visualizer.resolution)
    if hasattr(visualizer, 'field_smoothing'):
        visualizer.field_smoothing = config.get('field_smoothing', visualizer.field_smoothing)
    if hasattr(visualizer, 'wave_complexity'):
        visualizer.wave_complexity = config.get('wave_complexity', visualizer.wave_complexity)
    if hasattr(visualizer, 'symmetry_factor'):
        visualizer.symmetry_factor = config.get('symmetry_factor', visualizer.symmetry_factor)


def _apply_unwrapping_visualizer_config(visualizer, config):
    """Apply configuration to a TorusUnwrappingVisualizer instance"""
    if hasattr(visualizer, 'field_resolution'):
        visualizer.field_resolution = config.get('field_resolution', visualizer.field_resolution)
    if hasattr(visualizer, 'vector_density'):
        visualizer.vector_density = config.get('vector_density', visualizer.vector_density)
    if hasattr(visualizer, 'field_alpha'):
        visualizer.field_alpha = config.get('field_alpha', visualizer.field_alpha)
    if hasattr(visualizer, 'show_relations'):
        visualizer.show_relations = config.get('show_relations', visualizer.show_relations)
    if hasattr(visualizer, 'show_vortices'):
        visualizer.show_vortices = config.get('show_vortices', visualizer.show_vortices)
    if hasattr(visualizer, 'show_void_regions'):
        visualizer.show_void_regions = config.get('show_void_regions', visualizer.show_void_regions)