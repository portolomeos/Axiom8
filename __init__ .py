"""
Axiom8 - Collapse Geometry Simulation Framework

A relational physics simulation framework where everything is relational:
- Movement, structure, behavior emerge from local interactions, not external rules
- Collapse is the only source of committed structure
- Awareness fields evolve through relational tension, not imposed trajectories
- Polarity bias represents directional memory, not force
- Ancestry defines entanglement, not spatial distance

Enhanced with:
- Void-Decay principle for handling incompatible structures
- Toroidal referencing for proper topological dynamics
"""

# Package version
__version__ = '0.4.0'

# Import configuration module
from axiom8.configure import (
    get_config,
    update_config,
    load_config,
    save_config,
    get_predefined_config,
    initialize_with_config,
    CollapseGeometryConfig
)

# Expose functions to initialize components without direct imports
# This avoids circular dependencies by moving imports into functions

def create_relational_manifold():
    """Create a new relational manifold instance"""
    from axiom8.core.relational_manifold import RelationalManifold
    manifold = RelationalManifold()
    # Initialize with configuration
    initialize_with_config(manifold)
    return manifold

def create_simulation_engine(manifold=None, state=None, config=None):
    """Create a new simulation engine instance"""
    from axiom8.core.engine import EmergentDualityEngine
    if manifold is None:
        manifold = create_relational_manifold()
    if state is None:
        state = create_simulation_state()
    
    # Create engine with config
    if config is None:
        config = get_config().to_dict()
    elif isinstance(config, CollapseGeometryConfig):
        config = config.to_dict()
        
    engine = EnhancedContinuousDualityEngine(manifold, state, config)
    
    # Initialize with configuration
    initialize_with_config(engine)
    return engine

def create_simulation_state():
    """Create a new simulation state instance"""
    from axiom8.core.state import SimulationState
    state = SimulationState()
    # Initialize with configuration
    initialize_with_config(state)
    return state

def create_3d_torus_visualizer():
    """Create a 3D torus simulation visualizer"""
    try:
        from axiom8.visualizer.torus_simulation_visualizer import TorusSimulationVisualizer
        visualizer = TorusSimulationVisualizer()
        # Initialize with configuration
        initialize_with_config(visualizer)
        return visualizer
    except ImportError:
        print("Warning: TorusSimulationVisualizer not available. Visualization components may be missing.")
        return None

def create_torus_unwrapping_visualizer():
    """Create a 2D torus unwrapping visualizer"""
    try:
        from axiom8.visualizer.torus_unwrapping_visualizer import TorusUnwrappingVisualizer
        visualizer = TorusUnwrappingVisualizer()
        # Initialize with configuration
        initialize_with_config(visualizer)
        return visualizer
    except ImportError:
        print("Warning: TorusUnwrappingVisualizer not available. Visualization components may be missing.")
        return None

# Create a complete simulation environment
def create_simulation(config=None):
    """
    Create a complete simulation environment with manifold, engine, and state.
    
    Args:
        config: Optional configuration dictionary or CollapseGeometryConfig instance
        
    Returns:
        tuple: (manifold, engine, state)
    """
    # Apply configuration if provided
    if config is not None:
        if isinstance(config, dict):
            update_config(**config)
        elif isinstance(config, CollapseGeometryConfig):
            get_config().set_config(config)
    
    # Create components
    manifold = create_relational_manifold()
    state = create_simulation_state()
    engine = create_simulation_engine(manifold, state)
    
    return manifold, engine, state

# Helper function to seed a manifold with initial grains
def seed_manifold(manifold, num_grains=10, connectivity=0.3, create_opposites=True):
    """
    Seed a manifold with initial grains and connections.
    
    Args:
        manifold: RelationalManifold to seed
        num_grains: Number of initial grains to create
        connectivity: Probability of connection between grains
        create_opposites: Whether to create opposite grain pairs
        
    Returns:
        list: IDs of created grains
    """
    import random
    import math
    
    grain_ids = []
    
    # Create grains
    for i in range(num_grains):
        # Create grain with toroidal coordinates
        theta = random.random() * 2 * math.pi
        phi = random.random() * 2 * math.pi
        
        grain = manifold.add_grain(f"grain_{i}", theta=theta, phi=phi)
        grain_ids.append(grain.id)
        
        # Set initial awareness
        grain.awareness = random.uniform(0.1, 0.5)
    
    # Create connections
    for i, grain1_id in enumerate(grain_ids):
        for j, grain2_id in enumerate(grain_ids[i+1:], i+1):
            if random.random() < connectivity:
                # Create bidirectional relation
                manifold.connect_grains(grain1_id, grain2_id, relation_strength=random.uniform(0.3, 0.8))
    
    # Create opposite pairs (if requested)
    if create_opposites and len(grain_ids) >= 2:
        pairs_to_create = min(len(grain_ids) // 4, 2)  # Create a few opposite pairs
        
        for _ in range(pairs_to_create):
            i, j = random.sample(range(len(grain_ids)), 2)
            manifold.set_opposite_grains(grain_ids[i], grain_ids[j])
    
    return grain_ids

# Fix import in config_space.py
def _fix_axiom7_imports():
    """
    Fix axiom7 imports in config_space.py by redirecting them to axiom8.
    This is an internal helper function to fix compatibility issues.
    """
    try:
        import sys
        import types
        
        # Create a fake axiom7 module that redirects to axiom8
        axiom7_module = types.ModuleType('axiom7')
        
        # Redirect axiom7.collapse_rules to axiom8.collapse_rules
        import axiom8.collapse_rules
        axiom7_module.collapse_rules = axiom8.collapse_rules
        
        # Add to sys.modules
        sys.modules['axiom7'] = axiom7_module
        
        return True
    except Exception as e:
        print(f"Warning: Failed to fix axiom7 imports: {e}")
        return False

# Try to fix axiom7 imports at module load time
_fix_axiom7_imports()

# All exposed classes and functions - these will be lazily imported when needed
__all__ = [
    # Configuration
    'get_config',
    'update_config',
    'load_config',
    'save_config',
    'get_predefined_config',
    'CollapseGeometryConfig',
    
    # Factory functions
    'create_relational_manifold',
    'create_simulation_engine',
    'create_simulation_state',
    'create_3d_torus_visualizer',
    'create_torus_unwrapping_visualizer',
    'create_simulation',
    'seed_manifold',
    
    # Class names for type hints - these will be imported when accessed
    'RelationalManifold',
    'EnhancedContinuousDualityEngine',
    'ContinuousDualityEngine',
    'SimulationState',
    'Grain',
    'ConfigurationPoint',
    'ConfigurationSpace',
    'EpistomologyRelation',
    'EpistomologyField',
    'RelativeRotationTensor',
    'EmergentFieldRule',
    'TorusSimulationVisualizer',
    'TorusUnwrappingVisualizer',
    
    # Version
    '__version__'
]

# Lazy class imports for type hints
# This avoids loading everything at import time but still makes classes available
# through the module namespace: axiom8.RelationalManifold

def __getattr__(name):
    if name == 'RelationalManifold':
        from axiom8.core.relational_manifold import RelationalManifold
        return RelationalManifold
    elif name == 'EnhancedContinuousDualityEngine':
        from axiom8.core.engine import EnhancedContinuousDualityEngine
        return EnhancedContinuousDualityEngine
    elif name == 'ContinuousDualityEngine':  # Alias
        from axiom8.core.engine import EnhancedContinuousDualityEngine
        return EnhancedContinuousDualityEngine
    elif name == 'SimulationState':
        from axiom8.core.state import SimulationState
        return SimulationState
    elif name == 'Grain':
        from axiom8.collapse_rules.grain_dynamics import Grain
        return Grain
    elif name == 'ConfigurationPoint':
        from axiom8.collapse_rules.config_space import ConfigurationPoint
        return ConfigurationPoint
    elif name == 'ConfigurationSpace':
        from axiom8.collapse_rules.config_space import ConfigurationSpace
        return ConfigurationSpace
    elif name == 'EpistomologyRelation':
        from axiom8.collapse_rules.polarity_space import EpistomologyRelation
        return EpistomologyRelation
    elif name == 'EpistomologyField':
        from axiom8.collapse_rules.polarity_space import EpistomologyField
        return EpistomologyField
    elif name == 'RelativeRotationTensor':
        from axiom8.collapse_rules.polarity_space import RelativeRotationTensor
        return RelativeRotationTensor
    elif name == 'EmergentFieldRule':
        from axiom8.collapse_rules.emergent_field_rules import EmergentFieldRule
        return EmergentFieldRule
    # New torus visualizer classes
    elif name == 'TorusSimulationVisualizer':
        from axiom8.visualizer.torus_simulation_visualizer import TorusSimulationVisualizer
        return TorusSimulationVisualizer
    elif name == 'TorusUnwrappingVisualizer':
        from axiom8.visualizer.torus_unwrapping_visualizer import TorusUnwrappingVisualizer
        return TorusUnwrappingVisualizer
    else:
        raise AttributeError(f"module 'axiom8' has no attribute '{name}'")