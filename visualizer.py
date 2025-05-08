"""
Visualizer Package - Visualization tools for Collapse Geometry

This package contains tools for visualizing the relational manifold,
field dynamics, and emergent structures. It prioritizes field-based 
visualization that emphasizes continuous dynamics rather than discrete 
node-line networks.
"""

# Import classes from the base visualizer module using relative imports
from .visualizer import RelationalVisualizer, AnimatedVisualizer, FieldVisualizer

# Import specialized visualizers with error handling
try:
    from axiom8.visualizer.torus_simulation_visualizer import OverhauledTorusVisualizer
    from axiom8.visualizer.torus_unwrapping_visualizer import TorusUnwrappingVisualizer
except ImportError:
    # Provide minimal stub classes if imports fail
    class TorusSimulationVisualizer:
        def __init__(self, **kwargs):
            pass
            
    class TorusUnwrappingVisualizer:
        def __init__(self, **kwargs):
            pass

# Import all visualization functions with error handling
try:
    from .visualizer import (
        # 3D visualization functions
        visualize_3d_torus,
        visualize_vortices_3d,
        visualize_phase_domains_3d,
        visualize_torus_cross_section,
        
        # 2D visualization functions
        visualize_unwrapped_torus,
        visualize_vector_field,
        visualize_phase_domains_2d,
        
        # Specialized field dynamics visualization
        create_field_dynamics_visualization,
        
        # Comprehensive visualization
        visualize_torus_complete
    )
except ImportError:
    pass

# Export all visualization functions and classes
__all__ = [
    # Base classes
    'RelationalVisualizer',
    'AnimatedVisualizer',
    'FieldVisualizer',
    
    # Visualizer implementations
    'TorusSimulationVisualizer',
    'TorusUnwrappingVisualizer',
    
    # 3D visualization functions
    'visualize_3d_torus',
    'visualize_vortices_3d',
    'visualize_phase_domains_3d',
    'visualize_torus_cross_section',
    
    # 2D visualization functions
    'visualize_unwrapped_torus',
    'visualize_vector_field',
    'visualize_phase_domains_2d',
    
    # Specialized field dynamics visualization
    'create_field_dynamics_visualization',
    
    # Comprehensive visualization
    'visualize_torus_complete'
]