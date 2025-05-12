"""
Visualizer Package - Visualization tools for Collapse Geometry

This package contains tools for visualizing the relational manifold,
field dynamics, and emergent structures. It prioritizes field-based 
visualization that emphasizes continuous dynamics rather than discrete 
node-line networks.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import base visualizer classes
try:
    from axiom8.visualizer.visualizer import RelationalVisualizer, AnimatedVisualizer, FieldVisualizer
except ImportError:
    try:
        from axiom7.visualizer.visualizer import RelationalVisualizer, AnimatedVisualizer, FieldVisualizer
    except ImportError:
        # Define minimal fallback classes if imports fail
        class RelationalVisualizer:
            """Fallback relational visualizer base class"""
            def __init__(self): pass
            
        class AnimatedVisualizer:
            """Fallback animated visualizer base class"""
            def __init__(self): pass
            
        class FieldVisualizer:
            """Fallback field visualizer base class"""
            def __init__(self): pass

# Import the enhanced torus simulation visualizer
try:
    from axiom8.visualizer.enhanced_torus_visualizer import EnhancedCollapseVisualizer as EnhancedTorusVisualizer
    from axiom8.visualizer.enhanced_torus_visualizer import create_enhanced_visualizer
except ImportError:
    try:
        from axiom7.visualizer.enhanced_torus_visualizer import EnhancedCollapseVisualizer as EnhancedTorusVisualizer
        from axiom7.visualizer.enhanced_torus_visualizer import create_enhanced_visualizer
    except ImportError:
        try:
            # Try direct import from local directory
            from .enhanced_torus_visualizer import EnhancedCollapseVisualizer as EnhancedTorusVisualizer
            from .enhanced_torus_visualizer import create_enhanced_visualizer
        except ImportError:
            # Define a minimal fallback if the import fails
            class EnhancedTorusVisualizer:
                """Fallback enhanced torus simulation visualizer"""
                def __init__(self, *args, **kwargs): pass
                def update_state(self, manifold): pass
                def render_torus_3d(self, **kwargs): 
                    fig, ax = plt.subplots()
                    ax.text(0.5, 0.5, "EnhancedTorusVisualizer not available", 
                          ha='center', va='center')
                    return fig, ax
                def create_unwrapped_visualization(self, **kwargs):
                    fig, ax = plt.subplots()
                    ax.text(0.5, 0.5, "EnhancedTorusVisualizer not available", 
                          ha='center', va='center')
                    return fig, ax
                def create_combined_visualization(self, **kwargs):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.text(0.5, 0.5, "EnhancedTorusVisualizer not available",
                          ha='center', va='center')
                    return fig
                def create_ancestry_visualization(self):
                    fig, ax = plt.subplots()
                    ax.text(0.5, 0.5, "EnhancedTorusVisualizer not available",
                          ha='center', va='center')
                    return fig
                def create_vortex_dynamics_visualization(self):
                    fig, ax = plt.subplots()
                    ax.text(0.5, 0.5, "EnhancedTorusVisualizer not available",
                          ha='center', va='center')
                    return fig
                def create_field_composition_visualization(self):
                    fig, ax = plt.subplots()
                    ax.text(0.5, 0.5, "EnhancedTorusVisualizer not available",
                          ha='center', va='center')
                    return fig
                def create_structure_enhanced_visualization(self, **kwargs):
                    fig, ax = plt.subplots()
                    ax.text(0.5, 0.5, "EnhancedTorusVisualizer not available",
                          ha='center', va='center')
                    return fig, ax
                
            def create_enhanced_visualizer(*args, **kwargs):
                return EnhancedTorusVisualizer()

# Export all visualization functions and classes
__all__ = [
    # Base classes
    'RelationalVisualizer',
    'AnimatedVisualizer',
    'FieldVisualizer',
    
    # Enhanced visualizer class and factory
    'EnhancedTorusVisualizer',
    'create_enhanced_visualizer',
    
    # Convenience visualization functions
    'visualize_enhanced_torus',
    'visualize_unwrapped_torus',
    'visualize_combined_torus',
    'visualize_ancestry_network',
    'visualize_vortex_dynamics',
    'visualize_field_composition'
]