"""
Core module for Axiom8 Collapse Geometry framework.

This module provides the central interfaces and coordination layers for the
collapse geometry framework, where reality emerges from constraint-based
collapse rather than classical physics.
"""

# Import primary classes and interfaces
from axiom8.core.engine import EmergentDualityEngine
from axiom8.core.relational_manifold import RelationalManifold, ToroidalCoordinator

# Import key classes from collapse rules for convenience
from axiom8.collapse_rules.grain_dynamics import (
    Grain,
    RelationalGrainSystem,
    create_random_grain
)
from axiom8.collapse_rules.config_space import (
    ConfigurationSpace,
    ConfigurationPoint
)
from axiom8.collapse_rules.polarity_space import (
    PolarityField, 
    EpistemologyRelation
)

# Make primary interfaces available at package level
__all__ = [
    'RelationalManifold',
    'ToroidalCoordinator',
    'EmergentDualityEngine',
    'Grain',
    'RelationalGrainSystem',
    'ConfigurationSpace',
    'ConfigurationPoint',
    'PolarityField',
    'EpistemologyRelation'
]