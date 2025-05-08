"""
Grain Dynamics - Pure rules for grain evolution in the Collapse Geometry framework

Implements rules for how polarity space grains evolve, with nonlinear response curves,
tensor-based coherence, and emergent relational properties.

This module contains ONLY rules for how grains evolve, not the state itself.
All functions accept polarity space inputs and return computed values.

Enhanced with projection interfaces for better communication between components.
Integrated curvature feedback loop for proper physical behavior.
"""

import math
import random
import numpy as np
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from collections import defaultdict

# Type checking for proper imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from axiom8.collapse_rules.config_space import ConfigurationSpace, ConfigurationPoint
    from axiom8.collapse_rules.polarity_space import EpistemologyRelation, PolarityField

# Define epsilon - threshold for effectively zero values
EPSILON = 0.01

def calculate_grain_saturation(
    polarity: float,
    awareness: float,
    ancestry: Set[str],
    collapse_metric: float,
    relations: Dict[str, float],
    coherence: float,
    ancestry_depth: int = 0,
    ancestry_breadth: int = 0,
    coherence_eigenvalues: List[float] = None,
    flow_divergence: float = 0.0,
    local_curvature: float = 0.0,  # Added curvature parameter
    curvature_gradient = None,     # Added curvature gradient
    config_space = None
) -> float:
    """
    Compute grain saturation dynamically from structural properties.
    Saturation emerges from geometric impositions, not as a fundamental property.
    
    Args:
        polarity: Grain's polarity value
        awareness: Grain's awareness value
        ancestry: Set of ancestor IDs
        collapse_metric: Accumulated collapse value
        relations: Dict mapping related_id -> relation_strength
        coherence: Coherence value
        ancestry_depth: Depth of ancestry tree
        ancestry_breadth: Breadth of ancestry tree
        coherence_eigenvalues: Eigenvalues of coherence tensor
        flow_divergence: Flow field divergence
        local_curvature: Local curvature value
        curvature_gradient: Curvature gradient vector
        config_space: Optional configuration space for geometric factors
        
    Returns:
        Computed saturation value (0.0 to 1.0)
    """
    # For superposition states, saturation is always zero
    # This implements the zero=infinite potential principle
    if is_in_superposition(polarity, awareness, ancestry, collapse_metric, relations, coherence_eigenvalues):
        return 0.0
        
    # === COMPUTE SATURATION FROM STRUCTURAL PROPERTIES ===
    # 1. Ancestry contribution - now using topology metrics
    # Complex topology creates higher saturation through constraints
    ancestry_count = len(ancestry)
    ancestry_factor = min(0.4, ancestry_count * 0.02)
    
    # Topology influence - complexity increases saturation
    topology_factor = 0.0
    if ancestry_depth > 0:
        # Depth contribution - deeper ancestry = more saturation
        depth_factor = min(0.3, ancestry_depth * 0.05)
        
        # Breadth contribution - broader ancestry = more saturation
        breadth_factor = min(0.2, ancestry_breadth * 0.04)
        
        topology_factor = depth_factor + breadth_factor
    
    # Combine for total ancestry contribution
    ancestry_saturation = ancestry_factor + topology_factor * 0.5
    
    # 2. Collapse history contribution - now with nonlinear scaling
    # Higher collapse metric = diminishing returns on saturation
    if collapse_metric > 0:
        collapse_saturation = min(0.7, 0.3 * math.pow(collapse_metric, 0.8))
    else:
        collapse_saturation = 0.0
    
    # 3. Relational commitment contribution
    if not relations:
        relation_saturation = 0.0
    else:
        # Calculate relational complexity
        relation_strength = sum(abs(r) for r in relations.values()) / len(relations)
        relation_complexity = len(relations) * 0.02
        
        # Combine with nonlinear scaling
        relation_saturation = min(0.5, relation_strength * 0.3 * (1 + relation_complexity))
    
    # 4. Coherence loss contribution using tensor eigenvalues
    # Anisotropic coherence = higher saturation (structure has committed)
    if coherence_eigenvalues is not None:
        # Check if eigenvalues is a NumPy array
        if isinstance(coherence_eigenvalues, np.ndarray):
            # Calculate anisotropy using NumPy functions safely
            max_val = np.max(coherence_eigenvalues)
            min_val = np.min(coherence_eigenvalues)
            anisotropy = (max_val - min_val) / max(max_val, 0.001)
        else:
            # Handle regular list
            anisotropy = (max(coherence_eigenvalues) - min(coherence_eigenvalues)) / max(max(coherence_eigenvalues), 0.001)
        
        # Anisotropy directly contributes to saturation
        coherence_saturation = min(0.4, anisotropy * 0.5)
    else:
        # Fallback to scalar coherence
        coherence_loss = 1.0 - coherence
        coherence_saturation = coherence_loss * 0.3
    
    # 5. Flow field contribution
    # Higher flow field divergence = higher saturation
    flow_saturation = min(0.3, abs(flow_divergence) * 0.5)
    
    # 6. Curvature contribution (NEW)
    # Higher local curvature = higher saturation 
    # Curvature represents structural constraints that increase saturation
    curvature_saturation = min(0.4, local_curvature * 0.5)
    
    # Increase saturation if curvature gradient is aligned with awareness direction
    curvature_alignment = 0.0
    if curvature_gradient is not None and isinstance(curvature_gradient, np.ndarray):
        if np.linalg.norm(curvature_gradient) > 0.01:
            # Alignment with awareness (assuming z-component reflects awareness)
            # This captures how curvature "pulls" in the awareness direction
            curvature_dir = curvature_gradient / np.linalg.norm(curvature_gradient)
            awareness_sign = 1.0 if awareness > 0 else -1.0
            alignment = curvature_dir[2] * awareness_sign  # Z-component alignment with awareness
            curvature_alignment = max(0.0, alignment * 0.2)
    
    # Combined curvature contribution
    total_curvature_saturation = curvature_saturation + curvature_alignment
    
    # === GEOMETRIC FACTORS FROM CONFIGURATION SPACE ===
    geometry_saturation = 0.0
    if config_space:
        # Add geometric factors here if needed
        pass
    
    # === DETERMINE FINAL SATURATION WITH NONLINEAR INTEGRATION ===
    # Base saturation components
    components = [
        ancestry_saturation * 0.23,      # Reduced slightly to make room for curvature
        collapse_saturation * 0.23,      # Reduced slightly to make room for curvature
        relation_saturation * 0.18,      # Reduced slightly to make room for curvature
        coherence_saturation * 0.13,     # Reduced slightly to make room for curvature
        flow_saturation * 0.08,          # Reduced slightly to make room for curvature
        total_curvature_saturation * 0.1,  # NEW: Curvature contribution (10%)
        geometry_saturation * 0.05
    ]
    
    # Calculate core saturation 
    base_saturation = sum(components)
    
    # Apply nonlinear sigmoid-like curve for final scaling
    # This creates clearer separation between low/high saturation regimes
    k = 4.0  # Sigmoid steepness
    midpoint = 0.4  # Sigmoid midpoint
    
    # Sigmoid function: 1 / (1 + e^(-k(x - midpoint)))
    sigmoid = lambda x: 1 / (1 + math.exp(-k * (x - midpoint)))
    
    # Scale to [0,1] range
    sigmoid_min = sigmoid(0)
    sigmoid_max = sigmoid(1)
    
    # Apply sigmoid scaling
    scaled_saturation = (sigmoid(base_saturation) - sigmoid_min) / (sigmoid_max - sigmoid_min)
    
    # Ensure bounds
    saturation = max(0.0, min(1.0, scaled_saturation))
    
    # Return computed saturation
    return saturation

def is_in_superposition(
    polarity: float,
    awareness: float,
    ancestry: Set[str],
    collapse_metric: float,
    relations: Dict[str, float],
    coherence_eigenvalues = None,
    unbounded_potential: bool = True
) -> bool:
    """
    Determine if a grain would be in a state of superposition (unbounded potential).
    Implements the Superpositional Zero Principle where zero = infinite potential.
    
    Args:
        polarity: Grain's polarity value
        awareness: Grain's awareness value
        ancestry: Set of ancestor IDs
        collapse_metric: Accumulated collapse value
        relations: Dict mapping related_id -> relation_strength
        coherence_eigenvalues: Optional eigenvalues of coherence tensor
        unbounded_potential: Whether explicitly in unbounded state
        
    Returns:
        True if in superposition, False otherwise
    """
    # Calculate saturation using the provided properties
    # For this check, we use a simplified calculation
    saturation = 0.0
    if len(ancestry) > 0:
        saturation += min(0.3, len(ancestry) * 0.02)
    if collapse_metric > 0:
        saturation += min(0.3, collapse_metric * 0.3)
    if relations:
        rel_strength = sum(abs(r) for r in relations.values()) / len(relations)
        saturation += min(0.3, rel_strength * 0.3)
    
    # Check for superposition criteria
    is_unsaturated = saturation < EPSILON
    is_near_zero_awareness = abs(awareness) < EPSILON
    has_no_ancestry = not ancestry
    no_collapse_history = collapse_metric < EPSILON
    
    # Check for coherence tensor isotropy (indicates superposition)
    tensor_isotropy = True
    if coherence_eigenvalues is not None:
        # Handle NumPy arrays properly
        if isinstance(coherence_eigenvalues, np.ndarray):
            # Check if all eigenvalues are nearly identical (isotropic)
            if coherence_eigenvalues.size > 0:  # Ensure array is not empty
                max_val = np.max(coherence_eigenvalues)
                min_val = np.min(coherence_eigenvalues)
                if max_val - min_val > 0.1:
                    tensor_isotropy = False
        else:
            # Handle regular list
            if coherence_eigenvalues and len(coherence_eigenvalues) > 0:
                if max(coherence_eigenvalues) - min(coherence_eigenvalues) > 0.1:
                    tensor_isotropy = False
    
    # Check for relational memory
    has_no_relations = len(relations) == 0
    has_minimal_relations = sum(1 for r in relations.values() if abs(r) > 0.1) == 0
    
    # Superposition = unresolved potential = zero that equals infinity
    return ((is_unsaturated and is_near_zero_awareness) or 
            (is_unsaturated and (has_no_ancestry or no_collapse_history)) or
            (is_unsaturated and (has_no_relations or has_minimal_relations)) or
            (tensor_isotropy and unbounded_potential))

def calculate_degrees_of_freedom(
    saturation: float,
    ancestry_depth: int = 0,
    ancestry_breadth: int = 0,
    ancestry_count: int = 0,
    relations: Dict[str, float] = None,
    coherence_eigenvalues = None,
    coherence: float = 1.0,
    local_curvature: float = 0.0,  # Added curvature parameter
    is_superposition: bool = False
) -> float:
    """
    Calculate the degrees of freedom for a grain.
    Freedom decreases as structure commits through saturation.
    
    Args:
        saturation: Grain saturation value
        ancestry_depth: Depth of ancestry tree
        ancestry_breadth: Breadth of ancestry tree
        ancestry_count: Count of ancestors
        relations: Dict mapping related_id -> relation_strength
        coherence_eigenvalues: Eigenvalues of coherence tensor
        coherence: Scalar coherence value
        local_curvature: Local curvature value
        is_superposition: Whether grain is in superposition
        
    Returns:
        Freedom value (0.0 to 1.0)
    """
    if is_superposition:
        return 1.0  # Maximum freedom in superposition
    
    # Freedom decreases as saturation increases - nonlinear curve
    # Diminishing reduction at higher saturations
    freedom_from_saturation = 1.0 - math.pow(saturation, 0.8)
    
    # Freedom decreases with ancestry topology
    # Complex ancestry = less freedom
    if ancestry_depth > 0 and ancestry_breadth > 0:
        topology_factor = min(0.6, (ancestry_depth * 0.05 + ancestry_breadth * 0.03))
        freedom_from_ancestry = 1.0 - topology_factor
    else:
        # Fallback to simple ancestry count
        ancestry_factor = min(1.0, ancestry_count * 0.05)
        freedom_from_ancestry = 1.0 - ancestry_factor
    
    # Freedom decreases with relation commitment and complexity
    if not relations:
        relation_constraint = 0.0
    else:
        # Base constraint from relation strengths
        strength_constraint = sum(abs(r) for r in relations.values()) / len(relations)
        
        # Added constraint from relation count (complexity)
        complexity_factor = min(0.3, len(relations) * 0.02)
        
        relation_constraint = strength_constraint * (1.0 + complexity_factor)
        
    freedom_from_relations = 1.0 - min(1.0, relation_constraint)
    
    # Freedom decreases with coherence tensor anisotropy
    # Isotropic coherence = high freedom, anisotropic = low freedom
    if coherence_eigenvalues is not None:
        # Handle NumPy arrays properly
        if isinstance(coherence_eigenvalues, np.ndarray):
            # Calculate anisotropy safely
            if coherence_eigenvalues.size > 0:  # Ensure array is not empty
                max_val = np.max(coherence_eigenvalues)
                min_val = np.min(coherence_eigenvalues)
                anisotropy = (max_val - min_val) / max(max_val, 0.001)
            else:
                anisotropy = 0.0
        else:
            # Handle regular list or tuple
            if coherence_eigenvalues and len(coherence_eigenvalues) > 0:
                anisotropy = (max(coherence_eigenvalues) - min(coherence_eigenvalues)) / max(max(coherence_eigenvalues), 0.001)
            else:
                anisotropy = 0.0
        
        # Anisotropy directly reduces freedom
        freedom_from_coherence = 1.0 - min(0.7, anisotropy * 0.8)
    else:
        # Fallback to scalar coherence
        freedom_from_coherence = coherence
    
    # NEW: Freedom decreases with curvature
    # Higher curvature = more structural constraints = less freedom
    freedom_from_curvature = 1.0 - min(0.8, local_curvature * 0.7)
    
    # Combined freedom - weighted factors with nonlinear combination
    # This uses a geometric mean to ensure all factors matter
    components = [
        freedom_from_saturation,
        freedom_from_ancestry,
        freedom_from_relations,
        freedom_from_coherence,
        freedom_from_curvature  # Added curvature component
    ]
    
    weights = [0.35, 0.15, 0.15, 0.15, 0.2]  # Updated weights to include curvature
    
    # Weighted geometric mean
    freedom = 1.0
    for component, weight in zip(components, weights):
        # Avoid zero values
        component = max(0.01, component)
        freedom *= math.pow(component, weight)
    
    return max(0.0, min(1.0, freedom))

def calculate_nonlinear_response(
    parameter: str,
    magnitude: float,
    saturation: float,
    local_curvature: float = 0.0,  # Added curvature parameter
    response_curves: Dict[str, Dict[str, float]] = None
) -> float:
    """
    Calculate nonlinear response for a parameter based on field properties.
    
    Args:
        parameter: Parameter type ('awareness', 'polarity', or 'coherence')
        magnitude: Base magnitude of change
        saturation: Current saturation value
        local_curvature: Local curvature value
        response_curves: Optional response curve parameters
        
    Returns:
        Nonlinear scaled response value
    """
    # Default response curves if not provided
    if response_curves is None:
        response_curves = {
            'awareness': {'baseline': 0.2, 'nonlinearity': 1.5, 'saturation_factor': -0.5, 'curvature_factor': 0.3},
            'polarity': {'baseline': 0.3, 'nonlinearity': 1.2, 'saturation_factor': -0.3, 'curvature_factor': 0.2},
            'coherence': {'baseline': 0.2, 'nonlinearity': 1.0, 'saturation_factor': -0.7, 'curvature_factor': -0.2}
        }
    
    if parameter not in response_curves:
        return magnitude
        
    # Get response curve parameters
    curve = response_curves[parameter]
    baseline = curve['baseline']
    nonlinearity = curve['nonlinearity']
    saturation_factor = curve['saturation_factor']
    
    # NEW: Add curvature factor to response curves
    curvature_factor = curve.get('curvature_factor', 0.0)
    
    # Calculate saturation effect
    # High saturation reduces response (harder to change committed structure)
    saturation_effect = 1.0 + saturation_factor * saturation
    
    # NEW: Calculate curvature effect
    # Curvature can enhance or dampen response based on parameter
    curvature_effect = 1.0 + curvature_factor * local_curvature
    
    # Calculate nonlinear scaling
    # power > 1: diminishing returns for large changes
    # power < 1: amplifying effect for small changes
    nonlinear_magnitude = math.pow(magnitude, nonlinearity)
    
    # Combine with baseline, saturation and curvature effects
    response = baseline * nonlinear_magnitude * saturation_effect * curvature_effect
    
    return response

def process_structure_collapse(
    awareness: float,
    polarity: float,
    coherence: float,
    collapse_strength: float,
    saturation: float,
    is_superposition: bool,
    local_curvature: float = 0.0,      # Added curvature parameters
    curvature_gradient = None,         # Added curvature parameters
    ancestry_curvature: float = 0.0,   # Added ancestry curvature
    response_curves: Dict[str, Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Process a positive (structure-forming) collapse event.
    
    Args:
        awareness: Current awareness value
        polarity: Current polarity value
        coherence: Current coherence value
        collapse_strength: Positive collapse strength (0.0 to 1.0)
        saturation: Current saturation value
        is_superposition: Whether grain is in superposition
        local_curvature: Local curvature value
        curvature_gradient: Curvature gradient vector
        ancestry_curvature: Ancestry-derived curvature
        response_curves: Optional response curve parameters
        
    Returns:
        Dictionary with updated awareness, polarity, coherence values
    """
    # If in superposition, resolve it
    if is_superposition:
        # Simple transition from superposition
        new_awareness = collapse_strength * 0.2
        new_polarity = max(polarity, 0.1)  # Ensure positive polarity
        new_coherence = 0.8 + random.random() * 0.2  # Initial coherence
    else:
        # Calculate nonlinear awareness response with curvature influence
        awareness_response = calculate_nonlinear_response(
            'awareness', collapse_strength, saturation, local_curvature, response_curves)
        
        # Update awareness toward positive
        new_awareness = min(1.0, awareness + awareness_response)
        
        # Calculate nonlinear polarity response with curvature influence
        polarity_response = calculate_nonlinear_response(
            'polarity', collapse_strength, saturation, local_curvature, response_curves)
        
        # Update polarity toward positive
        new_polarity = min(1.0, polarity + polarity_response)
        
        # Calculate nonlinear coherence response with curvature influence
        coherence_response = calculate_nonlinear_response(
            'coherence', collapse_strength * 0.7, saturation, local_curvature, response_curves)
        
        # Update coherence - structure increases coherence
        new_coherence = min(1.0, coherence + coherence_response)
        
        # NEW: Apply curvature gradient influence to direct awareness and polarity flow
        if curvature_gradient is not None and isinstance(curvature_gradient, np.ndarray):
            if np.linalg.norm(curvature_gradient) > 0.01:
                # Normalize gradient
                curvature_dir = curvature_gradient / np.linalg.norm(curvature_gradient)
                
                # Use curvature to bias polarity along the z-component
                curvature_influence = local_curvature * 0.2
                polarity_bias = curvature_influence * curvature_dir[2]  # Use z-component for polarity
                new_polarity = min(1.0, new_polarity + polarity_bias)
                
                # Enhance awareness in high curvature regions
                awareness_bias = local_curvature * 0.15
                new_awareness = min(1.0, new_awareness + awareness_bias)
                
                # Apply ancestry curvature enhancement
                if ancestry_curvature > 0:
                    # Ancestral curvature boosts awareness more (stronger recursive effects)
                    ancestry_boost = ancestry_curvature * 0.1
                    new_awareness = min(1.0, new_awareness + ancestry_boost)
    
    return {
        'awareness': new_awareness,
        'polarity': new_polarity,
        'coherence': new_coherence,
        'resolution': not is_superposition or collapse_strength > 0.1
    }

def process_decay_collapse(
    awareness: float,
    polarity: float,
    coherence: float,
    collapse_strength: float,  # Negative value
    saturation: float,
    is_superposition: bool,
    local_curvature: float = 0.0,      # Added curvature parameters
    curvature_gradient = None,         # Added curvature parameters
    ancestry_curvature: float = 0.0,   # Added ancestry curvature
    response_curves: Dict[str, Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Process a negative (decay) collapse event.
    
    Args:
        awareness: Current awareness value
        polarity: Current polarity value
        coherence: Current coherence value
        collapse_strength: Negative collapse strength (-1.0 to 0.0)
        saturation: Current saturation value
        is_superposition: Whether grain is in superposition
        local_curvature: Local curvature value
        curvature_gradient: Curvature gradient vector
        ancestry_curvature: Ancestry-derived curvature
        response_curves: Optional response curve parameters
        
    Returns:
        Dictionary with updated awareness, polarity, coherence values
    """
    # No effect on superposition states - they have nothing to decay
    if is_superposition:
        return {
            'awareness': awareness,
            'polarity': polarity,
            'coherence': coherence,
            'resolution': False
        }
    
    # Get magnitude (absolute value)
    magnitude = abs(collapse_strength)
    
    # Calculate nonlinear awareness response with curvature influence
    # Higher saturation = less affected by decay
    awareness_response = calculate_nonlinear_response(
        'awareness', magnitude, saturation, local_curvature, response_curves)
    
    # Update awareness (decay pulls toward negative)
    new_awareness = max(-1.0, awareness - awareness_response)
    
    # Calculate nonlinear polarity response with curvature influence
    polarity_response = calculate_nonlinear_response(
        'polarity', magnitude, saturation, local_curvature, response_curves)
    
    # Update polarity toward negative
    new_polarity = max(-1.0, polarity - polarity_response)
    
    # Calculate nonlinear coherence response with curvature influence
    coherence_response = calculate_nonlinear_response(
        'coherence', magnitude, saturation, local_curvature, response_curves)
    
    # Reduce coherence
    new_coherence = max(0.3, coherence - coherence_response)
    
    # NEW: Apply curvature gradient influence for decay collapse
    # Different dynamics compared to structure collapse
    if curvature_gradient is not None and isinstance(curvature_gradient, np.ndarray):
        if np.linalg.norm(curvature_gradient) > 0.01:
            # Normalize gradient
            curvature_dir = curvature_gradient / np.linalg.norm(curvature_gradient)
            
            # For decay, high curvature can "trap" decay (reducing its effect)
            # This models how curved spacetime contains energy
            damping_factor = min(0.5, local_curvature * 0.7)
            
            # Apply partial recovery based on curvature (dampening decay)
            awareness_dampening = damping_factor * 0.15
            new_awareness = min(1.0, new_awareness + awareness_dampening)
            
            # For decay in high ancestry curvature regions, preserve some structure
            if ancestry_curvature > 0.3:
                # Ancestral curvature resists decay (legacy protection)
                resistance = ancestry_curvature * 0.2
                new_awareness = min(1.0, new_awareness + resistance)
    
    return {
        'awareness': new_awareness,
        'polarity': new_polarity,
        'coherence': new_coherence,
        'resolution': False  # Decay never resolves superposition
    }

def calculate_collapse_propagation(
    source_polarity: float,
    source_saturation: float,
    source_coherence: float,
    target_polarity: float,
    collapse_strength: float,
    coherence_eigenvalues = None,
    coherence_eigenvectors = None,
    flow_vector = None,
    relational_vector = None,
    source_curvature: float = 0.0,  # Added curvature parameters
    target_curvature: float = 0.0   # Added curvature parameters
) -> float:
    """
    Calculate collapse propagation strength from source to target.
    Works for both positive (structure) and negative (decay) collapse.
    
    Args:
        source_polarity: Source polarity value
        source_saturation: Source saturation value
        source_coherence: Source coherence value
        target_polarity: Target polarity value
        collapse_strength: Signed strength of collapse (-1.0 to 1.0)
        coherence_eigenvalues: Optional eigenvalues of coherence tensor
        coherence_eigenvectors: Optional eigenvectors of coherence tensor
        flow_vector: Optional flow vector
        relational_vector: Optional relational vector between source and target
        source_curvature: Source grain's local curvature
        target_curvature: Target grain's local curvature
        
    Returns:
        Propagated collapse strength
    """
    # Determine if we're propagating structure or decay
    is_decay = collapse_strength < 0
    
    # Calculate lightlike factor based on saturation and tensor properties
    # Low saturation = lightlike behavior = better propagation
    base_lightlike = max(0.0, 1.0 - source_saturation * 2.0)
    
    # Check coherence anisotropy for enhanced lightlike behavior
    # Light propagates best along eigenvectors with highest eigenvalues
    tensor_lightlike = 0.0
    if coherence_eigenvalues is not None and coherence_eigenvectors is not None:
        # Handle NumPy arrays properly
        if isinstance(coherence_eigenvalues, np.ndarray) and isinstance(coherence_eigenvectors, np.ndarray):
            # Get maximum eigenvalue and corresponding eigenvector
            if coherence_eigenvalues.size > 0:  # Ensure array is not empty
                max_idx = np.argmax(coherence_eigenvalues)
                max_eval = coherence_eigenvalues[max_idx]
                max_evec = coherence_eigenvectors[:, max_idx]
                
                # Calculate alignment with flow
                if flow_vector is not None:
                    flow_norm = np.linalg.norm(flow_vector)
                    if flow_norm > 0.01:
                        flow_dir = flow_vector / flow_norm
                        alignment = np.abs(np.dot(max_evec, flow_dir))
                        
                        # High alignment = enhanced light propagation
                        tensor_lightlike = alignment * max_eval * 0.3
        
    # Combined lightlike factor
    lightlike_factor = min(1.0, base_lightlike + tensor_lightlike)
    
    # Base propagation factor
    if lightlike_factor > 0.5:
        # Lightlike structures propagate with minimal loss
        base_factor = 0.9
    else:
        # Regular structure has more dissipation
        base_factor = 0.7
    
    # Adjust by phase coherence
    coherence_bonus = source_coherence * 0.2
    
    # Add tensor-based enhancement if propagating along primary eigenvector
    tensor_bonus = 0.0
    if coherence_eigenvalues is not None and coherence_eigenvectors is not None and relational_vector is not None:
        # Handle NumPy arrays properly
        if isinstance(coherence_eigenvalues, np.ndarray) and isinstance(coherence_eigenvectors, np.ndarray):
            rel_norm = np.linalg.norm(relational_vector)
            if rel_norm > 0.01:
                rel_dir = relational_vector / rel_norm
                
                # Check alignment with principal eigenvector
                if coherence_eigenvalues.size > 0:  # Ensure array is not empty
                    max_idx = np.argmax(coherence_eigenvalues)
                    max_evec = coherence_eigenvectors[:, max_idx]
                    
                    alignment = np.abs(np.dot(max_evec, rel_dir))
                    
                    # Higher alignment = better propagation
                    tensor_bonus = alignment * 0.1
    
    # NEW: Calculate curvature effect on propagation
    # Higher source curvature = enhanced propagation (like gravity "pulling")
    # Higher target curvature = capturing more of the propagation
    curvature_bonus = 0.0
    curvature_effect = source_curvature * 0.3 + target_curvature * 0.2
    
    # Curvature gradient effect
    if relational_vector is not None and np.linalg.norm(relational_vector) > 0.01:
        # Collapse flows down curvature gradients (like gravity)
        # The higher the curvature differential, the stronger the propagation
        curvature_differential = target_curvature - source_curvature
        
        # For positive differential (flowing into higher curvature)
        # Enhance propagation for structure, dampen for decay
        if curvature_differential > 0:
            curvature_bonus = curvature_differential * (0.2 if not is_decay else -0.1)
        else:
            # For negative differential (flowing out of higher curvature)
            # Dampen propagation for structure, enhance for decay
            curvature_bonus = abs(curvature_differential) * (-0.1 if not is_decay else 0.2)
    
    # Apply total curvature effect (capped)
    curvature_effect += curvature_bonus
    curvature_effect = max(-0.3, min(0.3, curvature_effect))
            
    # Calculate propagation factor with curvature
    propagation_factor = min(0.95, base_factor + coherence_bonus + tensor_bonus + curvature_effect)
    
    # Calculate polarity alignment
    polarity_alignment = source_polarity * target_polarity
    
    if polarity_alignment > 0:
        # Aligned polarities enhance propagation (constructive interference)
        alignment_boost = math.sqrt(abs(polarity_alignment)) * 0.1
        propagation_factor = min(0.97, propagation_factor + alignment_boost)
    elif polarity_alignment < -0.5:
        # Strongly opposed polarities reduce propagation (destructive interference)
        alignment_penalty = abs(polarity_alignment) * 0.2
        propagation_factor = max(0.1, propagation_factor - alignment_penalty)
    
    # Calculate propagated strength (preserving sign)
    propagated_strength = collapse_strength * propagation_factor
    
    # Add quantum fluctuation for lightlike behavior
    if lightlike_factor > 0.5 and random.random() < 0.1:
        # Fluctuation preserves direction (sign)
        sign = -1.0 if is_decay else 1.0
        
        # Scale fluctuation by coherence (higher coherence = smaller fluctuations)
        fluctuation_scale = 0.05 * (1.0 - source_coherence * 0.5)
        quantum_fluctuation = random.uniform(0, fluctuation_scale) * sign
        
        propagated_strength += quantum_fluctuation
    
    return propagated_strength

def predict_cascade_direction(
    polarity: float,
    saturation: float,
    coherence: float,
    is_superposition: bool,
    flow_vector = None,
    coherence_eigenvalues = None,
    coherence_eigenvectors = None,
    relations: Dict[str, float] = None,
    relational_vectors: Dict[str, np.ndarray] = None,
    local_curvature: float = 0.0,     # Added curvature parameters
    curvature_gradient = None         # Added curvature gradient
) -> Dict[str, Any]:
    """
    Predict direction and probability of cascade collapse.
    
    Args:
        polarity: Grain polarity value
        saturation: Grain saturation value
        coherence: Grain coherence value
        is_superposition: Whether grain is in superposition
        flow_vector: Optional flow vector
        coherence_eigenvalues: Optional eigenvalues of coherence tensor
        coherence_eigenvectors: Optional eigenvectors of coherence tensor
        relations: Optional dict mapping related_id -> relation_strength
        relational_vectors: Optional dict mapping related_id -> relational vector
        local_curvature: Local curvature value
        curvature_gradient: Curvature gradient vector
        
    Returns:
        Dictionary with cascade prediction metrics
    """
    # Calculate cascade potential 
    # Lightlike factor - high freedom = high cascade potential
    lightlike_factor = max(0.0, 1.0 - saturation * 2.0)
    
    # Get coherence tensor properties
    if coherence_eigenvalues is not None and coherence_eigenvectors is not None:
        # Handle NumPy arrays properly
        if isinstance(coherence_eigenvalues, np.ndarray) and isinstance(coherence_eigenvectors, np.ndarray):
            # Anisotropy factor - how directed is the coherence
            if coherence_eigenvalues.size > 0:  # Ensure array is not empty
                max_val = np.max(coherence_eigenvalues)
                min_val = np.min(coherence_eigenvalues)
                anisotropy = (max_val - min_val) / max(max_val, 0.001)
                
                # Get primary eigenvector (direction of highest coherence)
                max_idx = np.argmax(coherence_eigenvalues)
                primary_direction = coherence_eigenvectors[:, max_idx]
                
                # Coherence strength in primary direction
                coherence_strength = coherence_eigenvalues[max_idx]
            else:
                anisotropy = 0.0
                primary_direction = np.zeros(3)
                coherence_strength = 0.0
        else:
            # Handle regular lists/tuples
            if coherence_eigenvalues and len(coherence_eigenvalues) > 0:
                anisotropy = (max(coherence_eigenvalues) - min(coherence_eigenvalues)) / max(max(coherence_eigenvalues), 0.001)
                max_idx = coherence_eigenvalues.index(max(coherence_eigenvalues))
                primary_direction = coherence_eigenvectors[max_idx] if coherence_eigenvectors else np.zeros(3)
                coherence_strength = coherence_eigenvalues[max_idx]
            else:
                anisotropy = 0.0
                primary_direction = np.zeros(3)
                coherence_strength = 0.0
        
        # Combined coherence factor - higher for strong anisotropic coherence
        coherence_factor = anisotropy * coherence_strength
    else:
        # Fallback to scalar coherence
        coherence_factor = coherence - 0.5
        coherence_factor = max(0.0, coherence_factor) * 2.0
        primary_direction = np.zeros(3)
        anisotropy = 0.0
    
    # Polarization factor - strong polarity (any direction) = high potential
    polarity_factor = abs(polarity) * 0.5
    
    # Flow field contribution - strong directed flow = high potential
    flow_factor = 0.0
    if flow_vector is not None:
        flow_magnitude = np.linalg.norm(flow_vector)
        flow_factor = min(0.3, flow_magnitude * 0.5)
    
    # NEW: Curvature contribution to cascade potential
    # High curvature and strong gradient increase cascade potential
    curvature_factor = 0.0
    if local_curvature > 0:
        # Base curvature contribution
        curvature_factor = min(0.3, local_curvature * 0.5)
        
        # Enhance with gradient if available
        if curvature_gradient is not None and isinstance(curvature_gradient, np.ndarray):
            gradient_magnitude = np.linalg.norm(curvature_gradient)
            if gradient_magnitude > 0.01:
                # Gradient enhances cascade probability
                gradient_contribution = min(0.2, gradient_magnitude * 0.3)
                curvature_factor += gradient_contribution
    
    # Calculate cascade potential - include curvature contribution
    potential = (
        lightlike_factor * 0.4 +          # Reduced to make room for curvature
        coherence_factor * 0.15 +         # Reduced to make room for curvature
        polarity_factor * 0.15 +          # Reduced to make room for curvature
        flow_factor * 0.1 +               # Reduced to make room for curvature
        curvature_factor * 0.2            # NEW: Curvature contribution (20%)
    )
    
    # Clamp to range
    cascade_potential = max(0.0, min(1.0, potential))
    
    # Calculate preferred cascade direction
    # This combines coherence eigenvector, flow direction, polarity and curvature gradient
    cascade_direction = np.zeros(3)
    
    # Component vectors with weights
    direction_components = []
    weights = []
    
    # Add coherence direction if available
    if isinstance(primary_direction, np.ndarray) and np.linalg.norm(primary_direction) > 0.01:
        direction_components.append(primary_direction)
        weights.append(0.5)  # Reduced from 0.6 to make room for curvature
    
    # Add flow direction if available
    if flow_vector is not None and np.linalg.norm(flow_vector) > 0.01:
        flow_dir = flow_vector / np.linalg.norm(flow_vector)
        direction_components.append(flow_dir)
        weights.append(0.2)  # Reduced from 0.3 to make room for curvature
    
    # Add relational contribution from strongest relation
    if relations is not None and relational_vectors is not None:
        max_relation = 0.0
        max_rel_vec = None
        
        for rel_id, strength in relations.items():
            if abs(strength) > abs(max_relation) and rel_id in relational_vectors:
                max_relation = strength
                max_rel_vec = relational_vectors[rel_id]
        
        if max_rel_vec is not None:
            rel_norm = np.linalg.norm(max_rel_vec)
            if rel_norm > 0.01:
                rel_dir = max_rel_vec / rel_norm
                direction_components.append(rel_dir)
                weights.append(0.1)
    
    # NEW: Add curvature gradient direction
    if curvature_gradient is not None and isinstance(curvature_gradient, np.ndarray):
        if np.linalg.norm(curvature_gradient) > 0.01:
            # Normalize gradient
            curve_dir = curvature_gradient / np.linalg.norm(curvature_gradient)
            direction_components.append(curve_dir)
            weights.append(0.2)  # Significant influence (20%)
    
    # Combine components with weights
    if direction_components:
        # Normalize weights
        if sum(weights) > 0:
            normalized_weights = [w / sum(weights) for w in weights]
            
            # Weighted sum
            for component, weight in zip(direction_components, normalized_weights):
                cascade_direction += component * weight
            
            # Normalize final direction
            dir_norm = np.linalg.norm(cascade_direction)
            if dir_norm > 0.01:
                cascade_direction = cascade_direction / dir_norm
    
    # Calculate structure vs decay probability based on polarity
    structure_prob = 0.0
    decay_prob = 0.0
    
    # Base probabilities from polarity
    if polarity > 0:
        # Positive polarity - more likely to trigger structure cascade
        structure_prob = cascade_potential * (0.5 + polarity * 0.5)
        decay_prob = cascade_potential * 0.1  # Small chance of opposite cascade
    elif polarity < 0:
        # Negative polarity - more likely to trigger decay cascade
        decay_prob = cascade_potential * (0.5 + abs(polarity) * 0.5)
        structure_prob = cascade_potential * 0.1  # Small chance of opposite cascade
    else:
        # Neutral polarity - equal chance of either
        structure_prob = cascade_potential * 0.5
        decay_prob = cascade_potential * 0.5
    
    # NEW: Curvature influence on cascade type
    # Higher curvature biases toward structure formation
    # This implements gravity-like effect where curvature concentrates structure
    if local_curvature > 0.3:
        # Shift probabilities toward structure
        curvature_shift = local_curvature * 0.2
        structure_bias = min(0.2, curvature_shift)
        
        # Increase structure probability, decrease decay probability
        structure_prob = min(1.0, structure_prob + structure_bias)
        decay_prob = max(0.0, decay_prob - structure_bias * 0.5)
    
    # Higher for superposition states (they can collapse either way)
    if is_superposition:
        # Superposition inherently has high cascade potential
        # But direction remains governed by any polarity bias
        factor = 0.3
        structure_prob += factor
        decay_prob += factor
    
    return {
        'structure_prob': min(1.0, structure_prob),
        'decay_prob': min(1.0, decay_prob),
        'polarity': polarity,
        'lightlike_factor': lightlike_factor,
        'is_superposition': is_superposition,
        'cascade_potential': cascade_potential,
        'anisotropy': anisotropy,
        'local_curvature': local_curvature,  # Added curvature info
        'cascade_direction': cascade_direction.tolist() if isinstance(cascade_direction, np.ndarray) else cascade_direction,
        'flow_magnitude': flow_factor * 2.0  # Scale back to approximate magnitude
    }

def update_coherence_tensor(
    coherence_scalar: float,
    flow_vector = None,
    curvature_gradient = None,  # Added curvature gradient
    local_curvature: float = 0.0,  # Added local curvature
    is_superposition: bool = False
) -> Dict[str, Any]:
    """
    Calculate coherence tensor using flow, phase information and curvature.
    
    Args:
        coherence_scalar: Scalar coherence value
        flow_vector: Optional flow vector
        curvature_gradient: Optional curvature gradient vector
        local_curvature: Local curvature value
        is_superposition: Whether grain is in superposition
        
    Returns:
        Dictionary with coherence tensor, eigenvalues, and eigenvectors
    """
    # For superposition or missing flow, use isotropic tensor
    if is_superposition or (flow_vector is None and curvature_gradient is None):
        # Isotropic coherence (identity tensor scaled by coherence)
        coherence_tensor = np.eye(3) * coherence_scalar
        coherence_eigenvalues = np.ones(3) * coherence_scalar
        coherence_eigenvectors = np.eye(3)
        return {
            'tensor': coherence_tensor,
            'eigenvalues': coherence_eigenvalues,
            'eigenvectors': coherence_eigenvectors
        }
    
    # Calculate combined primary direction from flow and curvature
    primary_direction = None
    primary_magnitude = 0.0
    
    # Check flow vector
    if flow_vector is not None:
        flow_norm = np.linalg.norm(flow_vector)
        if flow_norm > 0.01:
            primary_direction = flow_vector / flow_norm
            primary_magnitude = flow_norm
    
    # NEW: Add curvature gradient influence
    if curvature_gradient is not None and isinstance(curvature_gradient, np.ndarray):
        curve_norm = np.linalg.norm(curvature_gradient)
        if curve_norm > 0.01:
            # If we already have a primary direction, blend them
            if primary_direction is not None:
                # Get normalized gradient
                curve_dir = curvature_gradient / curve_norm
                
                # Blend directions (60% flow, 40% curvature)
                # Weight depends on relative magnitudes and local curvature
                curve_weight = 0.4 * (local_curvature + 0.5)
                flow_weight = 1.0 - curve_weight
                
                # Weighted blend
                combined_dir = (primary_direction * flow_weight + curve_dir * curve_weight)
                combined_norm = np.linalg.norm(combined_dir)
                
                if combined_norm > 0.01:
                    primary_direction = combined_dir / combined_norm
                    primary_magnitude = max(primary_magnitude, curve_norm)
            else:
                # Use curvature gradient as primary direction
                primary_direction = curvature_gradient / curve_norm
                primary_magnitude = curve_norm
    
    # If no valid direction found, use isotropic tensor
    if primary_direction is None or primary_magnitude < 0.01:
        coherence_tensor = np.eye(3) * coherence_scalar
        coherence_eigenvalues = np.ones(3) * coherence_scalar
        coherence_eigenvectors = np.eye(3)
        return {
            'tensor': coherence_tensor,
            'eigenvalues': coherence_eigenvalues,
            'eigenvectors': coherence_eigenvectors
        }
    
    # Create projection matrix along primary direction
    projection = np.outer(primary_direction, primary_direction)
    
    # Create tensor with anisotropic coherence
    # Higher coherence along primary direction, lower orthogonal to it
    # NEW: Enhance anisotropy based on local curvature
    curvature_enhancement = 1.0 + local_curvature * 0.3
    
    # Calculate enhanced coherence values
    flow_coherence = coherence_scalar * 1.2 * curvature_enhancement  # Enhanced in flow direction
    orthogonal_coherence = coherence_scalar * 0.8  # Reduced orthogonal to flow
    
    # Combine for full tensor
    tensor = projection * flow_coherence + (np.eye(3) - projection) * orthogonal_coherence
    
    # Ensure eigenvalues in [0,1] range
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    eigenvalues = np.clip(eigenvalues, 0.0, 1.0)
    
    # Reconstruct tensor
    reconstructed = np.zeros((3, 3))
    for i in range(3):
        reconstructed += eigenvalues[i] * np.outer(eigenvectors[:, i], eigenvectors[:, i])
    
    return {
        'tensor': reconstructed,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors
    }

def update_flow_vector(
    current_flow: np.ndarray,
    relational_vectors: Dict[str, np.ndarray],
    gradients: Dict[str, float],
    curvature_gradient = None,   # Added curvature parameters
    local_curvature: float = 0.0
) -> Dict[str, Any]:
    """
    Update flow vector based on relational vectors, gradients and curvature.
    
    Args:
        current_flow: Current flow vector
        relational_vectors: Dict mapping related_id -> relational vector
        gradients: Dict mapping related_id -> gradient value
        curvature_gradient: Optional curvature gradient
        local_curvature: Local curvature value
        
    Returns:
        Dictionary with updated flow vector and related properties
    """
    # Initialize new flow contribution
    flow_contribution = np.zeros(3)
    
    # Sum flow contributions from gradients
    for rel_id, gradient in gradients.items():
        if rel_id in relational_vectors:
            # Scale relational vector by gradient
            vector = relational_vectors[rel_id]
            contribution = vector * gradient
            
            # Add to total
            flow_contribution += contribution
    
    # NEW: Add curvature gradient contribution
    # Flow is influenced by curvature gradient (like gravity)
    curvature_contribution = np.zeros(3)
    if curvature_gradient is not None and isinstance(curvature_gradient, np.ndarray):
        # Scale curvature influence by local curvature
        curvature_influence = local_curvature * 0.4
        curvature_contribution = curvature_gradient * curvature_influence
    
    # Update flow vector - weighted sum with curvature
    new_flow = 0.65 * current_flow + 0.25 * flow_contribution + 0.1 * curvature_contribution
    
    # Calculate magnitude
    flow_magnitude = np.linalg.norm(new_flow)
    
    # Calculate gradient tensor
    gradient_tensor = np.zeros((3, 3))
    for rel_id, gradient in gradients.items():
        if rel_id in relational_vectors:
            vector = relational_vectors[rel_id]
            # Outer product creates tensor contribution
            contribution = np.outer(vector, vector) * gradient
            # Add to tensor
            gradient_tensor += contribution
    
    # NEW: Add curvature to tensor
    if curvature_gradient is not None and isinstance(curvature_gradient, np.ndarray):
        # Create curvature tensor contribution
        curve_norm = np.linalg.norm(curvature_gradient)
        if curve_norm > 0.01:
            curve_dir = curvature_gradient / curve_norm
            curvature_tensor = np.outer(curve_dir, curve_dir) * local_curvature
            gradient_tensor += curvature_tensor * 0.3
    
    # Calculate divergence (trace of tensor)
    divergence = np.trace(gradient_tensor)
    
    # Calculate curvature effect on divergence
    # Higher curvature creates more convergent flow (negative divergence)
    if local_curvature > 0.3:
        curvature_divergence = -local_curvature * 0.2
        divergence += curvature_divergence
    
    return {
        'flow_vector': new_flow,
        'flow_magnitude': flow_magnitude,
        'gradient_tensor': gradient_tensor,
        'flow_divergence': divergence,
        'curvature_contribution': curvature_contribution.tolist() if isinstance(curvature_contribution, np.ndarray) else curvature_contribution
    }

def update_relational_neighborhood(
    relations: Dict[str, float],
    all_grain_ids: Set[str],
    self_id: str,
    threshold: float = 0.5,
    is_superposition: bool = False,
    local_curvature: float = 0.0  # Added curvature parameter
) -> Set[str]:
    """
    Update the relational neighborhood based on relation strengths and curvature.
    
    Args:
        relations: Dict mapping related_id -> relation_strength
        all_grain_ids: Set of all grain IDs
        self_id: ID of the grain being processed
        threshold: Relation strength threshold for neighborhood inclusion
        is_superposition: Whether grain is in superposition
        local_curvature: Local curvature value
        
    Returns:
        Set of neighbor grain IDs
    """
    # For superposition, adjust threshold - zero equals infinite potential
    effective_threshold = threshold
    if is_superposition:
        effective_threshold *= 0.5  # Lower threshold for superposition
    
    # NEW: Adjust threshold based on curvature
    # Higher curvature extends neighborhood (like gravity extends influence)
    if local_curvature > 0.2:
        curvature_factor = local_curvature * 0.3
        effective_threshold = max(0.1, effective_threshold - curvature_factor)
    
    neighborhood = set()
    for grain_id in all_grain_ids:
        if grain_id == self_id:
            continue
        
        # Add to neighborhood if relation is strong enough
        if grain_id in relations and abs(relations[grain_id]) >= effective_threshold:
            neighborhood.add(grain_id)
    
    return neighborhood

def resolve_from_superposition(
    awareness_level: float,
    polarity: float,
    local_curvature: float = 0.0,  # Added curvature parameter
    curvature_gradient = None      # Added curvature gradient
) -> Dict[str, Any]:
    """
    Calculate the values to resolve from superposition to a defined state.
    Incorporates curvature effects on resolution direction.
    
    Args:
        awareness_level: Awareness level to emerge with (-1.0 to 1.0)
        polarity: Current polarity value
        local_curvature: Local curvature value
        curvature_gradient: Curvature gradient vector
        
    Returns:
        Dictionary with awareness, polarity, and coherence values
    """
    # Derive awareness from polarity if not provided
    if awareness_level is None:
        # Awareness emerges from inherent polarity direction
        awareness_level = polarity * 0.2  # Scaled from polarity
    
    # NEW: Apply curvature bias to resolution direction
    # Curvature can bias resolution like gravity biases falling objects
    if local_curvature > 0.1 and curvature_gradient is not None:
        if isinstance(curvature_gradient, np.ndarray) and np.linalg.norm(curvature_gradient) > 0.01:
            # Extract z-component as awareness bias (assuming z maps to awareness)
            curve_dir = curvature_gradient / np.linalg.norm(curvature_gradient)
            curvature_bias = curve_dir[2] * local_curvature * 0.3
            
            # Apply to awareness level
            awareness_level += curvature_bias
    
    # Ensure non-zero value
    if abs(awareness_level) < 0.05:
        awareness_level = 0.05 * math.copysign(1.0, awareness_level or 1.0)
    
    # NEW: Apply curvature influence to polarity
    # Higher curvature enhances polarity bias
    if local_curvature > 0.2:
        # Boost polarity in its current direction
        polarity_boost = local_curvature * 0.2
        if abs(polarity) < 0.1:
            # For near-zero polarity, bias slightly positive
            # This models how gravity tends to create structure
            polarity = 0.1 + polarity_boost
        else:
            # Enhance existing polarity
            polarity = polarity * (1.0 + polarity_boost)
    
    # Adjust phase coherence with slight directional bias
    # NEW: Higher curvature increases initial coherence
    base_coherence = 0.8 + random.random() * 0.2
    curvature_coherence = 0.0
    if local_curvature > 0.1:
        curvature_coherence = local_curvature * 0.2
    
    # Combined coherence (capped)
    coherence = min(1.0, base_coherence + curvature_coherence)
    
    return {
        'awareness': awareness_level,
        'polarity': polarity,
        'coherence': coherence,
        'resolution': True
    }

def calculate_curvature_effects(
    collapse_strength: float,
    source_id: str = None,
    relational_vectors: Dict[str, np.ndarray] = None,
    flow_vector = None,
    ancestry_count: int = 0,
    ancestry_depth: int = 0,
    ancestry_breadth: int = 0,
    existing_curvature: float = 0.0,  # Added existing curvature parameter
    existing_gradient = None          # Added existing gradient parameter
) -> Dict[str, Any]:
    """
    Calculate curvature effects from a collapse event.
    
    Args:
        collapse_strength: Signed collapse strength
        source_id: Optional source grain ID
        relational_vectors: Optional dict mapping related_id -> relational vector
        flow_vector: Optional flow vector
        ancestry_count: Count of ancestors
        ancestry_depth: Depth of ancestry tree
        ancestry_breadth: Breadth of ancestry tree
        existing_curvature: Existing local curvature
        existing_gradient: Existing curvature gradient
        
    Returns:
        Dictionary with curvature values
    """
    # Base curvature from collapse
    base_curvature = abs(collapse_strength) * 0.1
    
    # Initial curvature gradient - preserve existing gradient if provided
    curvature_gradient = np.zeros(3)
    if existing_gradient is not None and isinstance(existing_gradient, np.ndarray):
        # Start with existing gradient but reduce its influence
        curvature_gradient = existing_gradient * 0.8
    
    # Calculate new curvature gradient contribution
    gradient_contribution = np.zeros(3)
    if source_id and relational_vectors and source_id in relational_vectors:
        # Curvature changes along relation vector
        vector = relational_vectors[source_id]
        
        # Scale by collapse strength
        gradient_contribution = vector * collapse_strength * 0.2
    elif flow_vector is not None and np.linalg.norm(flow_vector) > 0.01:
        # No clear direction, use existing flow vector
        flow_normalized = flow_vector / np.linalg.norm(flow_vector)
        gradient_contribution = flow_normalized * collapse_strength * 0.1
    
    # Add new gradient contribution
    curvature_gradient += gradient_contribution
    
    # Update ancestry curvature - this cumulates over generations
    # This implements how "mass creates spacetime curvature"
    ancestry_curvature = 0.0
    if ancestry_count > 1:
        # More complex ancestry creates more curvature
        ancestry_factor = min(0.5, ancestry_count * 0.05)
        
        # Topology adds to curvature
        if ancestry_depth > 0 and ancestry_breadth > 0:
            topology_factor = min(0.5, (ancestry_depth * 0.05 + ancestry_breadth * 0.03))
        else:
            topology_factor = 0.0
            
        ancestry_curvature = ancestry_factor + topology_factor * 0.5
    
    # Calculate new total local curvature
    # Blend existing curvature with new contributions
    new_local_curvature = existing_curvature * 0.8  # Decay existing curvature slowly
    new_local_curvature += base_curvature + ancestry_curvature * 0.5
    
    # Cap maximum curvature
    local_curvature = min(0.9, new_local_curvature)
    
    # For negative collapse (decay), reduce curvature
    if collapse_strength < 0:
        decay_factor = abs(collapse_strength) * 0.3
        local_curvature = max(0.0, local_curvature - decay_factor)
        
        # Decay also reduces gradient (flattens spacetime)
        if np.linalg.norm(curvature_gradient) > 0.01:
            decay_gradient = abs(collapse_strength) * 0.4
            curvature_gradient *= (1.0 - decay_gradient)
    
    return {
        'local_curvature': local_curvature,
        'curvature_gradient': curvature_gradient,
        'ancestry_curvature': ancestry_curvature,
        'base_curvature': base_curvature,
        'gradient_contribution': gradient_contribution.tolist() if isinstance(gradient_contribution, np.ndarray) else gradient_contribution
    }