"""
Enhanced Torus Simulation Visualizer for Collapse Geometry Framework

This visualizer is designed to represent the full richness of the Collapse Geometry
framework, focusing on:
1. Field-centric visualization (awareness, phase, polarity, tension fields)
2. Ancestry and recursive collapse visualization
3. Emergent structural patterns (vortices, lightlike pathways)
4. Backflow and curvature dynamics
5. Multi-scale visualization capabilities

Rather than treating the torus as a simple spatial container, this visualizer
represents it as the emergent manifestation of relational dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import time
import math
from typing import Dict, List, Tuple, Set, Optional, Any, Union

class EnhancedTorusVisualizer:
    """
    Advanced visualizer for the Collapse Geometry framework that prioritizes
    relational dynamics, ancestry structures, and emergent field patterns.
    """
    
    def __init__(self, resolution=100, field_smoothing=2.0):
        """
        Initialize the enhanced visualizer.
        
        Args:
            resolution: Resolution of field visualization grid
            field_smoothing: Smoothing factor for field visualization
        """
        # Core parameters
        self.resolution = resolution
        self.field_smoothing = field_smoothing
        self.manifold = None
        
        # Field visualization data
        self._initialize_fields()
        
        # Grain data
        self.grains = {}
        self.grain_positions = {}
        self.relation_strengths = defaultdict(dict)
        
        # Structural tracking
        self.vortices = []
        self.lightlike_pathways = {'structure': [], 'decay': []}
        self.ancestry_network = {}
        self.recursive_patterns = []
        self.phase_domains = []
        self.void_regions = []
        
        # Animation and timing
        self.time = 0.0
        self.last_update_time = time.time()
        self.animation_time = 0.0
    
    def _initialize_fields(self):
        """Initialize visualization field grids"""
        # Create toroidal coordinate grid
        self._theta_grid, self._phi_grid = np.meshgrid(
            np.linspace(0, 2*np.pi, self.resolution),
            np.linspace(0, 2*np.pi, self.resolution)
        )
        
        # Create field data arrays
        self._awareness_field = np.zeros((self.resolution, self.resolution))
        self._phase_field = np.zeros((self.resolution, self.resolution))
        self._polarity_field = np.zeros((self.resolution, self.resolution))
        self._tension_field = np.zeros((self.resolution, self.resolution))
        self._curvature_field = np.zeros((self.resolution, self.resolution))
        self._backflow_field = np.zeros((self.resolution, self.resolution))
        self._void_field = np.zeros((self.resolution, self.resolution))
        self._coherence_field = np.zeros((self.resolution, self.resolution))
        self._ancestry_field = np.zeros((self.resolution, self.resolution))
        
        # Vector fields for flow visualization
        self._vector_field_theta = np.zeros((self.resolution, self.resolution))
        self._vector_field_phi = np.zeros((self.resolution, self.resolution))
        
        # Create specialized colormaps
        self._initialize_colormaps()
    
    def _initialize_colormaps(self):
        """Create specialized colormaps for different field visualizations"""
        # Custom field-appropriate colormaps
        self.colormaps = {
            'awareness': self._create_awareness_colormap(),
            'phase': cm.hsv,
            'polarity': self._create_polarity_colormap(),
            'tension': cm.magma,
            'curvature': self._create_curvature_colormap(),
            'backflow': cm.inferno,
            'void': cm.binary,
            'coherence': self._create_coherence_colormap(),
            'ancestry': self._create_ancestry_colormap()
        }
    
    def _create_awareness_colormap(self):
        """Create custom colormap for awareness field"""
        return mcolors.LinearSegmentedColormap.from_list(
            'awareness_cmap',
            [(0.0, '#081B41'),  # Deep blue for low awareness
             (0.4, '#085CA9'),  # Medium blue
             (0.7, '#8A64D6'),  # Purple for medium awareness
             (0.9, '#FFB05C'),  # Orange
             (1.0, '#FFE74C')], # Yellow for high awareness
            N=256
        )
    
    def _create_polarity_colormap(self):
        """Create custom colormap for polarity field"""
        return mcolors.LinearSegmentedColormap.from_list(
            'polarity_cmap',
            [(0.0, '#9B2226'),  # Dark red for decay (negative)
             (0.3, '#E07A5F'),  # Light red
             (0.5, '#F2F2F2'),  # White for neutral
             (0.7, '#81B29A'),  # Light blue
             (1.0, '#3D5A80')], # Dark blue for structure (positive)
            N=256
        )
    
    def _create_curvature_colormap(self):
        """Create custom colormap for curvature field"""
        return mcolors.LinearSegmentedColormap.from_list(
            'curvature_cmap',
            [(0.0, '#FFFFFF'),  # White for no curvature
             (0.3, '#FEC5BB'),  # Light pink
             (0.6, '#E882A1'),  # Medium pink
             (0.8, '#9F5296'),  # Purple
             (1.0, '#673AB7')], # Deep purple for high curvature
            N=256
        )
    
    def _create_coherence_colormap(self):
        """Create custom colormap for coherence field"""
        return mcolors.LinearSegmentedColormap.from_list(
            'coherence_cmap',
            [(0.0, '#F0F0F0'),  # Light gray for low coherence
             (0.4, '#C2F0C2'),  # Light green
             (0.7, '#7AC17A'),  # Medium green
             (0.9, '#3E8E3E'),  # Dark green
             (1.0, '#1B501B')], # Very dark green for high coherence
            N=256
        )
    
    def _create_ancestry_colormap(self):
        """Create custom colormap for ancestry field"""
        return mcolors.LinearSegmentedColormap.from_list(
            'ancestry_cmap',
            [(0.0, '#F8F9FA'),  # White for no ancestry
             (0.3, '#CDB4DB'),  # Light purple
             (0.6, '#FF99C8'),  # Pink
             (0.8, '#FCB162'),  # Orange
             (1.0, '#FDCB58')], # Yellow for strong ancestry
            N=256
        )
    
    def update_state(self, manifold, state=None):
        """
        Update visualizer state from manifold.
        
        Args:
            manifold: RelationalManifold to visualize
            state: Optional simulation state object
        """
        # Store manifold reference
        self.manifold = manifold
        
        # Update timing data
        current_time = time.time()
        self.animation_time += (current_time - self.last_update_time) * 0.2
        self.last_update_time = current_time
        self.time = manifold.time if hasattr(manifold, 'time') else 0.0
        
        # Extract core manifold data
        self._extract_grain_data(manifold)
        self._extract_field_data(manifold)
        self._extract_structural_patterns(manifold)
        self._extract_ancestry_networks(manifold)
        
        # Process field data
        self._process_field_data()
        
        # Incorporate state data if provided
        if state:
            self._incorporate_state_data(state)
    
    def _extract_grain_data(self, manifold):
        """Extract grain-level data for visualization"""
        # Reset collections
        self.grains = {}
        self.grain_positions = {}
        self.relation_strengths = defaultdict(dict)
        
        # Process each grain
        for grain_id, grain in manifold.grains.items():
            # Track grain metadata for visualization
            self.grains[grain_id] = {
                'awareness': grain.awareness,
                'saturation': grain.grain_saturation,
                'activation': grain.grain_activation,
                'collapse_metric': grain.collapse_metric,
                'polarity': getattr(grain, 'polarity', 0.0),
                'ancestry': getattr(grain, 'ancestry', set()).copy(),
                'is_superposition': grain.is_in_superposition(),
                'degrees_of_freedom': grain.degrees_of_freedom
            }
            
            # Determine position on torus
            position = self._get_grain_position(grain, manifold, grain_id)
            self.grain_positions[grain_id] = position
            self.grains[grain_id]['position'] = position
            
            # Extract relations
            if hasattr(grain, 'relations'):
                for related_id, relation_strength in grain.relations.items():
                    # Store numeric strength value
                    if isinstance(relation_strength, (int, float)):
                        strength = relation_strength
                    elif hasattr(relation_strength, 'relation_strength'):
                        strength = relation_strength.relation_strength
                    else:
                        strength = 0.5  # Default value
                    
                    self.relation_strengths[grain_id][related_id] = strength
    
    def _get_grain_position(self, grain, manifold, grain_id):
        """Get grain position on the torus (theta, phi)"""
        # Try different ways to determine position based on available data
        
        # Try coordinator first
        if hasattr(manifold, 'toroidal_coordinator'):
            coordinator = manifold.toroidal_coordinator
            
            # Try explicit grain positions
            if hasattr(coordinator, 'grain_positions') and grain_id in coordinator.grain_positions:
                return coordinator.grain_positions[grain_id]
            
            # Try coordinator synchronization
            if hasattr(coordinator, 'synchronize_coordinates'):
                coordinator.synchronize_coordinates(grain_id)
        
        # Try config space
        if hasattr(manifold, 'config_space'):
            point = manifold.config_space.get_point(grain_id)
            if point and hasattr(point, 'get_toroidal_coordinates'):
                return point.get_toroidal_coordinates()
        
        # Try direct grain properties
        if hasattr(grain, 'theta') and hasattr(grain, 'phi'):
            return (grain.theta, grain.phi)
        
        # Fallback: generate position from relations
        return self._calculate_position_from_relations(grain_id, manifold)
    
    def _calculate_position_from_relations(self, grain_id, manifold):
        """Calculate position from relational structure if coordinates unavailable"""
        grain = manifold.grains[grain_id]
        polarity = getattr(grain, 'polarity', 0.0)
        
        # Base position on polarity, awareness, and grain properties
        # This creates a stable positioning system that maintains consistency
        # For deterministic pseudo-randomness, use ID as seed
        id_hash = sum(ord(c) * (i+1) for i, c in enumerate(grain_id)) % 997
        id_phase = (id_hash / 997) * 2 * np.pi
        
        # Map polarity to theta (structure = low theta, decay = high theta)
        theta_polarity_component = (1.0 - (polarity + 1.0) / 2.0) * 2 * np.pi
        theta_random_component = id_phase
        theta = (theta_polarity_component * 0.7 + theta_random_component * 0.3) % (2 * np.pi)
        
        # Map awareness and activation to phi
        phi_awareness = grain.awareness * np.pi
        phi_activation = grain.grain_activation * np.pi
        phi_random = id_phase / 2
        
        # Combine components
        phi = (phi_awareness * 0.4 + phi_activation * 0.4 + phi_random * 0.2) % (2 * np.pi)
        
        return (theta, phi)
    
    def _extract_field_data(self, manifold):
        """Extract field-level data for visualization"""
        # Reset field arrays
        self._awareness_field.fill(0)
        self._phase_field.fill(0)
        self._polarity_field.fill(0)
        self._tension_field.fill(0)
        self._curvature_field.fill(0)
        self._backflow_field.fill(0)
        self._void_field.fill(0)
        self._coherence_field.fill(0)
        self._ancestry_field.fill(0)
        
        # Base field strength for normalization
        base_field_strength = len(self.grains) * 0.01
        
        # Process each grain to build field data
        for grain_id, grain_data in self.grains.items():
            theta, phi = grain_data['position']
            
            # Get radius of influence based on awareness and activation
            radius_factor = 0.1 + 0.3 * grain_data['awareness'] + 0.1 * grain_data['activation']
            radius = max(2, int(self.resolution * radius_factor))
            
            # Apply to field at grid points
            for i in range(self.resolution):
                for j in range(self.resolution):
                    grid_theta = self._theta_grid[i, j]
                    grid_phi = self._phi_grid[i, j]
                    
                    # Calculate distance on torus (accounting for wrapping)
                    d_theta = min(abs(grid_theta - theta), 2*np.pi - abs(grid_theta - theta))
                    d_phi = min(abs(grid_phi - phi), 2*np.pi - abs(grid_phi - phi))
                    distance = np.sqrt(d_theta**2 + d_phi**2)
                    
                    # Calculate influence based on distance
                    if distance <= radius * 2 * np.pi / self.resolution:
                        # Calculate weight based on distance (gaussian-like)
                        weight = np.exp(-(distance**2) / (2 * (radius * 2 * np.pi / self.resolution / 2)**2))
                        
                        # Apply weighted grain properties to fields
                        self._awareness_field[i, j] += grain_data['awareness'] * weight
                        self._polarity_field[i, j] += grain_data['polarity'] * weight
                        
                        # Phase influenced by polarity and awareness
                        phase_contribution = (grain_data['polarity'] + 1) * np.pi * weight * 0.1
                        self._phase_field[i, j] = (self._phase_field[i, j] + phase_contribution) % (2 * np.pi)
                        
                        # Curvature from collapse metric
                        self._curvature_field[i, j] += grain_data['collapse_metric'] * weight
                        
                        # Ancestry field from ancestry set size
                        ancestry_size = len(grain_data['ancestry'])
                        self._ancestry_field[i, j] += (ancestry_size / 10.0) * weight
                        
                        # Backflow field for grains with recursive ancestry
                        if grain_id in grain_data['ancestry']:
                            self._backflow_field[i, j] += 0.5 * weight
                        
                        # Void field from superposition
                        if grain_data['is_superposition']:
                            self._void_field[i, j] += 0.8 * weight
                        
                        # Coherence field from degrees of freedom
                        coherence = 1.0 - grain_data['degrees_of_freedom']
                        self._coherence_field[i, j] += coherence * weight
        
        # Normalize the fields by base strength
        if base_field_strength > 0:
            self._awareness_field /= base_field_strength
            self._polarity_field /= base_field_strength
            self._curvature_field /= base_field_strength
            self._ancestry_field /= base_field_strength
            self._backflow_field /= base_field_strength
            self._void_field /= base_field_strength
            self._coherence_field /= base_field_strength
        
        # Also incorporate manifold-level fields if available
        self._incorporate_manifold_fields(manifold)
    
    def _incorporate_manifold_fields(self, manifold):
        """Incorporate existing fields from manifold if available"""
        # Use coordinator fields if available
        if hasattr(manifold, 'toroidal_coordinator'):
            coordinator = manifold.toroidal_coordinator
            
            # Coherence field -> coherence field
            if hasattr(coordinator, 'coherence_field'):
                for pos, coherence in coordinator.coherence_field.items():
                    if isinstance(pos, tuple) and len(pos) == 2:
                        theta, phi = pos
                        
                        # Convert to indices
                        i = int((phi / (2 * np.pi)) * self.resolution) % self.resolution
                        j = int((theta / (2 * np.pi)) * self.resolution) % self.resolution
                        
                        # Update coherence field
                        self._coherence_field[i, j] = max(self._coherence_field[i, j], coherence)
            
            # Tension field
            if hasattr(coordinator, 'tension_field'):
                for pos, tension in coordinator.tension_field.items():
                    if isinstance(pos, tuple) and len(pos) == 2:
                        theta, phi = pos
                        
                        # Convert to indices
                        i = int((phi / (2 * np.pi)) * self.resolution) % self.resolution
                        j = int((theta / (2 * np.pi)) * self.resolution) % self.resolution
                        
                        # Update tension field
                        self._tension_field[i, j] = max(self._tension_field[i, j], tension)
        
        # Direct manifold metrics
        if hasattr(manifold, 'field_coherence'):
            # Apply global coherence as base level
            coherence_base = manifold.field_coherence * 0.2
            self._coherence_field += coherence_base
        
        if hasattr(manifold, 'system_tension'):
            # Apply global tension as base level
            tension_base = manifold.system_tension * 0.2
            self._tension_field += tension_base
    
    def _extract_structural_patterns(self, manifold):
        """Extract emergent structural patterns from manifold"""
        # Reset structure lists
        self.vortices = []
        self.lightlike_pathways = {'structure': [], 'decay': []}
        self.phase_domains = []
        self.void_regions = []
        
        # Get vortices from coordinator if available
        if hasattr(manifold, 'toroidal_coordinator'):
            coordinator = manifold.toroidal_coordinator
            
            # Extract vortices
            if hasattr(coordinator, 'vortices'):
                self.vortices = coordinator.vortices.copy() if coordinator.vortices else []
            
            # Extract lightlike pathways
            if hasattr(coordinator, 'lightlike_pathways'):
                for pathway_type in ['structure', 'decay']:
                    if pathway_type in coordinator.lightlike_pathways:
                        pathways = coordinator.lightlike_pathways[pathway_type]
                        self.lightlike_pathways[pathway_type] = pathways.copy() if pathways else []
        
        # Extract void regions if available
        if hasattr(manifold, 'void_formation_events'):
            self.void_regions = manifold.void_formation_events[-10:] if manifold.void_formation_events else []
    
    def _extract_ancestry_networks(self, manifold):
        """Extract ancestry network data for visualization"""
        # Build ancestry network
        self.ancestry_network = {}
        self.recursive_patterns = []
        
        for grain_id, grain_data in self.grains.items():
            ancestry = grain_data['ancestry']
            
            # Skip empty ancestry
            if not ancestry:
                continue
            
            # Track ancestry connections
            self.ancestry_network[grain_id] = []
            
            for ancestor_id in ancestry:
                # Make sure ancestor exists in current grains
                if ancestor_id in self.grains:
                    self.ancestry_network[grain_id].append(ancestor_id)
            
            # Detect recursive patterns (self-reference)
            if grain_id in ancestry:
                self.recursive_patterns.append({
                    'grain_id': grain_id,
                    'position': grain_data['position'],
                    'strength': grain_data['collapse_metric']
                })
    
    def _incorporate_state_data(self, state):
        """
        Incorporate data from simulation state if provided.
        
        Args:
            state: SimulationState object
        """
        # If state has visualization fields, incorporate them
        if hasattr(state, '_awareness_field') and state._awareness_field is not None:
            # If the resolutions match, use directly
            if state._awareness_field.shape == self._awareness_field.shape:
                self._awareness_field = np.maximum(self._awareness_field, state._awareness_field)
                self._phase_field = state._phase_field
                
                if hasattr(state, '_polarity_field'):
                    self._polarity_field = np.maximum(self._polarity_field, state._polarity_field)
                
                if hasattr(state, '_tension_field'):
                    self._tension_field = np.maximum(self._tension_field, state._tension_field)
                
                if hasattr(state, '_backflow_field'):
                    self._backflow_field = np.maximum(self._backflow_field, state._backflow_field)
    
    def _process_field_data(self):
        """Process and normalize field data for visualization"""
        # Apply smoothing to fields
        self._apply_field_smoothing()
        
        # Normalize fields to [0,1] range
        self._normalize_fields()
        
        # Calculate vector fields
        self._calculate_vector_fields()
    
    def _apply_field_smoothing(self):
        """Apply smoothing to field data"""
        try:
            # Use scipy for better smoothing if available
            from scipy.ndimage import gaussian_filter
            
            self._awareness_field = gaussian_filter(self._awareness_field, sigma=self.field_smoothing)
            self._polarity_field = gaussian_filter(self._polarity_field, sigma=self.field_smoothing)
            self._tension_field = gaussian_filter(self._tension_field, sigma=self.field_smoothing)
            self._curvature_field = gaussian_filter(self._curvature_field, sigma=self.field_smoothing)
            self._backflow_field = gaussian_filter(self._backflow_field, sigma=self.field_smoothing)
            self._void_field = gaussian_filter(self._void_field, sigma=self.field_smoothing)
            self._coherence_field = gaussian_filter(self._coherence_field, sigma=self.field_smoothing)
            self._ancestry_field = gaussian_filter(self._ancestry_field, sigma=self.field_smoothing)
            
            # Phase field needs special handling for circular data
            phase_x = np.cos(self._phase_field)
            phase_y = np.sin(self._phase_field)
            
            phase_x_smooth = gaussian_filter(phase_x, sigma=self.field_smoothing)
            phase_y_smooth = gaussian_filter(phase_y, sigma=self.field_smoothing)
            
            self._phase_field = np.arctan2(phase_y_smooth, phase_x_smooth) % (2 * np.pi)
            
        except ImportError:
            # Simple rolling average if scipy not available
            for _ in range(int(self.field_smoothing * 2)):
                self._simple_smooth(self._awareness_field)
                self._simple_smooth(self._polarity_field)
                self._simple_smooth(self._tension_field)
                self._simple_smooth(self._curvature_field)
                self._simple_smooth(self._backflow_field)
                self._simple_smooth(self._void_field)
                self._simple_smooth(self._coherence_field)
                self._simple_smooth(self._ancestry_field)
                
                # Phase field needs special handling for circular data
                phase_x = np.cos(self._phase_field)
                phase_y = np.sin(self._phase_field)
                
                self._simple_smooth(phase_x)
                self._simple_smooth(phase_y)
                
                self._phase_field = np.arctan2(phase_y, phase_x) % (2 * np.pi)
    
    def _simple_smooth(self, field):
        """
        Apply a simple smoothing operation to a field.
        
        Args:
            field: 2D numpy array field
        """
        # Create a copy to avoid modifying while iterating
        orig = field.copy()
        rows, cols = field.shape
        
        for i in range(rows):
            for j in range(cols):
                # Get values with wrapping (torus)
                values = [
                    orig[i, j],
                    orig[(i-1) % rows, j],
                    orig[(i+1) % rows, j],
                    orig[i, (j-1) % cols],
                    orig[i, (j+1) % cols]
                ]
                
                # Set to average
                field[i, j] = sum(values) / len(values)
    
    def _normalize_fields(self):
        """Normalize field values to [0,1] range"""
        for field in [
            self._awareness_field, 
            self._polarity_field,
            self._tension_field,
            self._curvature_field,
            self._backflow_field,
            self._void_field,
            self._coherence_field,
            self._ancestry_field
        ]:
            if np.max(field) > np.min(field):
                field[:] = (field - np.min(field)) / (np.max(field) - np.min(field))
    
    def _calculate_vector_fields(self):
        """Calculate vector fields for flow visualization"""
        # Calculate combined vector field from polarity and phase gradients
        
        # Polarity gradient
        dx_polarity, dy_polarity = np.gradient(self._polarity_field)
        
        # Phase gradient (needs circular handling)
        dx_phase, dy_phase = np.zeros_like(self._phase_field), np.zeros_like(self._phase_field)
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                # Calculate phase differences with wrapping
                i_prev = (i - 1) % self.resolution
                i_next = (i + 1) % self.resolution
                j_prev = (j - 1) % self.resolution
                j_next = (j + 1) % self.resolution
                
                # X direction
                phase_diff_x = self._phase_field[i, j_next] - self._phase_field[i, j_prev]
                if phase_diff_x > np.pi:
                    phase_diff_x -= 2 * np.pi
                elif phase_diff_x < -np.pi:
                    phase_diff_x += 2 * np.pi
                
                # Y direction
                phase_diff_y = self._phase_field[i_next, j] - self._phase_field[i_prev, j]
                if phase_diff_y > np.pi:
                    phase_diff_y -= 2 * np.pi
                elif phase_diff_y < -np.pi:
                    phase_diff_y += 2 * np.pi
                
                dx_phase[i, j] = phase_diff_x / 2
                dy_phase[i, j] = phase_diff_y / 2
        
        # Combine gradients with weights
        dx_combined = dx_polarity * 0.6 + dx_phase * 0.4
        dy_combined = dy_polarity * 0.6 + dy_phase * 0.4
        
        # Calculate magnitudes
        magnitudes = np.sqrt(dx_combined**2 + dy_combined**2)
        
        # Normalize vectors
        mask = magnitudes > 0.01
        self._vector_field_theta[mask] = dx_combined[mask] / magnitudes[mask]
        self._vector_field_phi[mask] = dy_combined[mask] / magnitudes[mask]
        
        # Scale by field intensity
        intensity = 0.3 + 0.7 * (self._awareness_field + self._polarity_field) / 2
        self._vector_field_theta *= intensity
        self._vector_field_phi *= intensity
    
    def render_torus_3d(self, **kwargs):
        """
        Render a 3D torus visualization.
        
        Args:
            **kwargs: Additional parameters including:
                - color_by: Field to color by ('awareness', 'phase', 'polarity', etc.)
                - show_grains: Whether to show individual grains
                - show_relations: Whether to show relations between grains
                - show_ancestry: Whether to show ancestry relationships
                - show_vortices: Whether to show vortices
                - show_pathways: Whether to show lightlike pathways
                - show_recursion: Whether to show recursive patterns
                - major_radius: Major radius of torus
                - minor_radius: Minor radius of torus
                - view_angle: (elevation, azimuth) viewing angle
                - alpha: Transparency of torus surface
        
        Returns:
            Figure and axes objects
        """
        # Get parameters
        color_by = kwargs.get('color_by', 'awareness')
        show_grains = kwargs.get('show_grains', True)
        show_relations = kwargs.get('show_relations', False)
        show_ancestry = kwargs.get('show_ancestry', True)
        show_vortices = kwargs.get('show_vortices', True)
        show_pathways = kwargs.get('show_pathways', True)
        show_recursion = kwargs.get('show_recursion', True)
        show_vectors = kwargs.get('show_vectors', False)
        major_radius = kwargs.get('major_radius', 3.0)
        minor_radius = kwargs.get('minor_radius', 1.0)
        view_angle = kwargs.get('view_angle', (30, 45))
        surface_alpha = kwargs.get('alpha', 0.7)
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create 3D torus mesh
        torus_x, torus_y, torus_z = self._create_torus_mesh(major_radius, minor_radius)
        
        # Get field for coloring
        if color_by == 'awareness':
            field_data = self._awareness_field
            colormap = self.colormaps['awareness']
            field_label = 'Awareness Field'
        elif color_by == 'phase':
            field_data = self._phase_field / (2 * np.pi)  # Normalize to [0,1]
            colormap = self.colormaps['phase']
            field_label = 'Phase Field'
        elif color_by == 'polarity':
            # Transform polarity from [-1,1] to [0,1]
            field_data = (self._polarity_field + 1) / 2
            colormap = self.colormaps['polarity']
            field_label = 'Polarity Field'
        elif color_by == 'tension':
            field_data = self._tension_field
            colormap = self.colormaps['tension']
            field_label = 'Tension Field'
        elif color_by == 'curvature':
            field_data = self._curvature_field
            colormap = self.colormaps['curvature']
            field_label = 'Curvature Field'
        elif color_by == 'backflow':
            field_data = self._backflow_field
            colormap = self.colormaps['backflow']
            field_label = 'Backflow Field'
        elif color_by == 'void':
            field_data = self._void_field
            colormap = self.colormaps['void']
            field_label = 'Void Field'
        elif color_by == 'coherence':
            field_data = self._coherence_field
            colormap = self.colormaps['coherence']
            field_label = 'Coherence Field'
        elif color_by == 'ancestry':
            field_data = self._ancestry_field
            colormap = self.colormaps['ancestry']
            field_label = 'Ancestry Field'
        else:
            field_data = self._awareness_field
            colormap = self.colormaps['awareness']
            field_label = 'Awareness Field'
        
        # Plot torus surface colored by field
        torus_surface = ax.plot_surface(
            torus_x, torus_y, torus_z,
            facecolors=colormap(field_data),
            alpha=surface_alpha,
            antialiased=True,
            shade=True
        )
        
        # Draw a colorbar
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        mappable = ScalarMappable(norm=Normalize(0, 1), cmap=colormap)
        mappable.set_array(field_data)
        plt.colorbar(mappable, ax=ax, label=field_label)
        
        # Show grains if requested
        if show_grains:
            self._draw_grains(ax, major_radius, minor_radius)
        
        # Show relations if requested
        if show_relations:
            self._draw_relations(ax, major_radius, minor_radius)
        
        # Show ancestry if requested
        if show_ancestry:
            self._draw_ancestry(ax, major_radius, minor_radius)
        
        # Show vortices if requested
        if show_vortices:
            self._draw_vortices(ax, major_radius, minor_radius)
        
        # Show lightlike pathways if requested
        if show_pathways:
            self._draw_pathways(ax, major_radius, minor_radius)
        
        # Show recursion patterns if requested
        if show_recursion:
            self._draw_recursion_patterns(ax, major_radius, minor_radius)
        
        # Show vector field if requested
        if show_vectors:
            self._draw_vector_field_3d(ax, major_radius, minor_radius)
        
        # Configure view
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Set axis labels and scale
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        
        # Hide axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Add title
        plt.title(f'Collapse Geometry Torus Visualization - {field_label} (t={self.time:.2f})')
        
        return fig, ax
    
    def _create_torus_mesh(self, major_radius, minor_radius):
        """
        Create a 3D torus mesh.
        
        Args:
            major_radius: Major radius of torus
            minor_radius: Minor radius of torus
            
        Returns:
            Tuple of (x, y, z) mesh arrays
        """
        # Calculate 3D torus coordinates from toroidal coordinates
        torus_x = np.zeros((self.resolution, self.resolution))
        torus_y = np.zeros((self.resolution, self.resolution))
        torus_z = np.zeros((self.resolution, self.resolution))
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                phi = self._phi_grid[i, j]
                theta = self._theta_grid[i, j]
                
                torus_x[i, j] = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
                torus_y[i, j] = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
                torus_z[i, j] = minor_radius * np.sin(phi)
        
        return torus_x, torus_y, torus_z
    
    def _toroidal_to_cartesian(self, theta, phi, major_radius, minor_radius):
        """
        Convert toroidal coordinates to Cartesian.
        
        Args:
            theta: Angular coordinate around tube center
            phi: Angular coordinate around tube
            major_radius: Major radius of torus
            minor_radius: Minor radius of torus
            
        Returns:
            Tuple of (x, y, z) coordinates
        """
        x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
        y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
        z = minor_radius * np.sin(phi)
        
        return x, y, z
    
    def _draw_grains(self, ax, major_radius, minor_radius):
        """
        Draw grain representations on the torus.
        
        Args:
            ax: 3D axes to draw into
            major_radius: Major radius of torus
            minor_radius: Minor radius of torus
        """
        # First collect grains by type for better performance
        structure_grains = []
        decay_grains = []
        neutral_grains = []
        superposition_grains = []
        
        for grain_id, grain_data in self.grains.items():
            # Get position
            theta, phi = grain_data['position']
            x, y, z = self._toroidal_to_cartesian(theta, phi, major_radius, minor_radius)
            
            # Size based on awareness and activation
            size = 20 + 100 * grain_data['awareness'] * grain_data['activation']
            
            # Get polarity
            polarity = grain_data['polarity']
            
            # Adjust opacity based on saturation
            alpha = 0.3 + 0.7 * grain_data['saturation']
            
            # Categorize grain
            if grain_data['is_superposition']:
                superposition_grains.append((x, y, z, size, alpha))
            elif polarity > 0.2:
                # Structure-forming
                color_intensity = min(1.0, 0.5 + polarity/2)
                structure_grains.append((x, y, z, size, alpha, color_intensity))
            elif polarity < -0.2:
                # Structure-decaying
                color_intensity = min(1.0, 0.5 - polarity/2)
                decay_grains.append((x, y, z, size, alpha, color_intensity))
            else:
                # Neutral
                color_intensity = 0.5 + 0.5 * grain_data['awareness']
                neutral_grains.append((x, y, z, size, alpha, color_intensity))
        
        # Plot grains by type
        if structure_grains:
            xs, ys, zs, sizes, alphas, intensities = zip(*structure_grains)
            colors = [(0, 0, intensity) for intensity in intensities]  # Blues
            ax.scatter(xs, ys, zs, s=sizes, c=colors, alpha=alphas, edgecolors='black', linewidths=0.5)
        
        if decay_grains:
            xs, ys, zs, sizes, alphas, intensities = zip(*decay_grains)
            colors = [(intensity, 0, 0) for intensity in intensities]  # Reds
            ax.scatter(xs, ys, zs, s=sizes, c=colors, alpha=alphas, edgecolors='black', linewidths=0.5)
        
        if neutral_grains:
            xs, ys, zs, sizes, alphas, intensities = zip(*neutral_grains)
            colors = [(0, intensity, 0) for intensity in intensities]  # Greens
            ax.scatter(xs, ys, zs, s=sizes, c=colors, alpha=alphas, edgecolors='black', linewidths=0.5)
        
        if superposition_grains:
            xs, ys, zs, sizes, alphas = zip(*superposition_grains)
            ax.scatter(xs, ys, zs, marker='*', s=sizes, color='cyan', alpha=alphas, edgecolors='white')
    
    def _draw_relations(self, ax, major_radius, minor_radius):
        """
        Draw relations between grains.
        
        Args:
            ax: 3D axes to draw into
            major_radius: Major radius of torus
            minor_radius: Minor radius of torus
        """
        # Collect relations by type for better performance
        structure_relations = []
        decay_relations = []
        neutral_relations = []
        
        for grain_id, relations in self.relation_strengths.items():
            if grain_id not in self.grain_positions:
                continue
                
            # Get source position
            theta1, phi1 = self.grain_positions[grain_id]
            x1, y1, z1 = self._toroidal_to_cartesian(theta1, phi1, major_radius, minor_radius)
            
            # Get grain polarity
            source_polarity = self.grains[grain_id]['polarity'] if grain_id in self.grains else 0.0
            
            for related_id, strength in relations.items():
                if related_id not in self.grain_positions:
                    continue
                
                # Get target position
                theta2, phi2 = self.grain_positions[related_id]
                x2, y2, z2 = self._toroidal_to_cartesian(theta2, phi2, major_radius, minor_radius)
                
                # Avoid rendering relations that are too long (cross the torus)
                dx = x2 - x1
                dy = y2 - y1
                dz = z2 - z1
                distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                if distance > major_radius * 1.5:
                    continue
                
                # Line width based on strength
                lw = 0.5 + 2.0 * abs(strength)
                
                # Classify relation by polarity
                target_polarity = self.grains[related_id]['polarity'] if related_id in self.grains else 0.0
                combined_polarity = source_polarity + target_polarity
                
                if combined_polarity > 0.3:
                    structure_relations.append((x1, y1, z1, x2, y2, z2, lw, abs(strength)))
                elif combined_polarity < -0.3:
                    decay_relations.append((x1, y1, z1, x2, y2, z2, lw, abs(strength)))
                else:
                    neutral_relations.append((x1, y1, z1, x2, y2, z2, lw, abs(strength)))
        
        # Draw relations by type
        for x1, y1, z1, x2, y2, z2, lw, alpha in structure_relations:
            ax.plot([x1, x2], [y1, y2], [z1, z2], 
                    color='blue', linewidth=lw, alpha=min(0.6, alpha))
        
        for x1, y1, z1, x2, y2, z2, lw, alpha in decay_relations:
            ax.plot([x1, x2], [y1, y2], [z1, z2], 
                    color='red', linewidth=lw, alpha=min(0.6, alpha))
        
        for x1, y1, z1, x2, y2, z2, lw, alpha in neutral_relations:
            ax.plot([x1, x2], [y1, y2], [z1, z2], 
                    color='green', linewidth=lw, alpha=min(0.4, alpha))
    
    def _draw_ancestry(self, ax, major_radius, minor_radius):
        """
        Draw ancestry relationships on the torus.
        
        Args:
            ax: 3D axes to draw into
            major_radius: Major radius of torus
            minor_radius: Minor radius of torus
        """
        # Collect ancestry relationships for better performance
        direct_ancestry = []
        transitive_ancestry = []
        
        for grain_id, ancestors in self.ancestry_network.items():
            if grain_id not in self.grain_positions:
                continue
                
            # Get target position
            theta1, phi1 = self.grain_positions[grain_id]
            x1, y1, z1 = self._toroidal_to_cartesian(theta1, phi1, major_radius, minor_radius)
            
            for ancestor_id in ancestors:
                if ancestor_id not in self.grain_positions:
                    continue
                
                # Get source position
                theta2, phi2 = self.grain_positions[ancestor_id]
                x2, y2, z2 = self._toroidal_to_cartesian(theta2, phi2, major_radius, minor_radius)
                
                # Determine if direct parent or more distant ancestor
                if grain_id == ancestor_id:
                    # Self-reference, drawn separately
                    continue
                
                # Get strength from relation or grain data
                strength = 0.5
                if grain_id in self.relation_strengths and ancestor_id in self.relation_strengths[grain_id]:
                    strength = self.relation_strengths[grain_id][ancestor_id]
                
                # Line width based on strength and activation
                lw = 0.5 + 1.5 * strength
                
                # Skip relations that cross the torus (too long)
                dx = x2 - x1
                dy = y2 - y1
                dz = z2 - z1
                distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                if distance > major_radius * 1.5:
                    continue
                
                # Distinguish direct parents vs. distant ancestors
                if ancestor_id in self.grains and self.grains[ancestor_id]['polarity'] > 0:
                    # Structure-forming ancestors
                    direct_ancestry.append((x1, y1, z1, x2, y2, z2, lw))
                else:
                    # Other ancestors
                    transitive_ancestry.append((x1, y1, z1, x2, y2, z2, lw))
        
        # Draw direct ancestry (parent-child)
        for x1, y1, z1, x2, y2, z2, lw in direct_ancestry:
            ax.plot([x1, x2], [y1, y2], [z1, z2], 
                   color='darkblue', linewidth=lw, alpha=0.7, 
                   linestyle='-')
        
        # Draw transitive ancestry (distant ancestors) with different style
        for x1, y1, z1, x2, y2, z2, lw in transitive_ancestry:
            ax.plot([x1, x2], [y1, y2], [z1, z2], 
                   color='purple', linewidth=lw, alpha=0.5, 
                   linestyle='--')
    
    def _draw_vortices(self, ax, major_radius, minor_radius):
        """
        Draw vortices on the torus.
        
        Args:
            ax: 3D axes to draw into
            major_radius: Major radius of torus
            minor_radius: Minor radius of torus
        """
        # Skip if no vortices
        if not self.vortices:
            return
        
        for vortex in self.vortices:
            # Get position from vortex data
            if 'theta' in vortex and 'phi' in vortex:
                theta, phi = vortex['theta'], vortex['phi']
            elif 'center_id' in vortex and vortex['center_id'] in self.grain_positions:
                theta, phi = self.grain_positions[vortex['center_id']]
            else:
                continue
            
            # Get vortex properties
            strength = vortex.get('strength', 0.5)
            is_lightlike = vortex.get('is_lightlike', False)
            direction = vortex.get('direction', 'clockwise')
            polarity = vortex.get('polarity', 0.0)
            
            # Convert to cartesian coordinates
            center_x, center_y, center_z = self._toroidal_to_cartesian(theta, phi, major_radius, minor_radius)
            
            # Draw vortex marker
            marker_size = 150 + 200 * strength
            
            # Create vortex color based on polarity and lightlike nature
            if is_lightlike:
                # Lightlike vortices (ethereal)
                if direction == 'clockwise':
                    color = 'cyan'
                else:
                    color = 'magenta'
                edge_color = 'white'
            else:
                # Regular vortices
                if polarity > 0.2:
                    # Structure vortex
                    color = 'blue'
                    edge_color = 'darkblue'
                elif polarity < -0.2:
                    # Decay vortex
                    color = 'red'
                    edge_color = 'darkred'
                else:
                    # Neutral vortex
                    color = 'green'
                    edge_color = 'darkgreen'
            
            # Draw vortex marker
            ax.scatter([center_x], [center_y], [center_z], 
                       s=marker_size, color=color, alpha=0.6,
                       edgecolors=edge_color, linewidths=2)
            
            # Draw circulation indicator (spiral rings)
            self._draw_vortex_circulation(ax, theta, phi, direction, strength, polarity,
                                        major_radius, minor_radius)
    
    def _draw_vortex_circulation(self, ax, theta, phi, direction, strength, polarity,
                              major_radius, minor_radius):
        """
        Draw circulation indicator for a vortex.
        
        Args:
            ax: 3D axes to draw into
            theta: Angular position around torus
            phi: Angular position through torus
            direction: 'clockwise' or 'counterclockwise'
            strength: Vortex strength
            polarity: Vortex polarity
            major_radius: Major radius of torus
            minor_radius: Minor radius of torus
        """
        # Create a spiral path around the vortex center
        spiral_points = 30
        radius_scale = 0.15 * (0.5 + strength)
        max_radius = min(np.pi / 4, radius_scale * 2 * np.pi / 4)
        
        # Determine spiral direction
        direction_multiplier = 1 if direction == 'clockwise' else -1
        
        # Create spiral in toroidal coordinates
        spiral_theta = []
        spiral_phi = []
        
        for i in range(spiral_points):
            # Normalized angle from 0 to 2π
            angle = i / (spiral_points - 1) * 2 * np.pi
            
            # Create spiral effect with increasing radius
            spiral_radius = max_radius * (i / (spiral_points - 1))
            
            # Calculate offsets with direction
            dtheta = spiral_radius * np.cos(angle) * direction_multiplier
            dphi = spiral_radius * np.sin(angle)
            
            # Apply to center position with wrapping
            pt = (theta + dtheta) % (2 * np.pi)
            pp = (phi + dphi) % (2 * np.pi)
            
            spiral_theta.append(pt)
            spiral_phi.append(pp)
        
        # Convert to Cartesian
        spiral_x = []
        spiral_y = []
        spiral_z = []
        
        for t, p in zip(spiral_theta, spiral_phi):
            x, y, z = self._toroidal_to_cartesian(t, p, major_radius, minor_radius)
            spiral_x.append(x)
            spiral_y.append(y)
            spiral_z.append(z)
        
        # Determine color based on polarity
        if polarity > 0.2:
            # Structure vortex
            color = 'blue'
        elif polarity < -0.2:
            # Decay vortex
            color = 'red'
        else:
            # Neutral vortex
            color = 'green'
        
        # Draw spiral
        ax.plot(spiral_x, spiral_y, spiral_z, 
                color=color, linewidth=2, alpha=0.8)
        
        # Add arrow to indicate direction
        mid_idx = len(spiral_x) // 2
        if mid_idx > 0:
            arrow_x = spiral_x[mid_idx-1:mid_idx+1]
            arrow_y = spiral_y[mid_idx-1:mid_idx+1]
            arrow_z = spiral_z[mid_idx-1:mid_idx+1]
            
            # Draw arrow
            ax.quiver(arrow_x[0], arrow_y[0], arrow_z[0],
                      arrow_x[1] - arrow_x[0],
                      arrow_y[1] - arrow_y[0],
                      arrow_z[1] - arrow_z[0],
                      color=color, linewidth=3, alpha=0.9, 
                      arrow_length_ratio=0.4)
    
    def _draw_pathways(self, ax, major_radius, minor_radius):
        """
        Draw lightlike pathways on the torus.
        
        Args:
            ax: 3D axes to draw into
            major_radius: Major radius of torus
            minor_radius: Minor radius of torus
        """
        # Draw structure-forming pathways
        for pathway in self.lightlike_pathways['structure']:
            self._draw_pathway(ax, pathway, 'structure', major_radius, minor_radius)
        
        # Draw structure-decaying pathways
        for pathway in self.lightlike_pathways['decay']:
            self._draw_pathway(ax, pathway, 'decay', major_radius, minor_radius)
    
    def _draw_pathway(self, ax, pathway, pathway_type, major_radius, minor_radius):
        """
        Draw a single pathway on the torus.
        
        Args:
            ax: 3D axes to draw into
            pathway: Pathway data dictionary
            pathway_type: 'structure' or 'decay'
            major_radius: Major radius of torus
            minor_radius: Minor radius of torus
        """
        # Extract nodes from pathway
        nodes = pathway.get('nodes', [])
        
        # Skip if not enough nodes
        if len(nodes) < 2:
            return
        
        # Get node positions
        positions = []
        for node_id in nodes:
            if node_id in self.grain_positions:
                positions.append(self.grain_positions[node_id])
        
        # Skip if not enough positions
        if len(positions) < 2:
            return
        
        # Convert to Cartesian coordinates
        x_coords = []
        y_coords = []
        z_coords = []
        
        for theta, phi in positions:
            x, y, z = self._toroidal_to_cartesian(theta, phi, major_radius, minor_radius)
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
        
        # Get pathway properties
        avg_polarity = pathway.get('avg_polarity', 0.0)
        pathway_strength = pathway.get('pathway_strength', 0.5)
        
        # Line width based on strength
        lw = 1.0 + 2.0 * pathway_strength
        
        # Color based on type and polarity
        if pathway_type == 'structure':
            color = 'blue'
            zorder = 2
        else:  # decay
            color = 'red'
            zorder = 1
        
        # Alpha based on strength
        alpha = min(0.9, 0.3 + pathway_strength)
        
        # Draw pathway
        ax.plot(x_coords, y_coords, z_coords, 
               color=color, linewidth=lw, alpha=alpha, 
               zorder=zorder)
        
        # Add arrow to show direction
        if len(x_coords) > 2:
            mid_idx = len(x_coords) // 2
            ax.quiver(x_coords[mid_idx-1], y_coords[mid_idx-1], z_coords[mid_idx-1],
                     x_coords[mid_idx] - x_coords[mid_idx-1],
                     y_coords[mid_idx] - y_coords[mid_idx-1],
                     z_coords[mid_idx] - z_coords[mid_idx-1],
                     color=color, arrow_length_ratio=0.3,
                     linewidth=lw, alpha=alpha)
    
    def _draw_recursion_patterns(self, ax, major_radius, minor_radius):
        """
        Draw recursive patterns on the torus.
        
        Args:
            ax: 3D axes to draw into
            major_radius: Major radius of torus
            minor_radius: Minor radius of torus
        """
        # Skip if no recursive patterns
        if not self.recursive_patterns:
            return
        
        for pattern in self.recursive_patterns:
            # Get position
            theta, phi = pattern['position']
            grain_id = pattern['grain_id']
            
            # Get strength
            strength = pattern['strength']
            
            # Convert to Cartesian
            x, y, z = self._toroidal_to_cartesian(theta, phi, major_radius, minor_radius)
            
            # Create a self-referential loop
            radius = 0.2 * minor_radius * (0.5 + strength)
            loop_points = 30
            
            # Create a circle in local tangent plane
            # First, create a tangent vector to the torus
            tangent_theta = np.array([
                -np.sin(theta), 
                np.cos(theta), 
                0
            ])
            
            tangent_phi = np.array([
                -np.sin(phi) * np.cos(theta),
                -np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            
            # Create circle points in tangent plane
            loop_x = []
            loop_y = []
            loop_z = []
            
            for i in range(loop_points + 1):
                angle = i / loop_points * 2 * np.pi
                
                # Compute offset in tangent plane
                offset = (tangent_theta * np.cos(angle) + tangent_phi * np.sin(angle)) * radius
                
                # Add to center position
                loop_x.append(x + offset[0])
                loop_y.append(y + offset[1])
                loop_z.append(z + offset[2])
            
            # Draw loop
            ax.plot(loop_x, loop_y, loop_z, 
                   color='purple', linewidth=2.0, alpha=0.8)
            
            # Add arrow to indicate recursion direction
            mid_idx = 3 * loop_points // 4
            ax.quiver(loop_x[mid_idx-1], loop_y[mid_idx-1], loop_z[mid_idx-1],
                     loop_x[mid_idx] - loop_x[mid_idx-1],
                     loop_y[mid_idx] - loop_y[mid_idx-1],
                     loop_z[mid_idx] - loop_z[mid_idx-1],
                     color='purple', arrow_length_ratio=0.3,
                     linewidth=2, alpha=0.8)
    
    def _draw_vector_field_3d(self, ax, major_radius, minor_radius):
        """
        Draw the vector field on the torus in 3D.
        
        Args:
            ax: 3D axes to draw into
            major_radius: Major radius of torus
            minor_radius: Minor radius of torus
        """
        # Sample points for vectors (not all points for clarity)
        sample_rate = 8  # Show every 8th point for clarity
        
        # Collect vector data
        vectors = []
        
        for i in range(0, self.resolution, sample_rate):
            for j in range(0, self.resolution, sample_rate):
                theta = self._theta_grid[i, j]
                phi = self._phi_grid[i, j]
                
                # Get vector components
                v_theta = self._vector_field_theta[i, j]
                v_phi = self._vector_field_phi[i, j]
                
                # Skip weak vectors
                magnitude = np.sqrt(v_theta**2 + v_phi**2)
                if magnitude < 0.05:
                    continue
                
                # Vector scale based on field strength
                scale = 0.2 * minor_radius * magnitude
                
                # Convert position to Cartesian
                x, y, z = self._toroidal_to_cartesian(theta, phi, major_radius, minor_radius)
                
                # Get tangent directions
                tangent_theta = np.array([
                    -np.sin(theta), 
                    np.cos(theta), 
                    0
                ])
                
                tangent_phi = np.array([
                    -np.sin(phi) * np.cos(theta),
                    -np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ])
                
                # Combine to get 3D vector direction
                direction = tangent_theta * v_theta + tangent_phi * v_phi
                
                # Calculate endpoint
                end_x = x + direction[0] * scale
                end_y = y + direction[1] * scale
                end_z = z + direction[2] * scale
                
                # Determine color from polarity field
                polarity = self._polarity_field[i, j]
                
                if polarity > 0.1:
                    # Structure-forming (blue)
                    color = 'blue'
                elif polarity < -0.1:
                    # Structure-decaying (red)
                    color = 'red'
                else:
                    # Neutral (green)
                    color = 'green'
                
                # Store vector data
                vectors.append((x, y, z, direction[0] * scale, direction[1] * scale, direction[2] * scale, color))
        
        # Draw all vectors in one call for efficiency
        for x, y, z, dx, dy, dz, color in vectors:
            ax.quiver(x, y, z, dx, dy, dz, color=color, alpha=0.7, 
                     arrow_length_ratio=0.3, linewidth=1.5)
    
    def create_unwrapped_visualization(self, **kwargs):
        """
        Create a 2D unwrapped visualization of the torus.
        
        Args:
            **kwargs: Additional parameters including:
                - color_by: Field to color by ('awareness', 'phase', 'polarity', etc.)
                - show_grains: Whether to show individual grains
                - show_relations: Whether to show relations between grains
                - show_ancestry: Whether to show ancestry relationships
                - show_vortices: Whether to show vortices
                - show_pathways: Whether to show lightlike pathways
                - show_vectors: Whether to show vector field
        
        Returns:
            Figure and axes objects
        """
        # Get parameters
        color_by = kwargs.get('color_by', 'awareness')
        show_grains = kwargs.get('show_grains', True)
        show_relations = kwargs.get('show_relations', False)
        show_ancestry = kwargs.get('show_ancestry', True)
        show_vortices = kwargs.get('show_vortices', True)
        show_pathways = kwargs.get('show_pathways', True)
        show_vectors = kwargs.get('show_vectors', True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get field for coloring
        if color_by == 'awareness':
            field_data = self._awareness_field
            colormap = self.colormaps['awareness']
            field_label = 'Awareness Field'
        elif color_by == 'phase':
            field_data = self._phase_field / (2 * np.pi)  # Normalize to [0,1]
            colormap = self.colormaps['phase']
            field_label = 'Phase Field'
        elif color_by == 'polarity':
            # Transform polarity from [-1,1] to [0,1]
            field_data = (self._polarity_field + 1) / 2
            colormap = self.colormaps['polarity']
            field_label = 'Polarity Field'
        elif color_by == 'tension':
            field_data = self._tension_field
            colormap = self.colormaps['tension']
            field_label = 'Tension Field'
        elif color_by == 'curvature':
            field_data = self._curvature_field
            colormap = self.colormaps['curvature']
            field_label = 'Curvature Field'
        elif color_by == 'backflow':
            field_data = self._backflow_field
            colormap = self.colormaps['backflow']
            field_label = 'Backflow Field'
        elif color_by == 'void':
            field_data = self._void_field
            colormap = self.colormaps['void']
            field_label = 'Void Field'
        elif color_by == 'coherence':
            field_data = self._coherence_field
            colormap = self.colormaps['coherence']
            field_label = 'Coherence Field'
        elif color_by == 'ancestry':
            field_data = self._ancestry_field
            colormap = self.colormaps['ancestry']
            field_label = 'Ancestry Field'
        else:
            field_data = self._awareness_field
            colormap = self.colormaps['awareness']
            field_label = 'Awareness Field'
        
        # Plot the field as a heatmap
        mesh = ax.pcolormesh(self._theta_grid, self._phi_grid, field_data,
                           cmap=colormap, shading='auto')
        
        # Add colorbar
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label(field_label)
        
        # Show vector field if requested
        if show_vectors:
            self._draw_vector_field_2d(ax)
        
        # Show grains if requested
        if show_grains:
            self._draw_grains_2d(ax)
        
        # Show relations if requested
        if show_relations:
            self._draw_relations_2d(ax)
        
        # Show ancestry if requested
        if show_ancestry:
            self._draw_ancestry_2d(ax)
        
        # Show vortices if requested
        if show_vortices:
            self._draw_vortices_2d(ax)
        
        # Show lightlike pathways if requested
        if show_pathways:
            self._draw_pathways_2d(ax)
        
        # Configure axes
        ax.set_xlabel('θ (Angular coordinate around tube center)')
        ax.set_ylabel('φ (Angular coordinate through tube)')
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(0, 2*np.pi)
        
        # Set tick positions and labels
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        
        # Add title
        plt.title(f'Unwrapped Torus - {field_label} (t={self.time:.2f})')
        
        # Make layout tight
        plt.tight_layout()
        
        return fig, ax
    
    def _draw_vector_field_2d(self, ax):
        """
        Draw the vector field on the 2D unwrapped torus.
        
        Args:
            ax: Axes to draw into
        """
        # Sample points for vectors (not all points for clarity)
        sample_rate = 8  # Show every 8th point for clarity
        
        # Sample theta and phi coordinates
        thetas = self._theta_grid[::sample_rate, ::sample_rate]
        phis = self._phi_grid[::sample_rate, ::sample_rate]
        
        # Sample vector components
        v_thetas = self._vector_field_theta[::sample_rate, ::sample_rate]
        v_phis = self._vector_field_phi[::sample_rate, ::sample_rate]
        
        # Calculate magnitude for color
        magnitude = np.sqrt(v_thetas**2 + v_phis**2)
        
        # Draw vector field
        quiver = ax.quiver(thetas, phis, v_thetas, v_phis, magnitude,
                         cmap='viridis', scale=20, width=0.003,
                         alpha=0.7)
        
        # Add a small colorbar for vector magnitude
        cbar = plt.colorbar(quiver, ax=ax, orientation='vertical', shrink=0.6)
        cbar.set_label('Vector Magnitude')
    
    def _draw_grains_2d(self, ax):
        """
        Draw grains on the 2D unwrapped torus.
        
        Args:
            ax: Axes to draw into
        """
        # Categorize grains by type
        structure_grains = []
        decay_grains = []
        neutral_grains = []
        superposition_grains = []
        
        for grain_id, grain_data in self.grains.items():
            # Get position
            theta, phi = grain_data['position']
            
            # Size based on awareness and activation
            size = 10 + 50 * grain_data['awareness'] * grain_data['activation']
            
            # Get polarity
            polarity = grain_data['polarity']
            
            # Adjust opacity based on saturation
            alpha = 0.3 + 0.7 * grain_data['saturation']
            
            # Categorize grain
            if grain_data['is_superposition']:
                superposition_grains.append((theta, phi, size, alpha))
            elif polarity > 0.2:
                # Structure-forming
                color_intensity = min(1.0, 0.5 + polarity/2)
                structure_grains.append((theta, phi, size, alpha, color_intensity))
            elif polarity < -0.2:
                # Structure-decaying
                color_intensity = min(1.0, 0.5 - polarity/2)
                decay_grains.append((theta, phi, size, alpha, color_intensity))
            else:
                # Neutral
                color_intensity = 0.5 + 0.5 * grain_data['awareness']
                neutral_grains.append((theta, phi, size, alpha, color_intensity))
        
        # Plot grains by type
        if structure_grains:
            thetas, phis, sizes, alphas, intensities = zip(*structure_grains)
            colors = [(0, 0, intensity) for intensity in intensities]  # Blues
            ax.scatter(thetas, phis, s=sizes, c=colors, alpha=alphas, edgecolors='black', linewidths=0.5)
        
        if decay_grains:
            thetas, phis, sizes, alphas, intensities = zip(*decay_grains)
            colors = [(intensity, 0, 0) for intensity in intensities]  # Reds
            ax.scatter(thetas, phis, s=sizes, c=colors, alpha=alphas, edgecolors='black', linewidths=0.5)
        
        if neutral_grains:
            thetas, phis, sizes, alphas, intensities = zip(*neutral_grains)
            colors = [(0, intensity, 0) for intensity in intensities]  # Greens
            ax.scatter(thetas, phis, s=sizes, c=colors, alpha=alphas, edgecolors='black', linewidths=0.5)
        
        if superposition_grains:
            thetas, phis, sizes, alphas = zip(*superposition_grains)
            ax.scatter(thetas, phis, marker='*', s=sizes, color='cyan', alpha=alphas, edgecolors='white')
    
    def _draw_relations_2d(self, ax):
        """
        Draw relations on the 2D unwrapped torus.
        
        Args:
            ax: Axes to draw into
        """
        for grain_id, relations in self.relation_strengths.items():
            if grain_id not in self.grain_positions:
                continue
                
            # Get source position
            theta1, phi1 = self.grain_positions[grain_id]
            
            # Get grain polarity
            source_polarity = self.grains[grain_id]['polarity'] if grain_id in self.grains else 0.0
            
            for related_id, strength in relations.items():
                if related_id not in self.grain_positions:
                    continue
                
                # Get target position
                theta2, phi2 = self.grain_positions[related_id]
                
                # Handle wrapping - connect through the shortest path
                # For theta wrapping
                if abs(theta2 - theta1) > np.pi:
                    # Connect through the boundary
                    if theta1 < theta2:
                        theta1 += 2 * np.pi
                    else:
                        theta2 += 2 * np.pi
                
                # For phi wrapping
                if abs(phi2 - phi1) > np.pi:
                    # Connect through the boundary
                    if phi1 < phi2:
                        phi1 += 2 * np.pi
                    else:
                        phi2 += 2 * np.pi
                
                # Line width based on strength
                lw = 0.5 + 2.0 * abs(strength)
                
                # Get target polarity for combined effect
                target_polarity = self.grains[related_id]['polarity'] if related_id in self.grains else 0.0
                combined_polarity = source_polarity + target_polarity
                
                # Color based on polarity
                if combined_polarity > 0.3:
                    color = 'blue'
                    alpha = min(0.6, abs(strength))
                elif combined_polarity < -0.3:
                    color = 'red'
                    alpha = min(0.6, abs(strength))
                else:
                    color = 'green'
                    alpha = min(0.4, abs(strength))
                
                # Draw relation
                ax.plot([theta1, theta2], [phi1, phi2], 
                       color=color, linewidth=lw, alpha=alpha)
    
    def _draw_ancestry_2d(self, ax):
        """
        Draw ancestry relationships on the 2D unwrapped torus.
        
        Args:
            ax: Axes to draw into
        """
        for grain_id, ancestors in self.ancestry_network.items():
            if grain_id not in self.grain_positions:
                continue
                
            # Get target position
            theta1, phi1 = self.grain_positions[grain_id]
            
            for ancestor_id in ancestors:
                if ancestor_id not in self.grain_positions:
                    continue
                
                # Get source position
                theta2, phi2 = self.grain_positions[ancestor_id]
                
                # Handle wrapping - connect through the shortest path
                # For theta wrapping
                if abs(theta2 - theta1) > np.pi:
                    # Connect through the boundary
                    if theta1 < theta2:
                        theta1 += 2 * np.pi
                    else:
                        theta2 += 2 * np.pi
                
                # For phi wrapping
                if abs(phi2 - phi1) > np.pi:
                    # Connect through the boundary
                    if phi1 < phi2:
                        phi1 += 2 * np.pi
                    else:
                        phi2 += 2 * np.pi
                
                # Determine if direct parent or more distant ancestor
                if grain_id == ancestor_id:
                    # Self-reference, drawn separately
                    continue
                
                # Get strength from relation or grain data
                strength = 0.5
                if grain_id in self.relation_strengths and ancestor_id in self.relation_strengths[grain_id]:
                    strength = self.relation_strengths[grain_id][ancestor_id]
                
                # Line width based on strength
                lw = 0.5 + 1.5 * strength
                
                # Distinguish direct parents vs. distant ancestors
                if ancestor_id in self.grains and self.grains[ancestor_id]['polarity'] > 0:
                    # Structure-forming ancestors
                    ax.plot([theta1, theta2], [phi1, phi2], 
                           color='darkblue', linewidth=lw, alpha=0.7, 
                           linestyle='-')
                else:
                    # Other ancestors
                    ax.plot([theta1, theta2], [phi1, phi2], 
                           color='purple', linewidth=lw, alpha=0.5, 
                           linestyle='--')
    
    def _draw_vortices_2d(self, ax):
        """
        Draw vortices on the 2D unwrapped torus.
        
        Args:
            ax: Axes to draw into
        """
        # Skip if no vortices
        if not self.vortices:
            return
        
        for vortex in self.vortices:
            # Get position from vortex data
            if 'theta' in vortex and 'phi' in vortex:
                theta, phi = vortex['theta'], vortex['phi']
            elif 'center_id' in vortex and vortex['center_id'] in self.grain_positions:
                theta, phi = self.grain_positions[vortex['center_id']]
            else:
                continue
            
            # Get vortex properties
            strength = vortex.get('strength', 0.5)
            is_lightlike = vortex.get('is_lightlike', False)
            direction = vortex.get('direction', 'clockwise')
            polarity = vortex.get('polarity', 0.0)
            
            # Draw vortex marker
            marker_size = 100 + 200 * strength
            
            # Create vortex color based on polarity and lightlike nature
            if is_lightlike:
                # Lightlike vortices (ethereal)
                if direction == 'clockwise':
                    color = 'cyan'
                else:
                    color = 'magenta'
                edge_color = 'white'
            else:
                # Regular vortices
                if polarity > 0.2:
                    # Structure vortex
                    color = 'blue'
                    edge_color = 'darkblue'
                elif polarity < -0.2:
                    # Decay vortex
                    color = 'red'
                    edge_color = 'darkred'
                else:
                    # Neutral vortex
                    color = 'green'
                    edge_color = 'darkgreen'
            
            # Draw vortex marker
            ax.scatter([theta], [phi], 
                      s=marker_size, color=color, alpha=0.6,
                      edgecolors=edge_color, linewidths=2)
            
            # Draw spiral to indicate rotation
            self._draw_vortex_spiral_2d(ax, theta, phi, direction, strength, color)
    
    def _draw_vortex_spiral_2d(self, ax, theta, phi, direction, strength, color):
        """
        Draw a spiral to indicate vortex rotation in 2D.
        
        Args:
            ax: Axes to draw into
            theta: Vortex theta coordinate
            phi: Vortex phi coordinate
            direction: 'clockwise' or 'counterclockwise'
            strength: Vortex strength
            color: Spiral color
        """
        # Create a spiral path around the vortex center
        spiral_points = 100
        radius_scale = 0.15 * (0.5 + strength)
        max_radius = radius_scale * 2 * np.pi / 6
        
        # Determine spiral direction
        direction_multiplier = 1 if direction == 'clockwise' else -1
        
        # Create spiral points
        spiral_theta = []
        spiral_phi = []
        
        for i in range(spiral_points):
            # Increasing angle
            angle = i / spiral_points * 4 * np.pi
            
            # Increasing radius
            radius = max_radius * (i / spiral_points)
            
            # Calculate position with direction
            theta_offset = radius * np.cos(angle * direction_multiplier)
            phi_offset = radius * np.sin(angle * direction_multiplier)
            
            spiral_theta.append(theta + theta_offset)
            spiral_phi.append(phi + phi_offset)
        
        # Draw spiral
        ax.plot(spiral_theta, spiral_phi, color=color, linewidth=1.5, alpha=0.8)
        
        # Add arrow to indicate direction
        mid_idx = len(spiral_theta) // 2
        if mid_idx > 1:
            # Calculate direction vector
            dx = spiral_theta[mid_idx] - spiral_theta[mid_idx-1]
            dy = spiral_phi[mid_idx] - spiral_phi[mid_idx-1]
            
            # Draw arrow
            ax.quiver(spiral_theta[mid_idx-1], spiral_phi[mid_idx-1], 
                     dx, dy, color=color, scale=2, 
                     width=0.005, headwidth=5, alpha=0.9)
    
    def _draw_pathways_2d(self, ax):
        """
        Draw lightlike pathways on the 2D unwrapped torus.
        
        Args:
            ax: Axes to draw into
        """
        # Draw structure-forming pathways
        for pathway in self.lightlike_pathways['structure']:
            self._draw_pathway_2d(ax, pathway, 'structure')
        
        # Draw structure-decaying pathways
        for pathway in self.lightlike_pathways['decay']:
            self._draw_pathway_2d(ax, pathway, 'decay')
    
    def _draw_pathway_2d(self, ax, pathway, pathway_type):
        """
        Draw a single pathway on the 2D unwrapped torus.
        
        Args:
            ax: Axes to draw into
            pathway: Pathway data dictionary
            pathway_type: 'structure' or 'decay'
        """
        # Extract nodes from pathway
        nodes = pathway.get('nodes', [])
        
        # Skip if not enough nodes
        if len(nodes) < 2:
            return
        
        # Get node positions
        positions = []
        for node_id in nodes:
            if node_id in self.grain_positions:
                positions.append(self.grain_positions[node_id])
        
        # Skip if not enough positions
        if len(positions) < 2:
            return
        
        # Process positions to handle wrapping
        processed_thetas = []
        processed_phis = []
        
        # Start with first position
        theta_prev, phi_prev = positions[0]
        processed_thetas.append(theta_prev)
        processed_phis.append(phi_prev)
        
        # Process remaining positions
        for theta, phi in positions[1:]:
            # Handle theta wrapping
            if abs(theta - theta_prev) > np.pi:
                if theta_prev < theta:
                    theta_prev += 2 * np.pi
                else:
                    theta += 2 * np.pi
            
            # Handle phi wrapping
            if abs(phi - phi_prev) > np.pi:
                if phi_prev < phi:
                    phi_prev += 2 * np.pi
                else:
                    phi += 2 * np.pi
            
            # Update for next iteration
            processed_thetas.append(theta)
            processed_phis.append(phi)
            theta_prev, phi_prev = theta, phi
        
        # Get pathway properties
        avg_polarity = pathway.get('avg_polarity', 0.0)
        pathway_strength = pathway.get('pathway_strength', 0.5)
        
        # Line width based on strength
        lw = 1.0 + 2.0 * pathway_strength
        
        # Color based on type and polarity
        if pathway_type == 'structure':
            color = 'blue'
            zorder = 2
        else:  # decay
            color = 'red'
            zorder = 1
        
        # Alpha based on strength
        alpha = min(0.9, 0.3 + pathway_strength)
        
        # Draw pathway
        ax.plot(processed_thetas, processed_phis, 
               color=color, linewidth=lw, alpha=alpha, 
               zorder=zorder)
        
        # Add arrow to show direction
        if len(processed_thetas) > 2:
            mid_idx = len(processed_thetas) // 2
            ax.quiver(processed_thetas[mid_idx-1], processed_phis[mid_idx-1],
                     processed_thetas[mid_idx] - processed_thetas[mid_idx-1],
                     processed_phis[mid_idx] - processed_phis[mid_idx-1],
                     color=color, scale=10,
                     width=0.003, headwidth=5, alpha=alpha)
    
    def create_combined_visualization(self, **kwargs):
        """
        Create a combined 3D and 2D visualization of the torus.
        
        Args:
            **kwargs: Additional parameters including:
                - color_by: Field to color by ('awareness', 'phase', 'polarity', etc.)
                - show_grains: Whether to show individual grains
                - show_ancestry: Whether to show ancestry relationships
                - show_vortices: Whether to show vortices
                - show_pathways: Whether to show lightlike pathways
        
        Returns:
            Figure with both 3D and 2D visualizations
        """
        # Create figure with two subplots
        fig = plt.figure(figsize=(18, 8))
        
        # 3D subplot
        ax1 = fig.add_subplot(121, projection='3d')
        
        # 2D subplot
        ax2 = fig.add_subplot(122)
        
        # Render 3D torus
        kwargs['show_relations'] = False  # Simplify 3D view
        self._render_torus_on_axis(ax1, **kwargs)
        
        # Render 2D unwrapped torus
        self._render_unwrapped_on_axis(ax2, **kwargs)
        
        # Add shared title
        color_by = kwargs.get('color_by', 'awareness')
        fig.suptitle(f'Collapse Geometry Torus Visualization - {color_by.capitalize()} Field (t={self.time:.2f})',
                   fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def _render_torus_on_axis(self, ax, **kwargs):
        """
        Render a 3D torus on the given axis.
        
        Args:
            ax: 3D axis to render onto
            **kwargs: Visualization parameters
        """
        # Get parameters
        color_by = kwargs.get('color_by', 'awareness')
        show_grains = kwargs.get('show_grains', True)
        show_relations = kwargs.get('show_relations', False)
        show_ancestry = kwargs.get('show_ancestry', True)
        show_vortices = kwargs.get('show_vortices', True)
        show_pathways = kwargs.get('show_pathways', True)
        show_recursion = kwargs.get('show_recursion', True)
        major_radius = kwargs.get('major_radius', 3.0)
        minor_radius = kwargs.get('minor_radius', 1.0)
        view_angle = kwargs.get('view_angle', (30, 45))
        surface_alpha = kwargs.get('alpha', 0.7)
        
        # Create 3D torus mesh
        torus_x, torus_y, torus_z = self._create_torus_mesh(major_radius, minor_radius)
        
        # Get field for coloring
        if color_by == 'awareness':
            field_data = self._awareness_field
            colormap = self.colormaps['awareness']
        elif color_by == 'phase':
            field_data = self._phase_field / (2 * np.pi)  # Normalize to [0,1]
            colormap = self.colormaps['phase']
        elif color_by == 'polarity':
            # Transform polarity from [-1,1] to [0,1]
            field_data = (self._polarity_field + 1) / 2
            colormap = self.colormaps['polarity']
        elif color_by == 'tension':
            field_data = self._tension_field
            colormap = self.colormaps['tension']
        elif color_by == 'curvature':
            field_data = self._curvature_field
            colormap = self.colormaps['curvature']
        elif color_by == 'backflow':
            field_data = self._backflow_field
            colormap = self.colormaps['backflow']
        elif color_by == 'void':
            field_data = self._void_field
            colormap = self.colormaps['void']
        elif color_by == 'coherence':
            field_data = self._coherence_field
            colormap = self.colormaps['coherence']
        elif color_by == 'ancestry':
            field_data = self._ancestry_field
            colormap = self.colormaps['ancestry']
        else:
            field_data = self._awareness_field
            colormap = self.colormaps['awareness']
        
        # Plot torus surface colored by field
        torus_surface = ax.plot_surface(
            torus_x, torus_y, torus_z,
            facecolors=colormap(field_data),
            alpha=surface_alpha,
            antialiased=True,
            shade=True
        )
        
        # Show grains if requested
        if show_grains:
            self._draw_grains(ax, major_radius, minor_radius)
        
        # Show relations if requested
        if show_relations:
            self._draw_relations(ax, major_radius, minor_radius)
        
        # Show ancestry if requested
        if show_ancestry:
            self._draw_ancestry(ax, major_radius, minor_radius)
        
        # Show vortices if requested
        if show_vortices:
            self._draw_vortices(ax, major_radius, minor_radius)
        
        # Show pathways if requested
        if show_pathways:
            self._draw_pathways(ax, major_radius, minor_radius)
        
        # Show recursion patterns if requested
        if show_recursion:
            self._draw_recursion_patterns(ax, major_radius, minor_radius)
        
        # Configure view
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Set axis labels and scale
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        
        # Hide axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    
    def _render_unwrapped_on_axis(self, ax, **kwargs):
        """
        Render a 2D unwrapped torus on the given axis.
        
        Args:
            ax: 2D axis to render onto
            **kwargs: Visualization parameters
        """
        # Get parameters
        color_by = kwargs.get('color_by', 'awareness')
        show_grains = kwargs.get('show_grains', True)
        show_relations = kwargs.get('show_relations', False)
        show_ancestry = kwargs.get('show_ancestry', False)  # Simplified for combined view
        show_vortices = kwargs.get('show_vortices', True)
        show_pathways = kwargs.get('show_pathways', True)
        show_vectors = kwargs.get('show_vectors', True)
        
        # Get field for coloring
        if color_by == 'awareness':
            field_data = self._awareness_field
            colormap = self.colormaps['awareness']
        elif color_by == 'phase':
            field_data = self._phase_field / (2 * np.pi)  # Normalize to [0,1]
            colormap = self.colormaps['phase']
        elif color_by == 'polarity':
            # Transform polarity from [-1,1] to [0,1]
            field_data = (self._polarity_field + 1) / 2
            colormap = self.colormaps['polarity']
        elif color_by == 'tension':
            field_data = self._tension_field
            colormap = self.colormaps['tension']
        elif color_by == 'curvature':
            field_data = self._curvature_field
            colormap = self.colormaps['curvature']
        elif color_by == 'backflow':
            field_data = self._backflow_field
            colormap = self.colormaps['backflow']
        elif color_by == 'void':
            field_data = self._void_field
            colormap = self.colormaps['void']
        elif color_by == 'coherence':
            field_data = self._coherence_field
            colormap = self.colormaps['coherence']
        elif color_by == 'ancestry':
            field_data = self._ancestry_field
            colormap = self.colormaps['ancestry']
        else:
            field_data = self._awareness_field
            colormap = self.colormaps['awareness']
        
        # Plot the field as a heatmap
        mesh = ax.pcolormesh(self._theta_grid, self._phi_grid, field_data,
                           cmap=colormap, shading='auto')
        
        # Show vector field if requested (for 2D only)
        if show_vectors:
            self._draw_vector_field_2d(ax)
        
        # Show grains if requested
        if show_grains:
            self._draw_grains_2d(ax)
        
        # Show relations if requested
        if show_relations:
            self._draw_relations_2d(ax)
        
        # Show ancestry if requested
        if show_ancestry:
            self._draw_ancestry_2d(ax)
        
        # Show vortices if requested
        if show_vortices:
            self._draw_vortices_2d(ax)
        
        # Show lightlike pathways if requested
        if show_pathways:
            self._draw_pathways_2d(ax)
        
        # Configure axes
        ax.set_xlabel('θ (Angular coordinate around tube center)')
        ax.set_ylabel('φ (Angular coordinate through tube)')
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(0, 2*np.pi)
        
        # Set tick positions and labels
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    def create_field_composition_visualization(self):
        """
        Create a visualization showing multiple fields for comparison.
        
        Returns:
            Figure with multiple field visualizations
        """
        # Create a 2x3 grid for key fields
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        # Field configurations
        fields = [
            ('awareness', self._awareness_field, self.colormaps['awareness'], 'Awareness Field'),
            ('phase', self._phase_field / (2 * np.pi), self.colormaps['phase'], 'Phase Field'),
            ('polarity', (self._polarity_field + 1) / 2, self.colormaps['polarity'], 'Polarity Field'),
            ('curvature', self._curvature_field, self.colormaps['curvature'], 'Curvature Field'),
            ('backflow', self._backflow_field, self.colormaps['backflow'], 'Backflow Field'),
            ('ancestry', self._ancestry_field, self.colormaps['ancestry'], 'Ancestry Field')
        ]
        
        # Render each field
        for i, (field_name, field_data, colormap, title) in enumerate(fields):
            row, col = i // 3, i % 3
            ax = axs[row, col]
            
            # Plot field
            mesh = ax.pcolormesh(self._theta_grid, self._phi_grid, field_data,
                               cmap=colormap, shading='auto')
            
            # Add colorbar
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label(field_name.capitalize())
            
            # Add title
            ax.set_title(title)
            
            # Configure axes
            ax.set_xlabel('θ')
            ax.set_ylabel('φ')
            ax.set_xlim(0, 2*np.pi)
            ax.set_ylim(0, 2*np.pi)
            
            # Set tick positions and labels
            ax.set_xticks([0, np.pi, 2*np.pi])
            ax.set_xticklabels(['0', 'π', '2π'])
            ax.set_yticks([0, np.pi, 2*np.pi])
            ax.set_yticklabels(['0', 'π', '2π'])
        
        # Add main title
        fig.suptitle(f'Collapse Geometry Field Composition (t={self.time:.2f})', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def create_ancestry_visualization(self):
        """
        Create a specialized visualization focusing on ancestry networks.
    
        Returns:
            Figure with ancestry network visualization
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
    
        # Plot ancestry field as background
        mesh = ax.pcolormesh(self._theta_grid, self._phi_grid, self._ancestry_field,
                           cmap=self.colormaps['ancestry'], shading='auto', alpha=0.7)
    
        # Add colorbar for ancestry field
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label('Ancestry Density')
    
        # Draw grains with different markers based on their ancestry characteristics
        self_referential_grains = []
        deep_ancestry_grains = []
        regular_grains = []
        superposition_grains = []
    
        for grain_id, grain_data in self.grains.items():
            # Get position
            theta, phi = grain_data['position']
        
            # Size based on awareness and saturation
            size = 20 + 100 * grain_data['awareness'] * grain_data['saturation']
        
            # Alpha based on activation
            alpha = 0.4 + 0.6 * grain_data['activation']
        
            # Categorize based on ancestry
            ancestry = grain_data['ancestry']
            ancestry_size = len(ancestry)
        
            if grain_data['is_superposition']:
                superposition_grains.append((theta, phi, size, alpha))
            elif grain_id in ancestry:  # Self-referential
                self_referential_grains.append((theta, phi, size, alpha, ancestry_size))
            elif ancestry_size > 5:  # Deep ancestry
                deep_ancestry_grains.append((theta, phi, size, alpha, ancestry_size))
            else:  # Regular grains
                regular_grains.append((theta, phi, size, alpha, ancestry_size))
    
        # Draw ancestry connections
        for grain_id, ancestors in self.ancestry_network.items():
            if grain_id not in self.grain_positions:
                continue
            
            # Get target position
            theta1, phi1 = self.grain_positions[grain_id]
        
            for ancestor_id in ancestors:
                if ancestor_id not in self.grain_positions or ancestor_id == grain_id:
                    continue
            
                # Get source position
                theta2, phi2 = self.grain_positions[ancestor_id]
            
                # Handle wrapping - connect through the shortest path
                # For theta wrapping
                if abs(theta2 - theta1) > np.pi:
                    # Connect through the boundary
                    if theta1 < theta2:
                        theta1 += 2 * np.pi
                    else:
                        theta2 += 2 * np.pi
            
                # For phi wrapping
                if abs(phi2 - phi1) > np.pi:
                    # Connect through the boundary
                    if phi1 < phi2:
                        phi1 += 2 * np.pi
                    else:
                        phi2 += 2 * np.pi
            
                # Determine line style based on relationship
                if ancestor_id in self.grains[grain_id]['ancestry'] and grain_id in self.grains.get(ancestor_id, {}).get('ancestry', set()):
                    # Bidirectional ancestry (recursive loop)
                    ax.plot([theta1, theta2], [phi1, phi2], 
                           color='purple', linewidth=1.5, alpha=0.7, 
                           linestyle='-', zorder=3)
                else:
                    # One-way ancestry
                    ax.plot([theta1, theta2], [phi1, phi2], 
                           color='navy', linewidth=1.0, alpha=0.5, 
                           linestyle='--', zorder=2)
    
        # Draw regular grains
        if regular_grains:
            thetas, phis, sizes, alphas, ancestry_sizes = zip(*regular_grains)
            colors = plt.cm.plasma(np.array(ancestry_sizes) / 10)
        
            # Handle sizes separately to fix multiplication issue
            sizes_array = np.array(sizes)  # Convert to numpy array for element-wise multiplication
            ax.scatter(thetas, phis, s=sizes_array, c=colors, alpha=alphas, edgecolors='black', linewidths=0.5, zorder=4)
    
        # Draw deep ancestry grains
        if deep_ancestry_grains:
            thetas, phis, sizes, alphas, ancestry_sizes = zip(*deep_ancestry_grains)
            colors = plt.cm.plasma(np.array(ancestry_sizes) / 15)
        
            # Handle sizes separately to fix multiplication issue
            sizes_array = np.array(sizes)  # Convert to numpy array for element-wise multiplication
            ax.scatter(thetas, phis, s=sizes_array, c=colors, alpha=alphas, edgecolors='black', linewidths=1.0, zorder=5,
                      marker='s')  # Square marker
    
        # Draw self-referential grains - THIS IS WHERE THE BUG WAS
        if self_referential_grains:
            thetas, phis, sizes, alphas, ancestry_sizes = zip(*self_referential_grains)
            colors = plt.cm.plasma(np.array(ancestry_sizes) / 15)
        
            # Handle sizes separately to fix multiplication issue
            sizes_array = np.array(sizes) * 1.5  # Convert to numpy array for multiplication
            ax.scatter(thetas, phis, s=sizes_array, c=colors, alpha=alphas, edgecolors='purple', linewidths=1.5, zorder=6,
                      marker='*')  # Star marker
    
        # Draw superposition grains
        if superposition_grains:
            thetas, phis, sizes, alphas = zip(*superposition_grains)
        
            # Handle sizes separately to fix multiplication issue
            sizes_array = np.array(sizes)  # Convert to numpy array
            ax.scatter(thetas, phis, s=sizes_array, color='cyan', alpha=alphas, edgecolors='white', linewidths=1.0, zorder=7,
                      marker='d')  # Diamond marker
    
        # Draw recursive patterns
        for pattern in self.recursive_patterns:
            # Get position
            theta, phi = pattern['position']
            strength = pattern['strength']
        
            # Create a self-referential loop
            radius = 0.1 * (0.5 + strength)
            loop_points = 50
            loop_theta = []
            loop_phi = []
        
            for i in range(loop_points + 1):
                angle = i / loop_points * 2 * np.pi
                loop_theta.append(theta + radius * np.cos(angle))
                loop_phi.append(phi + radius * np.sin(angle))
        
            # Draw loop
            ax.plot(loop_theta, loop_phi, color='purple', linewidth=2.0, alpha=0.8, zorder=8)
        
            # Add arrow to indicate recursion direction
            mid_idx = 3 * loop_points // 4
            ax.quiver(loop_theta[mid_idx-1], loop_phi[mid_idx-1],
                     loop_theta[mid_idx] - loop_theta[mid_idx-1],
                     loop_phi[mid_idx] - loop_phi[mid_idx-1],
                     color='purple', scale=5, width=0.005,
                     headwidth=8, alpha=0.8, zorder=9)
    
        # Configure axes
        ax.set_xlabel('θ (Angular coordinate around tube center)')
        ax.set_ylabel('φ (Angular coordinate through tube)')
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(0, 2*np.pi)
    
        # Set tick positions and labels
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
        # Add title
        plt.title(f'Collapse Geometry Ancestry Network (t={self.time:.2f})')
    
        # Add legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    
        legend_elements = [
            Line2D([0], [0], color='navy', linestyle='--', label='One-way Ancestry'),
            Line2D([0], [0], color='purple', label='Recursive Ancestry'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Regular Grain'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=10, label='Deep Ancestry Grain'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='orange', markersize=12, label='Self-referential Grain'),
            Line2D([0], [0], marker='d', color='w', markerfacecolor='cyan', markersize=10, label='Superposition Grain')
        ]
    
        ax.legend(handles=legend_elements, loc='upper right')
    
        # Adjust layout
        plt.tight_layout()
    
        return fig
    
    def create_vortex_dynamics_visualization(self):
        """
        Create a specialized visualization focusing on vortex dynamics.
        
        Returns:
            Figure with vortex dynamics visualization
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Left plot: Phase field with vortices
        mesh1 = ax1.pcolormesh(self._theta_grid, self._phi_grid, self._phase_field / (2 * np.pi),
                            cmap=self.colormaps['phase'], shading='auto')
        
        # Add colorbar
        cbar1 = fig.colorbar(mesh1, ax=ax1)
        cbar1.set_label('Phase Field')
        
        # Draw vector field on phase plot
        self._draw_vector_field_2d(ax1)
        
        # Draw vortices on phase plot
        self._draw_vortices_2d(ax1)
        
        # Right plot: Curvature field with pathways
        mesh2 = ax2.pcolormesh(self._theta_grid, self._phi_grid, self._curvature_field,
                            cmap=self.colormaps['curvature'], shading='auto')
        
        # Add colorbar
        cbar2 = fig.colorbar(mesh2, ax=ax2)
        cbar2.set_label('Curvature Field')
        
        # Draw lightlike pathways on curvature plot
        self._draw_pathways_2d(ax2)
        
        # Draw vortices on curvature plot as well for comparison
        self._draw_vortices_2d(ax2)
        
        # Configure axes for both plots
        for ax in [ax1, ax2]:
            ax.set_xlabel('θ (Angular coordinate around tube center)')
            ax.set_ylabel('φ (Angular coordinate through tube)')
            ax.set_xlim(0, 2*np.pi)
            ax.set_ylim(0, 2*np.pi)
            
            # Set tick positions and labels
            ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
            ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
            ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
            ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        
        # Add titles
        ax1.set_title('Phase Field with Vortices and Vector Flow')
        ax2.set_title('Curvature Field with Lightlike Pathways')
        
        # Add main title
        fig.suptitle(f'Collapse Geometry Vortex Dynamics (t={self.time:.2f})', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig


# Factory function for easier creation
def create_enhanced_visualizer(manifold=None, resolution=100, field_smoothing=2.0):
    """
    Create an enhanced torus visualizer and optionally update with manifold.
    
    Args:
        manifold: Optional manifold to visualize
        resolution: Resolution of field visualization
        field_smoothing: Amount of field smoothing to apply
        
    Returns:
        EnhancedTorusVisualizer instance
    """
    visualizer = EnhancedTorusVisualizer(resolution=resolution, field_smoothing=field_smoothing)
    
    if manifold:
        visualizer.update_state(manifold)
        
    return visualizer