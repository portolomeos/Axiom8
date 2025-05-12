"""
Enhanced Collapse Geometry Visualizer Core - OPTIMIZED VERSION

The core module for the Collapse Geometry visualization system with improved
support for signed values, reduced redundancy, and better field handling.

Key improvements:
1. Removed redundant field storage and normalization
2. Enhanced handling of signed fields (polarity, curvature, activation, pressure)
3. Optimized field calculation with proper ancestry and recursive pattern display
4. Improved vector field calculation for better flow visualization
5. Better handling of toroidal topology with proper wraparound visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import math
from collections import defaultdict, deque
import time

def create_safe_norm(vmin, vcenter, vmax):
    """
    Create a colormap normalization that safely handles edge cases.
    
    Args:
        vmin: Minimum value
        vcenter: Center value
        vmax: Maximum value
        
    Returns:
        Matplotlib normalization object
    """
    # Ensure values are in ascending order to prevent ValueError
    if not (vmin < vcenter < vmax):
        # Adjust the values to create a valid norm
        if vmin >= 0 and vmax > 0:
            # All positive - use standard Normalize
            return colors.Normalize(vmin=vmin, vmax=vmax)
        elif vmin < 0 and vmax <= 0:
            # All negative - use standard Normalize
            return colors.Normalize(vmin=vmin, vmax=vmax)
        elif vmin >= vcenter:
            # Adjust center to be slightly above vmin
            vcenter = vmin + (vmax - vmin) * 0.05
            return colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        elif vmax <= vcenter:
            # Adjust center to be slightly below vmax
            vcenter = vmax - (vmax - vmin) * 0.05
            return colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        else:
            # Fallback to standard Normalize
            return colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        # Normal case: values are in the correct order
        return colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

class CollapseVisualizerBase:
    """
    Enhanced base class for all Collapse Geometry visualizers.
    Provides common functionality with improved handling of field data.
    """
    
    def __init__(self, resolution=100, field_smoothing=2.0):
        """
        Initialize the base visualizer.
        
        Args:
            resolution: Resolution of field visualization grid
            field_smoothing: Smoothing factor for field visualization
        """
        # Core parameters
        self.resolution = resolution
        self.field_smoothing = field_smoothing
        self.manifold = None
        
        # Initialize field data
        self._initialize_fields()
        
        # Grain and structural data
        self.grains = {}
        self.grain_positions = {}
        self.relation_strengths = defaultdict(dict)
        self.vortices = []
        self.lightlike_pathways = {'structure': [], 'decay': []}
        self.ancestry_network = {}
        self.recursive_patterns = []
        self.phase_domains = []
        self.void_regions = []
        
        # Timing info
        self.time = 0.0
        self.last_update_time = time.time()
        self.animation_time = 0.0
        
        # Visualization settings
        self.signed_field_visualization = True
        self.display_field_divergence = True
        self.display_recursive_patterns = True
        self.continuous_update_mode = True  # Enables field history tracking
        
        # Field history tracking for fluctuation visualization
        self.field_history = {}
        self.field_history_length = 20
        
        # Initialize visualization options
        self._initialize_visualization_options()
    
    def _initialize_fields(self):
        """Initialize visualization field grids and data structures"""
        # Create coordinate grids
        self._theta_grid, self._phi_grid = np.meshgrid(
            np.linspace(0, 2*np.pi, self.resolution),
            np.linspace(0, 2*np.pi, self.resolution)
        )
        
        # Create Cartesian projection grids
        self._x_grid, self._y_grid = np.meshgrid(
            np.linspace(-1, 1, self.resolution),
            np.linspace(-1, 1, self.resolution)
        )
        
        # Core field arrays - primary representation of manifold
        self._field_data = {
            # Unsigned fields (always positive)
            'awareness': np.zeros((self.resolution, self.resolution)),
            'phase': np.zeros((self.resolution, self.resolution)),
            'tension': np.zeros((self.resolution, self.resolution)),
            'backflow': np.zeros((self.resolution, self.resolution)),
            'void': np.zeros((self.resolution, self.resolution)),
            'coherence': np.zeros((self.resolution, self.resolution)),
            'ancestry': np.zeros((self.resolution, self.resolution)),
            
            # Signed fields (can be positive or negative)
            'polarity': np.zeros((self.resolution, self.resolution)),
            'curvature': np.zeros((self.resolution, self.resolution)),
            'activation': np.zeros((self.resolution, self.resolution)),
            'pressure': np.zeros((self.resolution, self.resolution))
        }
        
        # Track field statistics for better rendering
        self._field_stats = {field_name: {
            'min': -1.0 if field_name in ['polarity', 'curvature', 'activation', 'pressure'] else 0.0,
            'max': 1.0 if field_name in ['polarity', 'curvature', 'activation', 'pressure'] else 1.0,
            'mean': 0.0,
            'std': 0.0,
            'abs_max': 1.0
        } for field_name in self._field_data}
        
        # Vector field for flow visualization
        self._vector_field = {
            'theta': np.zeros((self.resolution, self.resolution)),
            'phi': np.zeros((self.resolution, self.resolution)),
            'magnitude': np.zeros((self.resolution, self.resolution)),
            'divergence': np.zeros((self.resolution, self.resolution))
        }
        
        # Initialize Cartesian projections
        self._cartesian_projections = {}
        
        # Track zero-crossings and features
        self._field_features = {
            'zero_crossings': defaultdict(list),
            'vortices': [],
            'circular_patterns': [],
            'recursive_regions': []
        }
    
    def _initialize_visualization_options(self):
        """Initialize visualization options, colormaps and rendering settings"""
        # Create field-specific colormaps
        self._colormaps = self._create_field_colormaps()
        
        # Create norms for each field type
        self._norms = self._create_field_norms()
        
        # Field titles and descriptions
        self.field_titles = {
            'awareness': 'Awareness Field',
            'phase': 'Phase Field',
            'polarity': 'Polarity Field (Structure vs Decay)',
            'tension': 'Tension Field',
            'curvature': 'Curvature Field',
            'backflow': 'Circular Backflow Field',
            'void': 'Void Formation Field',
            'coherence': 'Coherence Field',
            'ancestry': 'Ancestry Field',
            'activation': 'Activation Field',
            'pressure': 'Pressure Field (Awareness × Activation)'
        }
        
        # Field rendering options
        self.field_render_options = {
            'phase': {'colorbar': True, 'contour': False},
            'polarity': {'colorbar': True, 'contour': True, 'zero_contour': True},
            'pressure': {'colorbar': True, 'contour': True, 'zero_contour': True},
            'curvature': {'colorbar': True, 'contour': True, 'zero_contour': True},
            'activation': {'colorbar': True, 'contour': True, 'zero_contour': True}
        }
        
        # Default options
        self._default_render_options = {'colorbar': True, 'contour': True, 'zero_contour': False}
    
    def _create_field_colormaps(self):
        """Create specialized colormaps for different field visualizations"""
        # Standard colormaps
        default_maps = {
            'awareness': self._create_awareness_colormap(),
            'phase': cm.hsv,
            'tension': cm.magma,
            'backflow': cm.inferno,
            'void': cm.Greys,
            'coherence': self._create_coherence_colormap(),
            'ancestry': self._create_ancestry_colormap(),
        }
        
        # Signed field colormaps (diverging)
        signed_maps = {
            'polarity': self._create_signed_polarity_colormap(),
            'curvature': self._create_signed_curvature_colormap(),
            'activation': self._create_signed_activation_colormap(),
            'pressure': self._create_signed_pressure_colormap()
        }
        
        # Merge the dictionaries
        combined_maps = {**default_maps, **signed_maps}
        
        # Add specialized visualization maps
        combined_maps.update({
            'structure': self._create_structure_colormap(),
            'divergence': self._create_divergence_colormap(),
            'distortion': self._create_distortion_colormap(),
            'circulation': self._create_circulation_colormap(),
            'recursive': self._create_recursive_colormap()
        })
        
        return combined_maps
    
    def _create_field_norms(self):
        """Create normalization objects for field visualization"""
        norms = {}
        
        # Create norms for signed fields
        for field_name in ['polarity', 'curvature', 'activation', 'pressure']:
            # Default norm centered at zero
            norms[field_name] = create_safe_norm(
                vmin=-1.0, 
                vcenter=0.0, 
                vmax=1.0
            )
        
        return norms
    
    def _create_awareness_colormap(self):
        """Create custom colormap for awareness field"""
        return colors.LinearSegmentedColormap.from_list(
            'awareness_cmap',
            [(0.0, '#081B41'),  # Deep blue for low awareness
             (0.4, '#085CA9'),  # Medium blue
             (0.7, '#8A64D6'),  # Purple for medium awareness
             (0.9, '#FFB05C'),  # Orange
             (1.0, '#FFE74C')], # Yellow for high awareness
            N=256
        )
    
    def _create_signed_polarity_colormap(self):
        """Create enhanced colormap for signed polarity values with clear zero-point"""
        return colors.LinearSegmentedColormap.from_list(
            'signed_polarity_cmap',
            [(0.0, '#7F1D1D'),  # Deep red for strong negative (decay)
             (0.2, '#DC2626'),  # Medium red
             (0.4, '#FECACA'),  # Light red/pink
             (0.5, '#F9FAFB'),  # White/light gray for zero/neutral
             (0.6, '#BFDBFE'),  # Light blue
             (0.8, '#2563EB'),  # Medium blue
             (1.0, '#1E3A8A')], # Deep blue for strong positive (structure)
            N=256
        )
    
    def _create_signed_curvature_colormap(self):
        """Create enhanced colormap for signed curvature values"""
        return colors.LinearSegmentedColormap.from_list(
            'signed_curvature_cmap',
            [(0.0, '#064E3B'),  # Dark teal for negative curvature
             (0.25, '#10B981'), # Medium teal
             (0.45, '#D1FAE5'), # Light teal
             (0.5, '#F9FAFB'),  # White/light gray for zero
             (0.55, '#F5D0FE'), # Light purple
             (0.75, '#A855F7'), # Medium purple
             (1.0, '#581C87')], # Dark purple for positive curvature
            N=256
        )
    
    def _create_signed_activation_colormap(self):
        """Create enhanced colormap for signed activation values"""
        return colors.LinearSegmentedColormap.from_list(
            'signed_activation_cmap',
            [(0.0, '#991B1B'),  # Deep red for strong negative activation
             (0.3, '#F87171'),  # Medium red
             (0.45, '#FEE2E2'), # Light red
             (0.5, '#F9FAFB'),  # White/light gray for zero
             (0.55, '#D1FAE5'), # Light green
             (0.7, '#34D399'),  # Medium green
             (1.0, '#065F46')], # Deep green for strong positive activation
            N=256
        )
    
    def _create_signed_pressure_colormap(self):
        """Create enhanced colormap for signed pressure values"""
        return colors.LinearSegmentedColormap.from_list(
            'signed_pressure_cmap',
            [(0.0, '#92400E'),  # Deep orange for strong contraction
             (0.3, '#F59E0B'),  # Orange
             (0.45, '#FEF3C7'), # Light yellow
             (0.5, '#F9FAFB'),  # White/light gray for zero
             (0.55, '#DBEAFE'), # Light blue
             (0.7, '#3B82F6'),  # Medium blue
             (1.0, '#1E40AF')], # Deep blue for strong expansion
            N=256
        )
    
    def _create_coherence_colormap(self):
        """Create custom colormap for coherence field"""
        return colors.LinearSegmentedColormap.from_list(
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
        return colors.LinearSegmentedColormap.from_list(
            'ancestry_cmap',
            [(0.0, '#F8F9FA'),  # White for no ancestry
             (0.3, '#CDB4DB'),  # Light purple
             (0.6, '#FF99C8'),  # Pink
             (0.8, '#FCB162'),  # Orange
             (1.0, '#FDCB58')], # Yellow for strong ancestry
            N=256
        )
    
    def _create_structure_colormap(self):
        """Create custom colormap for structure enhancement"""
        return colors.LinearSegmentedColormap.from_list(
            'structure_cmap',
            [(0.0, '#1A1A2E'),  # Dark navy for low structure
             (0.3, '#16213E'),  # Navy blue
             (0.5, '#4a4daa'),  # Periwinkle blue
             (0.7, '#7A85CC'),  # Lavender
             (0.9, '#9FB4FF'),  # Light lavender
             (1.0, '#E0EAFF')], # Pale blue for high structure
            N=256
        )
    
    def _create_divergence_colormap(self):
        """Create colormap for divergence visualization"""
        return colors.LinearSegmentedColormap.from_list(
            'divergence_cmap',
            [(0.0, '#F87171'),  # Red for negative divergence (sink)
             (0.5, '#FFFFFF'),  # White for zero divergence
             (1.0, '#60A5FA')], # Blue for positive divergence (source)
            N=256
        )
    
    def _create_distortion_colormap(self):
        """Create custom colormap for distortion field"""
        return colors.LinearSegmentedColormap.from_list(
            'distortion_cmap',
            [(0.0, '#7F1D1D'),  # Deep red for negative distortion
             (0.3, '#DC2626'),  # Medium red
             (0.45, '#FECACA'), # Light red
             (0.5, '#F9FAFB'),  # White/light gray for zero
             (0.55, '#C7D2FE'), # Light purple
             (0.7, '#6366F1'),  # Medium purple
             (1.0, '#3730A3')], # Deep purple for positive distortion
            N=256
        )
    
    def _create_circulation_colormap(self):
        """Create custom colormap for circulation visualization"""
        return colors.LinearSegmentedColormap.from_list(
            'circulation_cmap',
            [(0.0, '#FFD700'),  # Gold for counterclockwise
             (0.5, '#F9FAFB'),  # White/light gray for zero
             (1.0, '#9333EA')], # Purple for clockwise
            N=256
        )
    
    def _create_recursive_colormap(self):
        """Create custom colormap for recursive pattern visualization"""
        return colors.LinearSegmentedColormap.from_list(
            'recursive_cmap',
            [(0.0, '#F9FAFB'),  # White for no recursion
             (0.3, '#FDE68A'),  # Light yellow
             (0.6, '#F59E0B'),  # Orange
             (0.8, '#B45309'),  # Dark orange
             (1.0, '#7F1D1D')], # Deep red for strong recursion
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
        
        # Clear previous field data
        self._clear_field_data()
        
        # Extract grain-level data
        self._extract_grain_data(manifold)
        
        # Generate field data from grains
        self._generate_field_data()
        
        # Extract structural patterns
        self._extract_structural_patterns(manifold)
        
        # Extract ancestry network
        self._extract_ancestry_network(manifold)
        
        # Process field data (smoothing, normalization)
        self._process_field_data()
        
        # Calculate derived fields
        self._calculate_derived_fields()
        
        # Detect field features
        self._detect_field_features()
        
        # Calculate vector fields
        self._calculate_vector_fields()
        
        # Generate Cartesian projections
        self._generate_cartesian_projections()
        
        # Incorporate simulation state data if provided
        if state:
            self._incorporate_state_data(state)
        
        # Track field history if enabled
        if self.continuous_update_mode:
            self._update_field_history()
    
    def _clear_field_data(self):
        """Clear all field data to prepare for updates"""
        # Reset all field arrays
        for field_name in self._field_data:
            self._field_data[field_name].fill(0.0)
        
        # Reset vector field
        for component in self._vector_field:
            self._vector_field[component].fill(0.0)
        
        # Reset field features
        for feature_type in self._field_features:
            if isinstance(self._field_features[feature_type], dict):
                self._field_features[feature_type].clear()
            elif isinstance(self._field_features[feature_type], list):
                self._field_features[feature_type] = []
            elif isinstance(self._field_features[feature_type], defaultdict):
                self._field_features[feature_type] = defaultdict(list)
    
    def _extract_grain_data(self, manifold):
        """Extract grain data and calculate positions for visualization"""
        # Reset collections
        self.grains = {}
        self.grain_positions = {}
        self.relation_strengths = defaultdict(dict)
        
        # Process each grain
        for grain_id, grain in manifold.grains.items():
            # Extract grain properties, including handling negative values properly
            grain_data = {
                'awareness': grain.awareness,
                'saturation': grain.grain_saturation,
                'activation': grain.grain_activation,
                'collapse_metric': grain.collapse_metric,
                'polarity': getattr(grain, 'polarity', 0.0),
                'ancestry': getattr(grain, 'ancestry', set()).copy(),
                'is_superposition': grain.is_in_superposition(),
                'is_self_referential': grain_id in getattr(grain, 'ancestry', set()),
                'recursive_tension': getattr(grain, 'recursive_tension', 0.0),
                'degrees_of_freedom': getattr(grain, 'degrees_of_freedom', 1.0 - grain.grain_saturation),
                'constraint_tension': getattr(grain, 'constraint_tension', 0.0),
                'circular_recursion_factor': getattr(grain, 'circular_recursion_factor', 0.0),
                'activity_state': getattr(grain, 'activity_state', "active")
            }
            
            # Determine position on torus - try different methods
            position = self._get_grain_position(grain, manifold, grain_id)
            grain_data['position'] = position
            
            # Store grain data
            self.grains[grain_id] = grain_data
            self.grain_positions[grain_id] = position
            
            # Extract relations
            if hasattr(grain, 'relations'):
                for related_id, relation_strength in grain.relations.items():
                    # Store numeric strength value
                    if isinstance(relation_strength, (int, float)):
                        strength = relation_strength
                    elif hasattr(relation_strength, 'relation_strength'):
                        strength = relation_strength.relation_strength
                    else:
                        strength = 0.5  # Default fallback
                    
                    self.relation_strengths[grain_id][related_id] = strength
    
    def _get_grain_position(self, grain, manifold, grain_id):
        """Get grain position on the torus (theta, phi) by trying different methods"""
        # Try direct grain properties first
        if hasattr(grain, 'theta') and hasattr(grain, 'phi'):
            return (grain.theta, grain.phi)
        
        # Try toroidal coordinator
        if hasattr(manifold, 'toroidal_coordinator'):
            coordinator = manifold.toroidal_coordinator
            
            # Try explicit grain positions
            if hasattr(coordinator, 'grain_positions') and grain_id in coordinator.grain_positions:
                return coordinator.grain_positions[grain_id]
            
            # Try to synchronize and get coordinates
            try:
                coordinator.synchronize_coordinates(grain_id)
                if hasattr(grain, 'theta') and hasattr(grain, 'phi'):
                    return (grain.theta, grain.phi)
            except:
                pass
        
        # Try config space
        if hasattr(manifold, 'config_space'):
            try:
                point = manifold.config_space.get_point(grain_id)
                if point and hasattr(point, 'get_toroidal_coordinates'):
                    return point.get_toroidal_coordinates()
            except:
                pass
        
        # Fallback: generate deterministic position from grain properties
        return self._calculate_position_from_properties(grain_id, grain)
    
    def _calculate_position_from_properties(self, grain_id, grain):
        """Calculate deterministic grain position from its intrinsic properties"""
        # Use properties to calculate a stable position
        # This ensures consistency in visualization
        
        # Create a deterministic seed from grain ID
        id_hash = sum(ord(c) * (i+1) for i, c in enumerate(grain_id)) % 997
        id_phase = (id_hash / 997) * 2 * np.pi
        
        # Use polarity to affect theta (structure = low theta, decay = high theta)
        polarity = getattr(grain, 'polarity', 0.0)
        theta_polarity = np.pi * (1.0 - (polarity + 1.0) / 2.0)  # Map [-1,1] to [π,0]
        
        # Mix with randomized component for diversity
        theta = (theta_polarity * 0.7 + id_phase * 0.3) % (2 * np.pi)
        
        # Use awareness and activation to affect phi
        awareness = grain.awareness
        activation = grain.grain_activation
        phi_component = np.pi * (awareness * 0.5 + activation * 0.5)
        
        # Mix with random component
        phi = (phi_component * 0.8 + id_phase * 0.2) % (2 * np.pi)
        
        return (theta, phi)
    
    def _generate_field_data(self):
        """Generate field data from grains"""
        # Process each grain to contribute to fields
        for grain_id, grain_data in self.grains.items():
            # Get position
            theta, phi = grain_data['position']
            
            # Calculate influence radius
            awareness = grain_data['awareness']
            activation = grain_data['activation']
            saturation = grain_data['saturation']
            
            # Base radius on grain properties
            radius_factor = 0.1 + 0.2 * abs(awareness) + 0.1 * abs(activation) + 0.1 * saturation
            grid_radius = max(2, int(self.resolution * radius_factor / (2 * np.pi)))
            
            # Calculate grid indices for grain position
            center_i = int((phi / (2 * np.pi)) * self.resolution) % self.resolution
            center_j = int((theta / (2 * np.pi)) * self.resolution) % self.resolution
            
            # Calculate influence range with wrapping
            i_range = range(center_i - grid_radius, center_i + grid_radius + 1)
            j_range = range(center_j - grid_radius, center_j + grid_radius + 1)
            
            # Apply grain influence to fields
            for i in i_range:
                i_wrapped = i % self.resolution  # Handle wrapping
                for j in j_range:
                    j_wrapped = j % self.resolution  # Handle wrapping
                    
                    # Calculate grid coordinates
                    grid_theta = self._theta_grid[i_wrapped, j_wrapped]
                    grid_phi = self._phi_grid[i_wrapped, j_wrapped]
                    
                    # Calculate toroidal distance (accounting for circular wraparound)
                    d_theta = min(abs(grid_theta - theta), 2*np.pi - abs(grid_theta - theta))
                    d_phi = min(abs(grid_phi - phi), 2*np.pi - abs(grid_phi - phi))
                    distance = np.sqrt(d_theta**2 + d_phi**2)
                    
                    # Apply gaussian falloff with distance
                    if distance <= grid_radius * 2 * np.pi / self.resolution:
                        # Calculate weight based on distance
                        sigma = grid_radius * 2 * np.pi / self.resolution / 2
                        weight = np.exp(-(distance**2) / (2 * sigma**2))
                        
                        # Apply contribution to each field
                        self._field_data['awareness'][i_wrapped, j_wrapped] += awareness * weight
                        self._field_data['polarity'][i_wrapped, j_wrapped] += grain_data['polarity'] * weight
                        self._field_data['activation'][i_wrapped, j_wrapped] += activation * weight
                        
                        # Apply to curvature field
                        self._field_data['curvature'][i_wrapped, j_wrapped] += grain_data['collapse_metric'] * weight
                        
                        # Apply to tension field - combine constraint and recursive tension
                        tension = grain_data['constraint_tension'] + 0.5 * grain_data['recursive_tension']
                        self._field_data['tension'][i_wrapped, j_wrapped] += tension * weight
                        
                        # Apply to ancestry field - based on ancestry size
                        ancestry_size = len(grain_data['ancestry'])
                        ancestry_factor = min(1.0, ancestry_size / 10.0)  # Cap at 1.0
                        self._field_data['ancestry'][i_wrapped, j_wrapped] += ancestry_factor * weight
                        
                        # Apply to backflow field - stronger for recursive patterns
                        if grain_data['is_self_referential']:
                            self._field_data['backflow'][i_wrapped, j_wrapped] += (
                                0.5 + 0.5 * grain_data['recursive_tension']
                            ) * weight
                        
                        # Apply to void field - stronger for superposition
                        if grain_data['is_superposition']:
                            self._field_data['void'][i_wrapped, j_wrapped] += 0.8 * weight
                        
                        # Apply to coherence field - based on degrees of freedom
                        coherence = 1.0 - grain_data['degrees_of_freedom']
                        self._field_data['coherence'][i_wrapped, j_wrapped] += coherence * weight
                        
                        # Apply to phase field
                        phase_contribution = (grain_data['polarity'] + 1) * np.pi * weight * 0.1
                        self._field_data['phase'][i_wrapped, j_wrapped] = (
                            self._field_data['phase'][i_wrapped, j_wrapped] + phase_contribution
                        ) % (2 * np.pi)
    
    def _extract_structural_patterns(self, manifold):
        """Extract emergent structural patterns from manifold"""
        # Reset structure collections
        self.vortices = []
        self.lightlike_pathways = {'structure': [], 'decay': []}
        self.phase_domains = []
        self.void_regions = []
        
        # Extract vortices from coordinator
        if hasattr(manifold, 'toroidal_coordinator'):
            coordinator = manifold.toroidal_coordinator
            
            # Get vortices
            if hasattr(coordinator, 'vortices'):
                self.vortices = coordinator.vortices.copy() if coordinator.vortices else []
            
            # Get lightlike pathways
            if hasattr(coordinator, 'lightlike_pathways'):
                for pathway_type in ['structure', 'decay']:
                    if pathway_type in coordinator.lightlike_pathways:
                        pathways = coordinator.lightlike_pathways[pathway_type]
                        self.lightlike_pathways[pathway_type] = pathways.copy() if pathways else []
        
        # Extract void regions
        if hasattr(manifold, 'void_formation_events'):
            self.void_regions = manifold.void_formation_events[-10:] if manifold.void_formation_events else []
        
        # Extract phase domains if available
        if hasattr(manifold, 'phase_domains'):
            self.phase_domains = manifold.phase_domains.copy() if manifold.phase_domains else []
        
        # Apply vortex data to field features
        self._field_features['vortices'] = self.vortices
    
    def _extract_ancestry_network(self, manifold):
        """Extract ancestry network and recursive patterns"""
        # Reset ancestry data
        self.ancestry_network = {}
        self.recursive_patterns = []
        
        # Build ancestry network
        for grain_id, grain_data in self.grains.items():
            ancestry = grain_data['ancestry']
            
            # Skip empty ancestry
            if not ancestry:
                continue
            
            # Track ancestry connections
            self.ancestry_network[grain_id] = [
                ancestor_id for ancestor_id in ancestry 
                if ancestor_id in self.grains  # Only include existing grains
            ]
            
            # Detect recursive patterns (self-reference)
            if grain_id in ancestry:
                # This grain references itself
                self.recursive_patterns.append({
                    'grain_id': grain_id,
                    'position': grain_data['position'],
                    'strength': grain_data['circular_recursion_factor'] if grain_data['circular_recursion_factor'] > 0 
                              else grain_data['recursive_tension'],
                    'is_circular': grain_data['circular_recursion_factor'] > 0
                })
        
        # Apply recursive patterns to field features
        self._field_features['circular_patterns'] = self.recursive_patterns
    
    def _process_field_data(self):
        """Process field data - apply smoothing and calculate statistics"""
        # Apply smoothing to fields
        self._apply_field_smoothing()
        
        # Calculate field statistics
        for field_name, field_data in self._field_data.items():
            # Calculate basic statistics
            self._field_stats[field_name]['min'] = np.min(field_data)
            self._field_stats[field_name]['max'] = np.max(field_data)
            self._field_stats[field_name]['mean'] = np.mean(field_data)
            self._field_stats[field_name]['std'] = np.std(field_data)
            
            # Calculate absolute max for signed fields
            if field_name in ['polarity', 'curvature', 'activation', 'pressure']:
                self._field_stats[field_name]['abs_max'] = max(
                    abs(self._field_stats[field_name]['min']),
                    abs(self._field_stats[field_name]['max'])
                )
    
    def _apply_field_smoothing(self):
        """Apply smoothing to field data"""
        try:
            # Use scipy for better smoothing if available
            from scipy.ndimage import gaussian_filter
            
            # Apply smoothing to unsigned fields
            for field_name in ['awareness', 'tension', 'backflow', 'void', 'coherence', 'ancestry']:
                self._field_data[field_name] = gaussian_filter(
                    self._field_data[field_name], sigma=self.field_smoothing
                )
            
            # Apply smoothing to signed fields
            for field_name in ['polarity', 'curvature', 'activation']:
                self._field_data[field_name] = gaussian_filter(
                    self._field_data[field_name], sigma=self.field_smoothing
                )
            
            # Special handling for phase field (circular data)
            phase_x = np.cos(self._field_data['phase'])
            phase_y = np.sin(self._field_data['phase'])
            
            phase_x_smooth = gaussian_filter(phase_x, sigma=self.field_smoothing)
            phase_y_smooth = gaussian_filter(phase_y, sigma=self.field_smoothing)
            
            self._field_data['phase'] = np.arctan2(phase_y_smooth, phase_x_smooth) % (2 * np.pi)
            
        except ImportError:
            # Simple rolling average if scipy not available
            for _ in range(int(self.field_smoothing * 2)):
                # Apply simple smoothing to all fields
                for field_name, field_data in self._field_data.items():
                    if field_name != 'phase':
                        self._simple_smooth(field_data)
                
                # Special handling for phase field
                phase_x = np.cos(self._field_data['phase'])
                phase_y = np.sin(self._field_data['phase'])
                
                self._simple_smooth(phase_x)
                self._simple_smooth(phase_y)
                
                self._field_data['phase'] = np.arctan2(phase_y, phase_x) % (2 * np.pi)
    
    def _simple_smooth(self, field):
        """Apply a simple smoothing operation to a field"""
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
    
    def _calculate_derived_fields(self):
        """Calculate derived fields from primary fields"""
        # Calculate pressure field (awareness * activation)
        awareness = self._field_data['awareness']
        activation = self._field_data['activation']
        self._field_data['pressure'] = awareness * activation
        
        # Calculate field statistics for pressure
        self._field_stats['pressure']['min'] = np.min(self._field_data['pressure'])
        self._field_stats['pressure']['max'] = np.max(self._field_data['pressure'])
        self._field_stats['pressure']['mean'] = np.mean(self._field_data['pressure'])
        self._field_stats['pressure']['std'] = np.std(self._field_data['pressure'])
        self._field_stats['pressure']['abs_max'] = max(
            abs(self._field_stats['pressure']['min']),
            abs(self._field_stats['pressure']['max'])
        )
    
    def _detect_field_features(self):
        """Detect features in fields like zero-crossings"""
        # Reset feature lists
        self._field_features['zero_crossings'] = defaultdict(list)
        
        # Detect zero-crossings in signed fields
        for field_name in ['polarity', 'curvature', 'activation', 'pressure']:
            self._detect_zero_crossings(field_name)
        
        # Detect recursive regions
        self._detect_recursive_regions()
    
    def _detect_zero_crossings(self, field_name):
        """Detect zero-crossings in a signed field"""
        field = self._field_data[field_name]
        zero_crossings = []
        
        rows, cols = field.shape
        
        for i in range(rows):
            for j in range(cols):
                # Check neighbors with toroidal wrapping
                neighbors = [
                    (i, (j+1) % cols),  # Right
                    ((i+1) % rows, j),  # Down
                    (i, (j-1) % cols),  # Left
                    ((i-1) % rows, j)   # Up
                ]
                
                current_val = field[i, j]
                
                # Skip if near zero (avoid noise)
                if abs(current_val) < 0.05:
                    continue
                
                # Check for sign change with neighbors
                for ni, nj in neighbors:
                    neighbor_val = field[ni, nj]
                    
                    # Check for sign change
                    if current_val * neighbor_val < 0:
                        # Store crossing point
                        zero_crossings.append({
                            'position': (i, j),
                            'neighbor': (ni, nj),
                            'value': current_val,
                            'theta': self._theta_grid[i, j],
                            'phi': self._phi_grid[i, j]
                        })
                        break
        
        # Store zero-crossings
        self._field_features['zero_crossings'][field_name] = zero_crossings
    
    def _detect_recursive_regions(self):
        """Detect regions with recursive patterns"""
        recursive_regions = []
        
        # Check for grains with circular recursion or self-reference
        for grain_id, grain_data in self.grains.items():
            if grain_data['is_self_referential'] or grain_data['circular_recursion_factor'] > 0:
                # This is a recursive grain
                theta, phi = grain_data['position']
                
                # Calculate strength
                strength = max(
                    grain_data['circular_recursion_factor'],
                    grain_data['recursive_tension']
                )
                
                # Add to regions
                recursive_regions.append({
                    'grain_id': grain_id,
                    'position': (theta, phi),
                    'strength': strength,
                    'is_circular': grain_data['circular_recursion_factor'] > 0
                })
        
        # Store recursive regions
        self._field_features['recursive_regions'] = recursive_regions
    
    def _calculate_vector_fields(self):
        """Calculate vector fields for visualization"""
        # Calculate gradients of field data
        
        # Gradients from polarity field
        polarity_field = self._field_data['polarity']
        dx_polarity = np.zeros_like(polarity_field)
        dy_polarity = np.zeros_like(polarity_field)
        
        # Calculate gradients manually with toroidal wrapping
        rows, cols = polarity_field.shape
        for i in range(rows):
            for j in range(cols):
                # Neighboring coordinates with wrapping
                i_prev = (i - 1) % rows
                i_next = (i + 1) % rows
                j_prev = (j - 1) % cols
                j_next = (j + 1) % cols
                
                # Calculate gradients using central difference
                dx_polarity[i, j] = (polarity_field[i, j_next] - polarity_field[i, j_prev]) / 2
                dy_polarity[i, j] = (polarity_field[i_next, j] - polarity_field[i_prev, j]) / 2
        
        # Phase field gradients - need special handling for circular values
        phase_field = self._field_data['phase']
        dx_phase = np.zeros_like(phase_field)
        dy_phase = np.zeros_like(phase_field)
        
        for i in range(rows):
            for j in range(cols):
                # Neighboring coordinates with wrapping
                i_prev = (i - 1) % rows
                i_next = (i + 1) % rows
                j_prev = (j - 1) % cols
                j_next = (j + 1) % cols
                
                # Calculate phase differences with proper circular handling
                dx_diff = phase_field[i, j_next] - phase_field[i, j_prev]
                if dx_diff > np.pi: dx_diff -= 2 * np.pi
                if dx_diff < -np.pi: dx_diff += 2 * np.pi
                
                dy_diff = phase_field[i_next, j] - phase_field[i_prev, j]
                if dy_diff > np.pi: dy_diff -= 2 * np.pi
                if dy_diff < -np.pi: dy_diff += 2 * np.pi
                
                dx_phase[i, j] = dx_diff / 2
                dy_phase[i, j] = dy_diff / 2
        
        # Pressure field gradients
        pressure_field = self._field_data['pressure']
        dx_pressure = np.zeros_like(pressure_field)
        dy_pressure = np.zeros_like(pressure_field)
        
        for i in range(rows):
            for j in range(cols):
                # Neighboring coordinates with wrapping
                i_prev = (i - 1) % rows
                i_next = (i + 1) % rows
                j_prev = (j - 1) % cols
                j_next = (j + 1) % cols
                
                # Calculate gradients
                dx_pressure[i, j] = (pressure_field[i, j_next] - pressure_field[i, j_prev]) / 2
                dy_pressure[i, j] = (pressure_field[i_next, j] - pressure_field[i_prev, j]) / 2
        
        # Combine field gradients to create vector field
        # Weighted combination of polarity, phase, and pressure gradients
        self._vector_field['theta'] = (
            dx_polarity * 0.5 +  # Polarity contribution
            dx_phase * 0.3 +     # Phase contribution
            dx_pressure * 0.2    # Pressure contribution
        )
        
        self._vector_field['phi'] = (
            dy_polarity * 0.5 +  # Polarity contribution
            dy_phase * 0.3 +     # Phase contribution
            dy_pressure * 0.2    # Pressure contribution
        )
        
        # Calculate vector magnitude
        self._vector_field['magnitude'] = np.sqrt(
            self._vector_field['theta']**2 + 
            self._vector_field['phi']**2
        )
        
        # Normalize vectors by magnitude
        mask = self._vector_field['magnitude'] > 0.01
        if np.any(mask):
            self._vector_field['theta'][mask] /= self._vector_field['magnitude'][mask]
            self._vector_field['phi'][mask] /= self._vector_field['magnitude'][mask]
        
        # Calculate divergence
        self._calculate_field_divergence()
    
    def _calculate_field_divergence(self):
        """Calculate divergence of the vector field"""
        # Calculate divergence: ∇·F = ∂u/∂x + ∂v/∂y
        rows, cols = self._vector_field['theta'].shape
        
        for i in range(rows):
            for j in range(cols):
                # Neighboring coordinates with wrapping
                i_prev = (i - 1) % rows
                i_next = (i + 1) % rows
                j_prev = (j - 1) % cols
                j_next = (j + 1) % cols
                
                # Calculate divergence using central difference
                du_dx = (self._vector_field['theta'][i, j_next] - 
                         self._vector_field['theta'][i, j_prev]) / 2
                
                dv_dy = (self._vector_field['phi'][i_next, j] - 
                         self._vector_field['phi'][i_prev, j]) / 2
                
                # Store divergence
                self._vector_field['divergence'][i, j] = du_dx + dv_dy
    
    def _generate_cartesian_projections(self):
        """Generate Cartesian projections of toroidal fields"""
        # Create a mask for the projection (circle)
        cartesian_mask = np.ones((self.resolution, self.resolution), dtype=bool)
        radius = 0.98  # Slightly smaller than 1 to avoid edge artifacts
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                x = self._x_grid[i, j]
                y = self._y_grid[i, j]
                distance = np.sqrt(x**2 + y**2)
                cartesian_mask[i, j] = distance <= radius
        
        # Clear existing projections
        self._cartesian_projections = {}
        
        # Project each field
        for field_name, field_data in self._field_data.items():
            # Create empty projection
            projection = np.full((self.resolution, self.resolution), np.nan)
            
            # Map toroidal coordinates to Cartesian
            for i in range(self.resolution):
                for j in range(self.resolution):
                    # Skip points outside mask
                    if not cartesian_mask[i, j]:
                        continue
                    
                    # Calculate polar coordinates
                    x = self._x_grid[i, j]
                    y = self._y_grid[i, j]
                    
                    r = np.sqrt(x**2 + y**2) / radius  # Normalize to [0, 1]
                    theta = np.arctan2(y, x) % (2 * np.pi)
                    
                    # Map to toroidal coordinates
                    toroidal_theta = theta
                    toroidal_phi = r * 2 * np.pi
                    
                    # Convert to toroidal indices
                    i_toroidal = int((toroidal_phi / (2 * np.pi)) * self.resolution) % self.resolution
                    j_toroidal = int((toroidal_theta / (2 * np.pi)) * self.resolution) % self.resolution
                    
                    # Get field value
                    projection[i, j] = field_data[i_toroidal, j_toroidal]
            
            # Store projection
            self._cartesian_projections[field_name] = projection
    
    def _update_field_history(self):
        """Update field history for tracking changes over time"""
        # Initialize history if needed
        if not self.field_history:
            for field_name in self._field_data:
                self.field_history[field_name] = deque(maxlen=self.field_history_length)
        
        # Add current fields to history
        for field_name, field_data in self._field_data.items():
            self.field_history[field_name].append({
                'time': self.time,
                'min': self._field_stats[field_name]['min'],
                'max': self._field_stats[field_name]['max'],
                'mean': self._field_stats[field_name]['mean']
            })
    
    def _incorporate_state_data(self, state):
        """Incorporate data from simulation state if provided"""
        # Handle the case where state has visualization fields
        if hasattr(state, '_field_data') and state._field_data is not None:
            for field_name in self._field_data:
                if field_name in state._field_data and state._field_data[field_name].shape == self._field_data[field_name].shape:
                    # Combine fields intelligently
                    if field_name in ['polarity', 'curvature', 'activation', 'pressure']:
                        # For signed fields, direct replacement preserves sign
                        self._field_data[field_name] = state._field_data[field_name].copy()
                    else:
                        # For unsigned fields, take maximum
                        self._field_data[field_name] = np.maximum(
                            self._field_data[field_name], 
                            state._field_data[field_name]
                        )
        # Handle older style state objects
        elif hasattr(state, '_awareness_field') and state._awareness_field is not None:
            if state._awareness_field.shape == self._field_data['awareness'].shape:
                # Update awareness field
                self._field_data['awareness'] = np.maximum(
                    self._field_data['awareness'], 
                    state._awareness_field
                )
                
                # Update phase field
                self._field_data['phase'] = state._phase_field
                
                # Update other fields if available
                if hasattr(state, '_polarity_field'):
                    self._field_data['polarity'] = state._polarity_field.copy()
                
                if hasattr(state, '_tension_field'):
                    self._field_data['tension'] = np.maximum(
                        self._field_data['tension'], 
                        state._tension_field
                    )
                
                if hasattr(state, '_curvature_field'):
                    self._field_data['curvature'] = state._curvature_field.copy()
                
                if hasattr(state, '_activation_field'):
                    self._field_data['activation'] = state._activation_field.copy()
                
                # Update derived fields
                self._calculate_derived_fields()
        
        # Incorporate field features if available
        if hasattr(state, '_field_features') and state._field_features is not None:
            for feature_type, features in state._field_features.items():
                if feature_type in self._field_features:
                    self._field_features[feature_type] = features
    
    def render_field(self, ax, field_name, title=None, **kwargs):
        """
        Render a field on a matplotlib axis.
        
        Args:
            ax: Matplotlib axis to plot on
            field_name: Name of the field to render
            title: Optional title override
            **kwargs: Additional rendering options including:
                - colorbar: Whether to show colorbar (default: True)
                - contour: Whether to show contour lines (default: True)
                - zero_contour: Whether to highlight zero contour (default: False)
                - show_grains: Whether to show grains (default: True)
                - show_vectors: Whether to show vector field (default: False)
        """
        # Get field data
        if field_name not in self._field_data:
            ax.text(0.5, 0.5, f"Field '{field_name}' not available", 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        field_data = self._field_data[field_name]
        
        # Get render options
        options = self._default_render_options.copy()
        options.update(self.field_render_options.get(field_name, {}))
        options.update(kwargs)
        
        # Get colormap
        cmap = self._colormaps.get(field_name, cm.viridis)
        
        # Get norm for signed fields
        norm = None
        if field_name in self._norms and options.get('use_signed_norm', True):
            # For signed fields, use TwoSlopeNorm with 0 as center
            field_min = self._field_stats[field_name]['min']
            field_max = self._field_stats[field_name]['max']
            
            # Only use signed norm if field actually crosses zero
            if field_min < 0 and field_max > 0:
                norm = create_safe_norm(
                    vmin=field_min,
                    vcenter=0.0,
                    vmax=field_max
                )
        
        # Handle phase field specially
        if field_name == 'phase':
            # Use direct imshow with hsv colormap for phase
            im = ax.imshow(
                field_data, 
                cmap=cm.hsv, 
                vmin=0, 
                vmax=2*np.pi,
                origin='lower', 
                extent=[0, 2*np.pi, 0, 2*np.pi],
                aspect='equal'
            )
        else:
            # Regular field rendering
            im = ax.imshow(
                field_data, 
                cmap=cmap,
                norm=norm,
                origin='lower', 
                extent=[0, 2*np.pi, 0, 2*np.pi],
                aspect='equal'
            )
        
        # Add colorbar if requested
        if options.get('colorbar', True):
            plt.colorbar(im, ax=ax, label=field_name.capitalize())
        
        # Add contours if requested
        if options.get('contour', True):
            # Create contour levels
            if norm:
                # For signed fields with norm
                levels = np.linspace(self._field_stats[field_name]['min'], 
                                    self._field_stats[field_name]['max'], 
                                    9)
            else:
                # For unsigned fields
                levels = np.linspace(np.min(field_data), np.max(field_data), 9)
            
            # Draw contours
            contours = ax.contour(
                self._theta_grid, 
                self._phi_grid, 
                field_data, 
                levels=levels, 
                colors='k', 
                alpha=0.3, 
                linewidths=0.5
            )
            
            # Add zero contour for signed fields if requested
            if options.get('zero_contour', False) and field_name in ['polarity', 'curvature', 'activation', 'pressure']:
                if self._field_stats[field_name]['min'] < 0 and self._field_stats[field_name]['max'] > 0:
                    zero_contours = ax.contour(
                        self._theta_grid, 
                        self._phi_grid, 
                        field_data, 
                        levels=[0], 
                        colors='r', 
                        linewidths=1.5
                    )
                    plt.clabel(zero_contours, inline=1, fontsize=8, fmt='%1.1f')
        
        # Show grains if requested
        if options.get('show_grains', True):
            self._draw_grains(ax, field_name)
        
        # Show vector field if requested
        if options.get('show_vectors', False):
            self._draw_vector_field(ax)
        
        # Show recursive patterns if requested
        if options.get('show_recursive', self.display_recursive_patterns):
            self._draw_recursive_patterns(ax)
        
        # Add title
        if title:
            ax.set_title(title)
        else:
            ax.set_title(self.field_titles.get(field_name, field_name.capitalize()))
        
        # Configure axes
        ax.set_xlabel('Theta')
        ax.set_ylabel('Phi')
        ax.set_xlim([0, 2*np.pi])
        ax.set_ylim([0, 2*np.pi])
        
        # Add Pi-based ticks
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    def _draw_grains(self, ax, field_name=None):
        """
        Draw grains on the visualization.
        
        Args:
            ax: Matplotlib axis to draw on
            field_name: Optional field name to customize appearance
        """
        # Draw each grain
        for grain_id, grain_data in self.grains.items():
            theta, phi = grain_data['position']
            
            # Calculate grain size based on properties
            awareness = grain_data['awareness']
            activation = grain_data['activation']
            polarity = grain_data['polarity']
            
            # Base size on absolute value of awareness and activation
            size = 20 + 100 * abs(awareness) * abs(activation)
            
            # Choose color based on polarity or field
            if field_name == 'polarity':
                # Color based on polarity
                if polarity > 0.2:
                    color = 'blue'  # Structure
                elif polarity < -0.2:
                    color = 'red'  # Decay
                else:
                    color = 'green'  # Neutral
            elif field_name == 'activation':
                # Color based on activation
                if activation > 0.2:
                    color = 'green'  # Positive activation
                elif activation < -0.2:
                    color = 'red'  # Negative activation
                else:
                    color = 'gray'  # Neutral
            elif field_name == 'ancestry':
                # Color based on ancestry size
                ancestry_size = len(grain_data['ancestry'])
                if ancestry_size > 5:
                    color = 'purple'  # Large ancestry
                elif ancestry_size > 0:
                    color = 'orange'  # Small ancestry
                else:
                    color = 'gray'  # No ancestry
            else:
                # Default coloring based on activity state
                activity_state = grain_data['activity_state']
                if activity_state == 'active':
                    color = 'blue'
                elif activity_state == 'saturated':
                    color = 'purple'
                elif activity_state == 'constrained':
                    color = 'orange'
                elif activity_state == 'inactive':
                    color = 'gray'
                else:
                    color = 'green'
            
            # Set opacity based on activation
            alpha = 0.3 + 0.7 * min(1.0, abs(activation))
            
            # Draw grain
            ax.scatter(
                theta, phi, 
                s=size, 
                color=color, 
                alpha=alpha, 
                edgecolors='black', 
                linewidths=0.5
            )
    
    def _draw_vector_field(self, ax):
        """
        Draw vector field arrows on the visualization.
        
        Args:
            ax: Matplotlib axis to draw on
        """
        # Subsample grid for clarity
        skip = max(1, self.resolution // 20)
        
        # Draw vector field
        ax.quiver(
            self._theta_grid[::skip, ::skip],
            self._phi_grid[::skip, ::skip],
            self._vector_field['theta'][::skip, ::skip],
            self._vector_field['phi'][::skip, ::skip],
            color='k',
            scale=20,
            width=0.003,
            alpha=0.7
        )
    
    def _draw_recursive_patterns(self, ax):
        """
        Draw recursive patterns on the visualization.
        
        Args:
            ax: Matplotlib axis to draw on
        """
        # Draw recursive patterns
        for pattern in self.recursive_patterns:
            theta, phi = pattern['position']
            strength = pattern['strength']
            is_circular = pattern.get('is_circular', False)
            
            # Size based on strength
            size = 50 + 100 * strength
            
            # Different marker for circular vs self-referential
            marker = 'o' if is_circular else '*'
            
            # Draw pattern
            ax.scatter(
                theta, phi,
                s=size,
                color='gold' if is_circular else 'orange',
                marker=marker,
                alpha=0.8,
                edgecolors='black',
                linewidths=1.0,
                zorder=10  # Ensure drawn on top
            )
            
            # Draw circle around recursive pattern
            circle = plt.Circle(
                (theta, phi),
                radius=0.2 + 0.3 * strength,
                fill=False,
                color='gold' if is_circular else 'orange',
                linestyle='--',
                alpha=0.6
            )
            ax.add_patch(circle)
    
    def create_combined_visualization(self, **kwargs):
        """
        Create a combined visualization showing multiple fields.
        
        Args:
            **kwargs: Additional parameters including:
                - fields: List of field names to include (default: ['polarity', 'awareness', 'activation', 'pressure'])
                - rows: Number of rows in the grid (default: 2)
                - show_grains: Whether to show grains (default: True)
                - show_vectors: Whether to show vector field (default: False)
                - figsize: Figure size (default: (18, 12))
                
        Returns:
            Figure and axes objects
        """
        # Get parameters
        fields = kwargs.get('fields', ['polarity', 'awareness', 'activation', 'pressure'])
        rows = kwargs.get('rows', 2)
        show_grains = kwargs.get('show_grains', True)
        show_vectors = kwargs.get('show_vectors', False)
        figsize = kwargs.get('figsize', (18, 12))
        
        # Calculate columns needed
        cols = (len(fields) + rows - 1) // rows
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.array(axes).flatten()  # Ensure axes is always array
        
        # Render each field
        for i, field_name in enumerate(fields):
            if i < len(axes):
                self.render_field(
                    axes[i], 
                    field_name,
                    show_grains=show_grains,
                    show_vectors=show_vectors
                )
        
        # Hide unused axes
        for i in range(len(fields), len(axes)):
            axes[i].axis('off')
        
        # Add overall title
        fig.suptitle(f'Collapse Geometry Visualization (t={self.time:.2f})', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig, axes
    
    def create_circular_recursion_visualization(self, **kwargs):
        """
        Create a specialized visualization focusing on circular recursion patterns.
        
        Args:
            **kwargs: Additional parameters
                
        Returns:
            Figure and axes objects
        """
        # Create figure with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Polarity field with zero contour
        self.render_field(
            axes[0, 0], 
            'polarity', 
            title='Polarity Field with Zero Contour', 
            zero_contour=True,
            show_recursive=True
        )
        
        # Backflow field with recursive patterns
        self.render_field(
            axes[0, 1], 
            'backflow', 
            title='Circular Backflow Field', 
            show_recursive=True,
            show_vectors=True
        )
        
        # Ancestry field
        self.render_field(
            axes[1, 0], 
            'ancestry', 
            title='Ancestry Field', 
            show_recursive=True
        )
        
        # Phase field with circular effects
        self.render_field(
            axes[1, 1], 
            'phase', 
            title='Phase Field with Circular Effects',
            show_recursive=True,
            contour=False
        )
        
        # Draw recursive patterns and connections
        for pattern in self.recursive_patterns:
            grain_id = pattern['grain_id']
            theta, phi = pattern['position']
            
            # Draw ancestry connections if available
            if grain_id in self.ancestry_network:
                for ancestor_id in self.ancestry_network[grain_id]:
                    # Skip if ancestor doesn't exist
                    if ancestor_id not in self.grain_positions:
                        continue
                    
                    # Get ancestor position
                    anc_theta, anc_phi = self.grain_positions[ancestor_id]
                    
                    # Draw connection on all plots
                    for ax in axes.flatten():
                        # Handle toroidal wrapping for connections
                        # If distance would be shorter going around, draw two line segments
                        
                        # Theta wrapping
                        if abs(theta - anc_theta) > np.pi:
                            # Draw two segments for theta wrapping
                            if theta < anc_theta:
                                # First segment
                                ax.plot([theta, 2*np.pi], [phi, phi], 'k-', alpha=0.3, linewidth=0.5)
                                # Second segment
                                ax.plot([0, anc_theta], [anc_phi, anc_phi], 'k-', alpha=0.3, linewidth=0.5)
                            else:
                                # First segment
                                ax.plot([anc_theta, 2*np.pi], [anc_phi, anc_phi], 'k-', alpha=0.3, linewidth=0.5)
                                # Second segment
                                ax.plot([0, theta], [phi, phi], 'k-', alpha=0.3, linewidth=0.5)
                        else:
                            # Direct connection for theta
                            theta_start, theta_end = theta, anc_theta
                            
                        # Phi wrapping
                        if abs(phi - anc_phi) > np.pi:
                            # Draw two segments for phi wrapping
                            if phi < anc_phi:
                                # First segment
                                ax.plot([theta, theta], [phi, 2*np.pi], 'k-', alpha=0.3, linewidth=0.5)
                                # Second segment
                                ax.plot([anc_theta, anc_theta], [0, anc_phi], 'k-', alpha=0.3, linewidth=0.5)
                            else:
                                # First segment
                                ax.plot([anc_theta, anc_theta], [anc_phi, 2*np.pi], 'k-', alpha=0.3, linewidth=0.5)
                                # Second segment
                                ax.plot([theta, theta], [0, phi], 'k-', alpha=0.3, linewidth=0.5)
                        else:
                            # Direct connection
                            ax.plot([theta, anc_theta], [phi, anc_phi], 'k-', alpha=0.3, linewidth=0.5)
        
        # Add overall title
        fig.suptitle(f'Circular Recursion Visualization (t={self.time:.2f})', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig, axes
    
    def create_pressure_visualization(self, **kwargs):
        """
        Create a visualization specifically showing pressure dynamics (awareness * activation).
        
        Args:
            **kwargs: Additional parameters
            
        Returns:
            Figure and axes objects
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get pressure field
        field_data = self._field_data['pressure']
        
        # Get appropriate colormap
        cmap = self._colormaps.get('pressure')
        
        # Get appropriate norm
        vmin = self._field_stats['pressure']['min']
        vmax = self._field_stats['pressure']['max']
        
        # Create safe norm centered at zero
        norm = create_safe_norm(vmin, 0.0, vmax)
        
        # Plot the pressure field
        im = ax.pcolormesh(
            self._theta_grid, 
            self._phi_grid, 
            field_data,
            cmap=cmap, 
            norm=norm, 
            shading='auto'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Pressure (Awareness × Activation)')
        
        # Add zero contour
        if vmin < 0 and vmax > 0:
            contours = ax.contour(
                self._theta_grid, 
                self._phi_grid, 
                field_data,
                levels=[0], 
                colors='k', 
                linewidths=2
            )
            plt.clabel(contours, inline=1, fontsize=10)
        
        # Draw grains with size based on absolute pressure
        for grain_id, grain_data in self.grains.items():
            theta, phi = grain_data['position']
            
            # Calculate pressure for this grain
            awareness = grain_data['awareness']
            activation = grain_data['activation']
            pressure = awareness * activation
            
            # Size based on absolute pressure
            size = 20 + 100 * abs(pressure)
            
            # Color and marker based on pressure sign
            if pressure > 0:
                color = 'blue'
                marker = 'o'  # Circle for expansion
            else:
                color = 'red'
                marker = 's'  # Square for contraction
            
            # Alpha based on absolute pressure
            alpha = 0.3 + 0.7 * min(1.0, abs(pressure))
            
            # Draw grain
            ax.scatter(
                theta, phi, 
                s=size, 
                color=color, 
                marker=marker,
                alpha=alpha, 
                edgecolors='black', 
                linewidths=0.5
            )
        
        # Draw vector field
        skip = max(1, self.resolution // 20)
        ax.quiver(
            self._theta_grid[::skip, ::skip],
            self._phi_grid[::skip, ::skip],
            self._vector_field['theta'][::skip, ::skip],
            self._vector_field['phi'][::skip, ::skip],
            color='k',
            scale=20,
            width=0.003,
            alpha=0.7
        )
        
        # Configure axes
        ax.set_xlabel('Theta')
        ax.set_ylabel('Phi')
        ax.set_xlim([0, 2*np.pi])
        ax.set_ylim([0, 2*np.pi])
        
        # Add tick marks at π/2 intervals
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add title
        plt.title(f'Pressure Field Visualization (t={self.time:.2f})')
        
        return fig, ax
    
    def create_distortion_visualization(self, **kwargs):
        """
        Create a visualization showing how the manifold is distorted by field values.
        
        Args:
            **kwargs: Additional parameters including:
                - distortion_factor: Factor controlling distortion magnitude (default: 0.3)
                - grid_resolution: Resolution of the grid (default: 20)
                - field_type: Field to use for distortion ('polarity', 'pressure', 'combined')
                - show_vectors: Whether to show vector field arrows (default: True)
                - show_zero_contour: Whether to show the zero contour line (default: True)
                
        Returns:
            Figure and axes objects
        """
        # Get parameters
        distortion_factor = kwargs.get('distortion_factor', 0.3)
        grid_resolution = kwargs.get('grid_resolution', 20)
        field_type = kwargs.get('field_type', 'combined')
        show_vectors = kwargs.get('show_vectors', True)
        show_zero_contour = kwargs.get('show_zero_contour', True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate base grid (undistorted)
        base_theta = np.linspace(0, 2*np.pi, grid_resolution)
        base_phi = np.linspace(0, 2*np.pi, grid_resolution)
        base_theta_grid, base_phi_grid = np.meshgrid(base_theta, base_phi)
        
        # Determine which field to use for distortion
        if field_type == 'polarity':
            distortion_field = self._field_data['polarity'].copy()
        elif field_type == 'pressure':
            distortion_field = self._field_data['pressure'].copy()
        elif field_type == 'activation':
            distortion_field = self._field_data['activation'].copy()
        else:  # 'combined'
            # Use a weighted combination
            distortion_field = (
                self._field_data['polarity'] * 0.6 + 
                self._field_data['pressure'] * 0.4
            )
        
        # Normalize distortion field for visualization
        distortion_max = np.max(np.abs(distortion_field))
        if distortion_max > 0:
            distortion_field = distortion_field / distortion_max
        
        # Interpolate distortion field to grid resolution
        try:
            from scipy.ndimage import zoom
            zoom_factor = grid_resolution / self.resolution
            distortion_on_grid = zoom(distortion_field, zoom_factor, order=1)
        except ImportError:
            # Simple interpolation if scipy not available
            distortion_on_grid = np.zeros((grid_resolution, grid_resolution))
            for i in range(grid_resolution):
                for j in range(grid_resolution):
                    # Map to indices in high-res field
                    hi_i = int((i / grid_resolution) * self.resolution)
                    hi_j = int((j / grid_resolution) * self.resolution)
                    # Sample high-res field
                    distortion_on_grid[i, j] = distortion_field[hi_i, hi_j]
        
        # Calculate distorted grid
        distorted_theta = base_theta_grid.copy()
        distorted_phi = base_phi_grid.copy()
        
        # Center of manifold (reference point for distortion)
        center_theta = np.pi
        center_phi = np.pi
        
        # Apply distortion - positive values push outward, negative pull inward
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                # Direction from center
                d_theta = base_theta_grid[i, j] - center_theta
                d_phi = base_phi_grid[i, j] - center_phi
                
                # Handle wrapping for directions
                if d_theta > np.pi: d_theta -= 2*np.pi
                if d_theta < -np.pi: d_theta += 2*np.pi
                if d_phi > np.pi: d_phi -= 2*np.pi
                if d_phi < -np.pi: d_phi += 2*np.pi
                
                # Normalize direction vector
                dist = np.sqrt(d_theta**2 + d_phi**2)
                if dist > 0:
                    d_theta /= dist
                    d_phi /= dist
                
                # Get distortion value at this point
                distortion = distortion_on_grid[i, j]
                
                # Apply distortion along radial direction
                # Positive distortion (expansion) pushes outward
                # Negative distortion (contraction) pulls inward
                distorted_theta[i, j] += d_theta * distortion * distortion_factor
                distorted_phi[i, j] += d_phi * distortion * distortion_factor
        
        # Plot the base grid (faintly)
        for i in range(grid_resolution):
            ax.plot(base_theta_grid[i, :], base_phi_grid[i, :], 'k-', alpha=0.1, linewidth=0.5)
            ax.plot(base_theta_grid[:, i], base_phi_grid[:, i], 'k-', alpha=0.1, linewidth=0.5)
        
        # Plot the distorted grid with color based on distortion
        # First create the color values
        cmap = self._colormaps.get('distortion')
        
        # Create norm for distortion
        vmin = np.min(distortion_on_grid)
        vmax = np.max(distortion_on_grid)
        vcenter = 0.0
        norm = create_safe_norm(vmin, vcenter, vmax)
        
        # Apply colormap
        colors = plt.cm.ScalarMappable(cmap=cmap, norm=norm).to_rgba(distortion_on_grid)
        
        # Plot horizontal grid lines (phi = constant)
        for i in range(grid_resolution):
            color_line = np.mean(colors[i, :], axis=0)
            ax.plot(distorted_theta[i, :], distorted_phi[i, :], '-', color=color_line, linewidth=1.5, alpha=0.7)
        
        # Plot vertical grid lines (theta = constant)
        for j in range(grid_resolution):
            color_line = np.mean(colors[:, j], axis=0)
            ax.plot(distorted_theta[:, j], distorted_phi[:, j], '-', color=color_line, linewidth=1.5, alpha=0.7)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Distortion (- Contraction, + Expansion)')
        
        # Add zero contour on original field if requested
        if show_zero_contour and np.min(distortion_field) < 0 and np.max(distortion_field) > 0:
            contours = ax.contour(
                base_theta_grid, 
                base_phi_grid, 
                distortion_on_grid,
                levels=[0], 
                colors='r', 
                linewidths=2
            )
            plt.clabel(contours, inline=1, fontsize=10, fmt='%1.1f')
        
        # Show vector field if requested
        if show_vectors:
            # Subsample for clearer vectors
            skip = max(1, grid_resolution // 10)
            
            # Calculate grid cell centers
            theta_centers = (base_theta[:-1] + base_theta[1:]) / 2
            phi_centers = (base_phi[:-1] + base_phi[1:]) / 2
            if len(theta_centers) < grid_resolution:
                theta_centers = np.append(theta_centers, base_theta[-1])
            if len(phi_centers) < grid_resolution:
                phi_centers = np.append(phi_centers, base_phi[-1])
            
            # Sample vector field
            vector_field_theta_sampled = np.zeros((grid_resolution, grid_resolution))
            vector_field_phi_sampled = np.zeros((grid_resolution, grid_resolution))
            
            for i in range(grid_resolution):
                for j in range(grid_resolution):
                    # Map to indices in high-res field
                    hi_i = int((i / grid_resolution) * self.resolution)
                    hi_j = int((j / grid_resolution) * self.resolution)
                    # Sample vector components
                    vector_field_theta_sampled[i, j] = self._vector_field['theta'][hi_i, hi_j]
                    vector_field_phi_sampled[i, j] = self._vector_field['phi'][hi_i, hi_j]
            
            # Draw vector field
            ax.quiver(
                base_theta_grid[::skip, ::skip],
                base_phi_grid[::skip, ::skip],
                vector_field_theta_sampled[::skip, ::skip],
                vector_field_phi_sampled[::skip, ::skip],
                color='k', 
                scale=20, 
                width=0.003, 
                alpha=0.7
            )
        
        # Draw grains
        for grain_id, grain_data in self.grains.items():
            theta, phi = grain_data['position']
            
            # Calculate size based on awareness and activation
            awareness = grain_data['awareness']
            activation = grain_data['activation']
            
            # Base size on absolute values to handle negative values properly
            size = 20 + 100 * abs(awareness) * abs(activation)
            
            # Get polarity for coloring
            polarity = grain_data['polarity']
            
            # Choose color based on polarity
            if polarity > 0.2:
                color = 'blue'  # Structure
            elif polarity < -0.2:
                color = 'red'   # Decay
            else:
                color = 'green' # Neutral
            
            # Draw grain
            ax.scatter(
                theta, phi, 
                s=size, 
                color=color, 
                alpha=0.7, 
                edgecolors='black', 
                linewidths=0.5
            )
        
        # Configure axes
        ax.set_xlabel('Theta')
        ax.set_ylabel('Phi')
        ax.set_xlim([0, 2*np.pi])
        ax.set_ylim([0, 2*np.pi])
        
        # Add tick marks at π/2 intervals
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add title
        field_label = 'Combined' if field_type == 'combined' else field_type.capitalize()
        plt.title(f'Manifold Distortion Visualization ({field_label} Field, t={self.time:.2f})')
        
        return fig, ax
    
    def create_toroidal_visualization(self, **kwargs):
        """
        Create a 3D toroidal visualization of the manifold.
        
        Args:
            **kwargs: Additional parameters including:
                - field_name: Field to use for coloring (default: 'polarity')
                - major_radius: Major radius of torus (default: 3.0)
                - minor_radius: Minor radius of torus (default: 1.0)
                - show_grains: Whether to show grains (default: True)
                - show_vectors: Whether to show vector field arrows (default: False)
                
        Returns:
            Figure and axes objects
        """
        # Get parameters
        field_name = kwargs.get('field_name', 'polarity')
        major_radius = kwargs.get('major_radius', 3.0)
        minor_radius = kwargs.get('minor_radius', 1.0)
        show_grains = kwargs.get('show_grains', True)
        show_vectors = kwargs.get('show_vectors', False)
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get field data
        field_data = self._field_data[field_name]
        
        # Get colormap
        cmap = self._colormaps.get(field_name)
        
        # Get norm for signed fields
        norm = None
        if field_name in self._norms:
            # For signed fields, use TwoSlopeNorm with 0 as center
            field_min = self._field_stats[field_name]['min']
            field_max = self._field_stats[field_name]['max']
            
            # Only use signed norm if field actually crosses zero
            if field_min < 0 and field_max > 0:
                norm = create_safe_norm(
                    vmin=field_min,
                    vcenter=0.0,
                    vmax=field_max
                )
        
        # Generate toroidal coordinates
        theta = np.linspace(0, 2*np.pi, self.resolution)
        phi = np.linspace(0, 2*np.pi, self.resolution)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        
        # Convert to Cartesian coordinates
        x = (major_radius + minor_radius * np.cos(phi_grid)) * np.cos(theta_grid)
        y = (major_radius + minor_radius * np.cos(phi_grid)) * np.sin(theta_grid)
        z = minor_radius * np.sin(phi_grid)
        
        # Plot surface
        if norm:
            surf = ax.plot_surface(x, y, z, facecolors=cmap(norm(field_data)), alpha=0.8)
        else:
            surf = ax.plot_surface(x, y, z, facecolors=cmap(field_data), alpha=0.8)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(field_data)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(self.field_titles.get(field_name, field_name.capitalize()))
        
        # Draw grains if requested
        if show_grains:
            for grain_id, grain_data in self.grains.items():
                theta, phi = grain_data['position']
                
                # Convert to Cartesian
                x_pos = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
                y_pos = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
                z_pos = minor_radius * np.sin(phi)
                
                # Calculate size
                awareness = grain_data['awareness']
                activation = grain_data['activation']
                size = 50 * abs(awareness) * abs(activation)
                
                # Get polarity for coloring
                polarity = grain_data['polarity']
                
                # Choose color based on polarity
                if polarity > 0.2:
                    color = 'blue'  # Structure
                elif polarity < -0.2:
                    color = 'red'   # Decay
                else:
                    color = 'green' # Neutral
                
                # Draw grain
                ax.scatter(
                    x_pos, y_pos, z_pos,
                    s=size,
                    color=color,
                    edgecolors='black',
                    linewidths=0.5,
                    alpha=0.7
                )
        
        # Show vectors if requested
        if show_vectors:
            # Subsample for clearer vectors
            skip = max(1, self.resolution // 10)
            
            # Get vector field components
            vec_theta = self._vector_field['theta'][::skip, ::skip]
            vec_phi = self._vector_field['phi'][::skip, ::skip]
            
            # Calculate 3D vector components
            # This is a simplified conversion from toroidal to Cartesian vectors
            
            # First, get the base positions
            sampled_theta = theta_grid[::skip, ::skip]
            sampled_phi = phi_grid[::skip, ::skip]
            
            x_pos = (major_radius + minor_radius * np.cos(sampled_phi)) * np.cos(sampled_theta)
            y_pos = (major_radius + minor_radius * np.cos(sampled_phi)) * np.sin(sampled_theta)
            z_pos = minor_radius * np.sin(sampled_phi)
            
            # Calculate vector projections into Cartesian space
            # These are approximate projections that preserve general flow
            vec_x = -vec_theta * np.sin(sampled_theta) - vec_phi * np.cos(sampled_theta) * np.sin(sampled_phi)
            vec_y = vec_theta * np.cos(sampled_theta) - vec_phi * np.sin(sampled_theta) * np.sin(sampled_phi)
            vec_z = vec_phi * np.cos(sampled_phi)
            
            # Scale vector magnitude for visualization
            scale_factor = 0.2 * minor_radius
            vec_x *= scale_factor
            vec_y *= scale_factor
            vec_z *= scale_factor
            
            # Draw vectors
            ax.quiver(
                x_pos, y_pos, z_pos,
                vec_x, vec_y, vec_z,
                color='k',
                alpha=0.6,
                linewidth=0.5
            )
        
        # Configure axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        max_range = max([
            np.max(x) - np.min(x),
            np.max(y) - np.min(y),
            np.max(z) - np.min(z)
        ])
        mid_x = (np.max(x) + np.min(x)) * 0.5
        mid_y = (np.max(y) + np.min(y)) * 0.5
        mid_z = (np.max(z) + np.min(z)) * 0.5
        ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
        ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
        ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
        
        # Add title
        plt.title(f'Toroidal Visualization ({field_name.capitalize()} Field, t={self.time:.2f})')
        
        return fig, ax
    
    def create_field_history_visualization(self, field_name='polarity', **kwargs):
        """
        Create a visualization showing how a field has changed over time.
        
        Args:
            field_name: Name of the field to track (default: 'polarity')
            **kwargs: Additional parameters
                
        Returns:
            Figure and axes objects
        """
        # Check if field history is available
        if not self.field_history or field_name not in self.field_history:
            # Create figure with error message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5, 0.5, 
                "Field history not available.\nEnable continuous_update_mode to track field history.", 
                ha='center', 
                va='center'
            )
            ax.set_title(f"Field History Visualization ({field_name})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return fig, ax
        
        # Get field history
        history = self.field_history[field_name]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract data
        times = [entry['time'] for entry in history]
        means = [entry['mean'] for entry in history]
        mins = [entry['min'] for entry in history]
        maxs = [entry['max'] for entry in history]
        
        # Plot mean with min/max range
        ax.plot(times, means, 'b-', linewidth=2, label=f'Mean {field_name}')
        ax.fill_between(times, mins, maxs, color='b', alpha=0.2, label='Min/Max Range')
        
        # Add zero line for signed fields
        if field_name in ['polarity', 'curvature', 'activation', 'pressure']:
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero')
        
        # Configure axes
        ax.set_xlabel('Time')
        ax.set_ylabel(field_name.capitalize())
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add title
        plt.title(f'{self.field_titles.get(field_name, field_name.capitalize())} History')
        
        return fig, ax
    
    def _toroidal_to_cartesian(self, theta, phi, major_radius=3.0, minor_radius=1.0):
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