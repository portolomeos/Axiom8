"""
Collapse Geometry Maps - Unwrapped Torus and Cartesian Projection Visualizer

This module provides enhanced visualization of the Collapse Geometry framework,
featuring both unwrapped torus (theta-phi) maps and Cartesian projections of the
toroidal manifold. It extends the core visualizer with specialized map renderers.

Updated to be compatible with the enhanced CollapseVisualizerBase implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import math
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

# Import from the enhanced visualizer core
# from axiom8.visualizer.visualizer_core import CollapseVisualizerBase

class CollapseGeometryMapVisualizer:
    """
    Enhanced visualization class for Collapse Geometry that focuses on topological
    mapping between different coordinate representations of the manifold.
    """
    
    def __init__(self, base_visualizer):
        """
        Initialize with a base visualizer instance.
        
        Args:
            base_visualizer: Instance of CollapseVisualizerBase
        """
        self.visualizer = base_visualizer
        self.available_fields = list(self.visualizer._field_data.keys())
    
    def create_unwrapped_torus_map(self, field_name='awareness', figsize=(12, 8), 
                                  show_grains=True, show_vectors=True, show_structures=True):
        """
        Create an unwrapped torus (theta-phi) map visualization.
        
        Args:
            field_name: Name of the field to visualize
            figsize: Figure size (width, height) in inches
            show_grains: Whether to show grains on the map
            show_vectors: Whether to show vector field
            show_structures: Whether to show vortices and pathways
            
        Returns:
            Figure and axes objects
        """
        # Validate field name
        if field_name not in self.available_fields:
            print(f"Warning: Field '{field_name}' not found. Using 'awareness' instead.")
            field_name = 'awareness'
        
        # Get field data
        field_data = self.visualizer._field_data.get(field_name, 
                                                   np.zeros((self.visualizer.resolution, self.visualizer.resolution)))
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get appropriate colormap and norm
        cmap = self.visualizer._colormaps.get(field_name, cm.viridis)
        
        # Get norm for signed fields
        norm = None
        if field_name in ['polarity', 'curvature', 'activation', 'pressure'] and self.visualizer.signed_field_visualization:
            # For signed fields, use TwoSlopeNorm with 0 as center
            field_min = self.visualizer._field_stats[field_name]['min']
            field_max = self.visualizer._field_stats[field_name]['max']
            
            # Only use signed norm if field actually crosses zero
            if field_min < 0 and field_max > 0:
                from axiom8.visualizer.visualizer_core import create_safe_norm
                norm = create_safe_norm(
                    vmin=field_min,
                    vcenter=0.0,
                    vmax=field_max
                )
        
        # Plot the field as a color mesh
        im = ax.pcolormesh(self.visualizer._theta_grid, self.visualizer._phi_grid, 
                         field_data, cmap=cmap, norm=norm, shading='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(self.visualizer.field_titles.get(field_name, field_name.capitalize()))
        
        # Special contour for signed fields
        if field_name in ['polarity', 'curvature', 'activation', 'pressure'] and self.visualizer.signed_field_visualization:
            if np.min(field_data) < 0 and np.max(field_data) > 0:
                contours = ax.contour(self.visualizer._theta_grid, self.visualizer._phi_grid, 
                                    field_data, levels=[0], colors='k', linewidths=2)
                plt.clabel(contours, inline=1, fontsize=10)
        
        # Show vector field if requested
        if show_vectors:
            # Subsample for clearer vectors
            skip = max(1, self.visualizer.resolution // 20)
            ax.quiver(self.visualizer._theta_grid[::skip, ::skip], 
                     self.visualizer._phi_grid[::skip, ::skip],
                     self.visualizer._vector_field['theta'][::skip, ::skip], 
                     self.visualizer._vector_field['phi'][::skip, ::skip],
                     color='k', scale=20, width=0.003, alpha=0.7)
        
        # Show grains if requested
        if show_grains:
            # Iterate through all grains
            for grain_id, grain_data in self.visualizer.grains.items():
                theta, phi = grain_data['position']
                
                # Calculate size based on properties
                awareness = grain_data['awareness']
                activation = grain_data['activation']
                
                # Base size on absolute value of awareness and activation
                size = 20 + 100 * abs(awareness) * abs(activation)
                
                # Get polarity for coloring
                polarity = grain_data['polarity']
                
                # Choose color based on polarity
                if polarity > 0.2:
                    color = 'blue'  # Structure/order
                elif polarity < -0.2:
                    color = 'red'   # Decay/chaos
                else:
                    color = 'green' # Neutral
                
                # Draw grain
                ax.scatter(theta, phi, s=size, color=color, alpha=0.7, 
                          edgecolors='black', linewidths=0.5)
                
                # Add grain ID label for larger grains
                if size > 100:
                    ax.text(theta, phi, f"{grain_id}", fontsize=8, 
                           ha='center', va='center', color='white')
        
        # Show structural patterns if requested
        if show_structures:
            # Show vortices
            for vortex in self.visualizer.vortices:
                # Check if vortex has a position (support both object and dict structures)
                if hasattr(vortex, 'position') and isinstance(vortex.position, tuple):
                    vortex_theta, vortex_phi = vortex.position
                    vortex_type = getattr(vortex, 'vortex_type', 'unknown')
                elif isinstance(vortex, dict) and 'center_id' in vortex:
                    # Get position from center grain
                    center_id = vortex['center_id']
                    if center_id in self.visualizer.grain_positions:
                        vortex_theta, vortex_phi = self.visualizer.grain_positions[center_id]
                    else:
                        continue
                    vortex_type = vortex.get('direction', 'unknown')
                elif isinstance(vortex, dict) and all(k in vortex for k in ['theta', 'phi']):
                    vortex_theta = vortex['theta']
                    vortex_phi = vortex['phi']
                    vortex_type = vortex.get('direction', 'unknown')
                else:
                    continue
                    
                # Determine marker based on vortex type
                if vortex_type == 'structure' or vortex_type == 'clockwise':
                    marker = '*'
                    color = 'cyan'
                elif vortex_type == 'decay' or vortex_type == 'counterclockwise':
                    marker = 'X'
                    color = 'magenta'
                else:
                    marker = 'P'
                    color = 'yellow'
                
                # Draw vortex marker
                ax.scatter(vortex_theta, vortex_phi, s=200, marker=marker, 
                          color=color, edgecolors='black', linewidths=1, alpha=0.8)
                    
            # Show lightlike pathways
            for pathway_type, pathways in self.visualizer.lightlike_pathways.items():
                for pathway in pathways:
                    # Check if pathway has path attribute or is a dict with nodes
                    if hasattr(pathway, 'path') and isinstance(pathway.path, list):
                        # Get path points
                        points = pathway.path
                        
                        # Extract theta and phi coordinates
                        thetas = [point[0] for point in points if isinstance(point, tuple) and len(point) == 2]
                        phis = [point[1] for point in points if isinstance(point, tuple) and len(point) == 2]
                    elif isinstance(pathway, dict) and 'nodes' in pathway:
                        # Get node positions
                        node_ids = pathway['nodes']
                        
                        # Extract positions of each node
                        thetas = []
                        phis = []
                        
                        for node_id in node_ids:
                            if node_id in self.visualizer.grain_positions:
                                theta, phi = self.visualizer.grain_positions[node_id]
                                thetas.append(theta)
                                phis.append(phi)
                    elif isinstance(pathway, dict) and 'node_positions' in pathway:
                        # Direct node positions
                        positions = pathway['node_positions']
                        
                        # Extract theta and phi
                        thetas = [pos[0] for pos in positions if isinstance(pos, tuple) and len(pos) == 2]
                        phis = [pos[1] for pos in positions if isinstance(pos, tuple) and len(pos) == 2]
                    else:
                        continue
                        
                    # Choose color based on pathway type
                    if pathway_type == 'structure':
                        color = 'blue'
                        linestyle = '-'
                    else:  # 'decay'
                        color = 'red'
                        linestyle = '--'
                    
                    # Draw pathway line
                    if thetas and phis:
                        ax.plot(thetas, phis, color=color, linestyle=linestyle, 
                               linewidth=2, alpha=0.7)
        
        # Draw recursive patterns if available
        if hasattr(self.visualizer, 'recursive_patterns') and self.visualizer.recursive_patterns:
            for pattern in self.visualizer.recursive_patterns:
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
        ax.set_title(f'Unwrapped Torus Map: {self.visualizer.field_titles.get(field_name, field_name.capitalize())} (t={self.visualizer.time:.2f})')
        
        return fig, ax
    
    def create_cartesian_projection_map(self, field_name='awareness', figsize=(10, 10), 
                                       show_grains=True, show_vectors=False, show_structures=True,
                                       projection_style='circular'):
        """
        Create a Cartesian projection map visualization.
        
        Args:
            field_name: Name of the field to visualize
            figsize: Figure size (width, height) in inches
            show_grains: Whether to show grains on the map
            show_vectors: Whether to show vector field (simplified for clarity)
            show_structures: Whether to show vortices and pathways
            projection_style: 'circular' or 'square' projection
            
        Returns:
            Figure and axes objects
        """
        # Validate field name
        if field_name not in self.available_fields:
            print(f"Warning: Field '{field_name}' not found. Using 'awareness' instead.")
            field_name = 'awareness'
        
        # Check if Cartesian projections available
        if not hasattr(self.visualizer, '_cartesian_projections') or not self.visualizer._cartesian_projections:
            # Generate projections if not already done
            self.visualizer._generate_cartesian_projections()
        
        # Get field data
        if field_name in self.visualizer._cartesian_projections:
            field_data = self.visualizer._cartesian_projections[field_name]
        else:
            print(f"Warning: Cartesian projection for '{field_name}' not found. Using 'awareness' instead.")
            field_name = 'awareness'
            field_data = self.visualizer._cartesian_projections.get('awareness', np.zeros((self.visualizer.resolution, self.visualizer.resolution)))
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get appropriate colormap and norm
        cmap = self.visualizer._colormaps.get(field_name, cm.viridis)
        
        # Get norm for signed fields
        norm = None
        if field_name in ['polarity', 'curvature', 'activation', 'pressure'] and self.visualizer.signed_field_visualization:
            # For signed fields, use TwoSlopeNorm with 0 as center
            field_min = self.visualizer._field_stats[field_name]['min']
            field_max = self.visualizer._field_stats[field_name]['max']
            
            # Only use signed norm if field actually crosses zero
            if field_min < 0 and field_max > 0:
                from axiom8.visualizer.visualizer_core import create_safe_norm
                norm = create_safe_norm(
                    vmin=field_min,
                    vcenter=0.0,
                    vmax=field_max
                )
        
        # Create mask for circular projection if needed
        if projection_style == 'circular':
            # Create circular mask
            mask = np.zeros_like(field_data, dtype=bool)
            center = self.visualizer.resolution // 2
            for i in range(self.visualizer.resolution):
                for j in range(self.visualizer.resolution):
                    # Calculate distance from center
                    r = np.sqrt((i - center)**2 + (j - center)**2)
                    # Set mask based on distance
                    mask[i, j] = r <= center * 0.98
            
            # Apply mask to data
            masked_data = np.copy(field_data)
            masked_data[~mask] = np.nan
            
            # Plot masked data
            im = ax.imshow(masked_data, cmap=cmap, norm=norm, origin='lower', 
                          extent=[-1, 1, -1, 1], interpolation='bilinear')
            
            # Add circle outline
            circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='-', 
                              linewidth=1.5)
            ax.add_artist(circle)
        else:
            # Standard square projection
            im = ax.imshow(field_data, cmap=cmap, norm=norm, origin='lower', 
                          extent=[-1, 1, -1, 1], interpolation='bilinear')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(self.visualizer.field_titles.get(field_name, field_name.capitalize()))
        
        # Special contour for signed fields
        if field_name in ['polarity', 'curvature', 'activation', 'pressure'] and self.visualizer.signed_field_visualization:
            if np.nanmin(field_data) < 0 and np.nanmax(field_data) > 0:
                contours = ax.contour(self.visualizer._x_grid, self.visualizer._y_grid, 
                                    field_data, levels=[0], colors='k', linewidths=2)
                plt.clabel(contours, inline=1, fontsize=10)
        
        # Show grains if requested
        if show_grains:
            # Need to project grain positions to Cartesian space
            for grain_id, grain_data in self.visualizer.grains.items():
                theta, phi = grain_data['position']
                
                # Project to Cartesian coordinates (simple disk projection)
                # phi determines radial distance from center
                r = phi / (2 * np.pi)  # Normalize to [0, 1]
                
                # Calculate x, y (standard polar to Cartesian)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                
                # Calculate size based on properties
                awareness = grain_data['awareness']
                activation = grain_data['activation']
                
                # Base size on absolute value of awareness and activation
                size = 20 + 100 * abs(awareness) * abs(activation) * 0.7  # Scale down a bit
                
                # Get polarity for coloring
                polarity = grain_data['polarity']
                
                # Choose color based on polarity
                if polarity > 0.2:
                    color = 'blue'  # Structure/order
                elif polarity < -0.2:
                    color = 'red'   # Decay/chaos
                else:
                    color = 'green' # Neutral
                
                # Draw grain
                ax.scatter(x, y, s=size, color=color, alpha=0.7, 
                          edgecolors='black', linewidths=0.5)
                
                # Add grain ID label for larger grains
                if size > 70:
                    ax.text(x, y, f"{grain_id}", fontsize=8, 
                           ha='center', va='center', color='white')
        
        # Show structural patterns if requested
        if show_structures:
            # Project vortices to Cartesian space
            for vortex in self.visualizer.vortices:
                # Check if vortex has a position (support both object and dict structures)
                if hasattr(vortex, 'position') and isinstance(vortex.position, tuple):
                    vortex_theta, vortex_phi = vortex.position
                    vortex_type = getattr(vortex, 'vortex_type', 'unknown')
                elif isinstance(vortex, dict) and 'center_id' in vortex:
                    # Get position from center grain
                    center_id = vortex['center_id']
                    if center_id in self.visualizer.grain_positions:
                        vortex_theta, vortex_phi = self.visualizer.grain_positions[center_id]
                    else:
                        continue
                    vortex_type = vortex.get('direction', 'unknown')
                elif isinstance(vortex, dict) and all(k in vortex for k in ['theta', 'phi']):
                    vortex_theta = vortex['theta']
                    vortex_phi = vortex['phi']
                    vortex_type = vortex.get('direction', 'unknown')
                else:
                    continue
                    
                # Project to Cartesian
                r = vortex_phi / (2 * np.pi)
                vortex_x = r * np.cos(vortex_theta)
                vortex_y = r * np.sin(vortex_theta)
                
                # Determine marker based on vortex type
                if vortex_type == 'structure' or vortex_type == 'clockwise':
                    marker = '*'
                    color = 'cyan'
                elif vortex_type == 'decay' or vortex_type == 'counterclockwise':
                    marker = 'X'
                    color = 'magenta'
                else:
                    marker = 'P'
                    color = 'yellow'
                
                # Draw vortex marker
                ax.scatter(vortex_x, vortex_y, s=200, marker=marker, 
                          color=color, edgecolors='black', linewidths=1, alpha=0.8)
                    
            # Project recursive patterns if available
            if hasattr(self.visualizer, 'recursive_patterns') and self.visualizer.recursive_patterns:
                for pattern in self.visualizer.recursive_patterns:
                    theta, phi = pattern['position']
                    strength = pattern['strength']
                    is_circular = pattern.get('is_circular', False)
                    
                    # Project to Cartesian
                    r = phi / (2 * np.pi)
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    
                    # Size based on strength
                    size = 50 + 100 * strength
                    
                    # Different marker for circular vs self-referential
                    marker = 'o' if is_circular else '*'
                    
                    # Draw pattern
                    ax.scatter(
                        x, y,
                        s=size * 0.7,  # Scale down a bit for Cartesian
                        color='gold' if is_circular else 'orange',
                        marker=marker,
                        alpha=0.8,
                        edgecolors='black',
                        linewidths=1.0,
                        zorder=10  # Ensure drawn on top
                    )
                    
                    # Draw circle around recursive pattern
                    circle = plt.Circle(
                        (x, y),
                        radius=(0.2 + 0.3 * strength) * r,  # Scale by radius
                        fill=False,
                        color='gold' if is_circular else 'orange',
                        linestyle='--',
                        alpha=0.6
                    )
                    ax.add_patch(circle)
                    
            # Project lightlike pathways to Cartesian space
            for pathway_type, pathways in self.visualizer.lightlike_pathways.items():
                for pathway in pathways:
                    # Check if pathway has path attribute or is a dict with nodes
                    if hasattr(pathway, 'path') and isinstance(pathway.path, list):
                        # Get path points
                        points = pathway.path
                        
                        # Project points to Cartesian
                        cart_points = []
                        for point in points:
                            if isinstance(point, tuple) and len(point) == 2:
                                theta, phi = point
                                r = phi / (2 * np.pi)
                                x = r * np.cos(theta)
                                y = r * np.sin(theta)
                                cart_points.append((x, y))
                    elif isinstance(pathway, dict) and 'nodes' in pathway:
                        # Get node positions
                        node_ids = pathway['nodes']
                        
                        # Project each node to Cartesian
                        cart_points = []
                        
                        for node_id in node_ids:
                            if node_id in self.visualizer.grain_positions:
                                theta, phi = self.visualizer.grain_positions[node_id]
                                r = phi / (2 * np.pi)
                                x = r * np.cos(theta)
                                y = r * np.sin(theta)
                                cart_points.append((x, y))
                    elif isinstance(pathway, dict) and 'node_positions' in pathway:
                        # Direct node positions
                        positions = pathway['node_positions']
                        
                        # Project to Cartesian
                        cart_points = []
                        
                        for pos in positions:
                            if isinstance(pos, tuple) and len(pos) == 2:
                                theta, phi = pos
                                r = phi / (2 * np.pi)
                                x = r * np.cos(theta)
                                y = r * np.sin(theta)
                                cart_points.append((x, y))
                    else:
                        continue
                        
                    # Extract x and y coordinates
                    xs = [point[0] for point in cart_points]
                    ys = [point[1] for point in cart_points]
                    
                    # Choose color based on pathway type
                    if pathway_type == 'structure':
                        color = 'blue'
                        linestyle = '-'
                    else:  # 'decay'
                        color = 'red'
                        linestyle = '--'
                    
                    # Draw pathway line
                    if xs and ys:
                        ax.plot(xs, ys, color=color, linestyle=linestyle, 
                               linewidth=2, alpha=0.7)
        
        # Configure axes
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add title
        projection_name = 'Circular' if projection_style == 'circular' else 'Square'
        ax.set_title(f'{projection_name} Cartesian Projection: {self.visualizer.field_titles.get(field_name, field_name.capitalize())} (t={self.visualizer.time:.2f})')
        
        return fig, ax
    
    def create_dual_map_visualization(self, field_name='awareness', figsize=(18, 9)):
        """
        Create a dual visualization showing both unwrapped torus and Cartesian projection.
        
        Args:
            field_name: Name of the field to visualize
            figsize: Figure size (width, height) in inches
            
        Returns:
            Figure object
        """
        # Validate field name
        if field_name not in self.available_fields:
            print(f"Warning: Field '{field_name}' not found. Using 'awareness' instead.")
            field_name = 'awareness'
            
        # Create figure with GridSpec for layout
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 2, width_ratios=[1.2, 1])
        
        # Create first axis for unwrapped torus map
        ax1 = fig.add_subplot(gs[0])
        
        # Create second axis for Cartesian projection
        ax2 = fig.add_subplot(gs[1])
        
        # Get field data
        field_data = self.visualizer._field_data[field_name]
        
        # Get appropriate colormap and norm
        cmap = self.visualizer._colormaps.get(field_name, cm.viridis)
        
        # Get norm for signed fields
        norm = None
        if field_name in ['polarity', 'curvature', 'activation', 'pressure'] and self.visualizer.signed_field_visualization:
            # For signed fields, use TwoSlopeNorm with 0 as center
            field_min = self.visualizer._field_stats[field_name]['min']
            field_max = self.visualizer._field_stats[field_name]['max']
            
            # Only use signed norm if field actually crosses zero
            if field_min < 0 and field_max > 0:
                from axiom8.visualizer.visualizer_core import create_safe_norm
                norm = create_safe_norm(
                    vmin=field_min,
                    vcenter=0.0,
                    vmax=field_max
                )
        
        # Plot unwrapped torus map
        im1 = ax1.pcolormesh(self.visualizer._theta_grid, self.visualizer._phi_grid, 
                           field_data, cmap=cmap, norm=norm, shading='auto')
        
        # Special contour for signed fields on unwrapped map
        if field_name in ['polarity', 'curvature', 'activation', 'pressure'] and self.visualizer.signed_field_visualization:
            if np.min(field_data) < 0 and np.max(field_data) > 0:
                contours = ax1.contour(self.visualizer._theta_grid, self.visualizer._phi_grid, 
                                     field_data, levels=[0], colors='k', linewidths=2)
                plt.clabel(contours, inline=1, fontsize=8)
        
        # Show vector field (subsampled)
        skip = max(1, self.visualizer.resolution // 20)
        ax1.quiver(self.visualizer._theta_grid[::skip, ::skip], 
                  self.visualizer._phi_grid[::skip, ::skip],
                  self.visualizer._vector_field['theta'][::skip, ::skip], 
                  self.visualizer._vector_field['phi'][::skip, ::skip],
                  color='k', scale=20, width=0.003, alpha=0.5)
        
        # Configure unwrapped torus axes
        ax1.set_xlabel('Theta')
        ax1.set_ylabel('Phi')
        ax1.set_xlim([0, 2*np.pi])
        ax1.set_ylim([0, 2*np.pi])
        
        # Add tick marks at π/2 intervals
        ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax1.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax1.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax1.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        
        # Add grid to unwrapped torus map
        ax1.grid(True, alpha=0.3)
        
        # Add title to unwrapped torus map
        ax1.set_title(f'Unwrapped Torus Map (θ-φ)')
        
        # --- Create Cartesian Projection ---
        # Check if Cartesian projections available
        if not hasattr(self.visualizer, '_cartesian_projections') or not self.visualizer._cartesian_projections:
            # Generate projections if not already done
            self.visualizer._generate_cartesian_projections()
        
        # Get Cartesian field data
        cartesian_data = self.visualizer._cartesian_projections.get(
            field_name, 
            np.zeros((self.visualizer.resolution, self.visualizer.resolution))
        )
        
        # Create circular mask
        mask = np.zeros_like(cartesian_data, dtype=bool)
        center = self.visualizer.resolution // 2
        for i in range(self.visualizer.resolution):
            for j in range(self.visualizer.resolution):
                # Calculate distance from center
                r = np.sqrt((i - center)**2 + (j - center)**2)
                # Set mask based on distance
                mask[i, j] = r <= center * 0.98
        
        # Apply mask to data
        masked_data = np.copy(cartesian_data)
        masked_data[~mask] = np.nan
        
        # Plot Cartesian projection
        im2 = ax2.imshow(masked_data, cmap=cmap, norm=norm, origin='lower', 
                        extent=[-1, 1, -1, 1], interpolation='bilinear')
        
        # Add circle outline
        circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='-', 
                          linewidth=1.5)
        ax2.add_artist(circle)
        
        # Special contour for signed fields on Cartesian projection
        if field_name in ['polarity', 'curvature', 'activation', 'pressure'] and self.visualizer.signed_field_visualization:
            if np.nanmin(cartesian_data) < 0 and np.nanmax(cartesian_data) > 0:
                contours = ax2.contour(self.visualizer._x_grid, self.visualizer._y_grid, 
                                     cartesian_data, levels=[0], colors='k', linewidths=2)
                plt.clabel(contours, inline=1, fontsize=8)
        
        # Configure Cartesian projection axes
        ax2.set_aspect('equal')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        
        # Add grid to Cartesian projection
        ax2.grid(True, alpha=0.3)
        
        # Add title to Cartesian projection
        ax2.set_title('Cartesian Projection (Disk)')
        
        # --- Add Grains to Both Maps ---
        # Draw grains on both maps
        for grain_id, grain_data in self.visualizer.grains.items():
            theta, phi = grain_data['position']
            
            # Calculate size based on properties
            awareness = grain_data['awareness']
            activation = grain_data['activation']
            
            # Base size on absolute value of awareness and activation
            size = 20 + 100 * abs(awareness) * abs(activation)
            size_cart = size * 0.7  # Scale down a bit for Cartesian
            
            # Get polarity for coloring
            polarity = grain_data['polarity']
            
            # Choose color based on polarity
            if polarity > 0.2:
                color = 'blue'  # Structure/order
            elif polarity < -0.2:
                color = 'red'   # Decay/chaos
            else:
                color = 'green' # Neutral
            
            # Draw grain on unwrapped torus map
            ax1.scatter(theta, phi, s=size, color=color, alpha=0.7, 
                       edgecolors='black', linewidths=0.5)
            
            # Project to Cartesian coordinates for second map
            r = phi / (2 * np.pi)  # Normalize to [0, 1]
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Draw grain on Cartesian projection
            ax2.scatter(x, y, s=size_cart, color=color, alpha=0.7, 
                       edgecolors='black', linewidths=0.5)
            
            # Add labels for significant grains
            if size > 100:
                ax1.text(theta, phi, f"{grain_id}", fontsize=8, 
                        ha='center', va='center', color='white')
                ax2.text(x, y, f"{grain_id}", fontsize=8, 
                        ha='center', va='center', color='white')
        
        # --- Add Correspondence Lines ---
        # Add sample correspondence lines between the two maps
        # Use evenly spaced points around the torus
        n_correspondence = 8  # Number of correspondence points
        for i in range(n_correspondence):
            # Calculate theta positions (evenly spaced around)
            theta = i * 2 * np.pi / n_correspondence
            
            # Three phi positions to show inner, middle, and outer regions
            for j, phi_factor in enumerate([0.25, 0.5, 0.75]):
                phi = phi_factor * 2 * np.pi
                
                # Project to Cartesian
                r = phi / (2 * np.pi)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                
                # Determine color based on position
                colors = ['#FFD700', '#FF6347', '#7FFFD4']  # Gold, tomato, aquamarine
                color = colors[j % len(colors)]
                
                # Draw point on unwrapped map
                ax1.scatter(theta, phi, s=25, color=color, edgecolors='black', linewidths=0.5, 
                           marker='o', alpha=0.8, zorder=5)
                
                # Draw point on Cartesian map
                ax2.scatter(x, y, s=25, color=color, edgecolors='black', linewidths=0.5, 
                           marker='o', alpha=0.8, zorder=5)
        
        # --- Finalize Figure ---
        # Add colorbar between the plots
        cbar_ax = fig.add_axes([0.46, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = plt.colorbar(im1, cax=cbar_ax)
        cbar.set_label(self.visualizer.field_titles.get(field_name, field_name.capitalize()))
        
        # Add main title
        fig.suptitle(f'Dual Mapping Visualization: {self.visualizer.field_titles.get(field_name, field_name.capitalize())} (t={self.visualizer.time:.2f})', 
                    fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
        
        return fig
        
    def create_field_comparison_visualization(self, field_names=None, figsize=(18, 12)):
        """
        Create a comparison visualization of multiple fields.
        
        Args:
            field_names: List of field names to compare (or None for default selection)
            figsize: Figure size (width, height) in inches
            
        Returns:
            Figure object
        """
        # Default field selection if none provided
        if field_names is None:
            field_names = ['awareness', 'polarity', 'phase', 'activation', 'pressure', 'curvature']
        
        # Limit number of fields to 6 for readability
        if len(field_names) > 6:
            print(f"Warning: Too many fields selected ({len(field_names)}). Limiting to first 6.")
            field_names = field_names[:6]
        
        # Validate field names
        valid_fields = []
        for name in field_names:
            if name in self.available_fields:
                valid_fields.append(name)
            else:
                print(f"Warning: Field '{name}' not found. Skipping.")
        field_names = valid_fields
        
        # If no valid fields, return None
        if not field_names:
            print("Error: No valid fields to visualize.")
            return None
        
        # Determine grid layout
        if len(field_names) <= 3:
            rows, cols = 1, len(field_names)
        else:
            rows = 2
            cols = (len(field_names) + 1) // 2  # Ceiling division
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Handle case with single field
        if len(field_names) == 1:
            axes = np.array([axes])
        
        # Flatten axes for easier indexing
        if rows > 1 or cols > 1:
            axes = axes.flatten()
        
        # Create visualizations for each field
        for i, field_name in enumerate(field_names):
            ax = axes[i]
            
            # Get field data
            field_data = self.visualizer._field_data[field_name]
            
            # Get appropriate colormap and norm
            cmap = self.visualizer._colormaps.get(field_name, cm.viridis)
            
            # Get norm for signed fields
            norm = None
            if field_name in ['polarity', 'curvature', 'activation', 'pressure'] and self.visualizer.signed_field_visualization:
                # For signed fields, use TwoSlopeNorm with 0 as center
                field_min = self.visualizer._field_stats[field_name]['min']
                field_max = self.visualizer._field_stats[field_name]['max']
                
                # Only use signed norm if field actually crosses zero
                if field_min < 0 and field_max > 0:
                    from axiom8.visualizer.visualizer_core import create_safe_norm
                    norm = create_safe_norm(
                        vmin=field_min,
                        vcenter=0.0,
                        vmax=field_max
                    )
            
            # Plot the field
            im = ax.pcolormesh(self.visualizer._theta_grid, self.visualizer._phi_grid, 
                             field_data, cmap=cmap, norm=norm, shading='auto')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label=field_name.capitalize())
            
            # Special contour for signed fields
            if field_name in ['polarity', 'curvature', 'activation', 'pressure'] and self.visualizer.signed_field_visualization:
                if np.min(field_data) < 0 and np.max(field_data) > 0:
                    contours = ax.contour(self.visualizer._theta_grid, self.visualizer._phi_grid, 
                                        field_data, levels=[0], colors='k', linewidths=1)
            
            # Draw grains
            for grain_id, grain_data in self.visualizer.grains.items():
                theta, phi = grain_data['position']
                
                # Calculate size based on properties
                awareness = grain_data['awareness']
                activation = grain_data['activation']
                
                # Base size on absolute value of awareness and activation
                size = 20 + 100 * abs(awareness) * abs(activation) * 0.4  # Scale down for multi-plot
                
                # Get polarity for coloring
                polarity = grain_data['polarity']
                
                # Choose color based on polarity
                if polarity > 0.2:
                    color = 'blue'  # Structure/order
                elif polarity < -0.2:
                    color = 'red'   # Decay/chaos
                else:
                    color = 'green' # Neutral
                
                # Draw grain
                ax.scatter(theta, phi, s=size, color=color, alpha=0.7, 
                          edgecolors='black', linewidths=0.5)
            
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
            
            # Add title
            ax.set_title(self.visualizer.field_titles.get(field_name, field_name.capitalize()))
        
        # Hide any unused axes
        for i in range(len(field_names), len(axes)):
            axes[i].axis('off')
        
        # Add main title
        fig.suptitle(f'Field Comparison (t={self.visualizer.time:.2f})', fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
        
        return fig
        
    def create_3d_torus_visualization(self, field_name='awareness', figsize=(12, 10), alpha=0.8):
        """
        Create a 3D visualization of the field on a torus.
        
        Args:
            field_name: Name of the field to visualize
            figsize: Figure size (width, height) in inches
            alpha: Transparency of the torus surface
            
        Returns:
            Figure and axes objects
        """
        # Validate field name
        if field_name not in self.available_fields:
            print(f"Warning: Field '{field_name}' not found. Using 'awareness' instead.")
            field_name = 'awareness'
        
        # Get field data
        field_data = self.visualizer._field_data[field_name]
        
        # Create figure with 3D projection
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Define torus parameters
        major_radius = 3.0  # Distance from center of tube to center of torus
        minor_radius = 1.0  # Radius of the tube
        
        # Get appropriate colormap and norm
        cmap = self.visualizer._colormaps.get(field_name, cm.viridis)
        
        # Get norm for signed fields
        norm = None
        if field_name in ['polarity', 'curvature', 'activation', 'pressure'] and self.visualizer.signed_field_visualization:
            # For signed fields, use TwoSlopeNorm with 0 as center
            field_min = self.visualizer._field_stats[field_name]['min']
            field_max = self.visualizer._field_stats[field_name]['max']
            
            # Only use signed norm if field actually crosses zero
            if field_min < 0 and field_max > 0:
                from axiom8.visualizer.visualizer_core import create_safe_norm
                norm = create_safe_norm(
                    vmin=field_min,
                    vcenter=0.0,
                    vmax=field_max
                )
        
        # Create mesh grid for the torus
        theta = np.linspace(0, 2*np.pi, self.visualizer.resolution)
        phi = np.linspace(0, 2*np.pi, self.visualizer.resolution)
        theta, phi = np.meshgrid(theta, phi)
        
        # Convert to Cartesian coordinates
        x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
        y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
        z = minor_radius * np.sin(phi)
        
        # Plot the torus surface with field colors
        if norm:
            surface = ax.plot_surface(x, y, z, facecolors=cmap(norm(field_data)), 
                                    alpha=alpha, linewidth=0, antialiased=True)
        else:
            surface = ax.plot_surface(x, y, z, facecolors=cmap(field_data), 
                                    alpha=alpha, linewidth=0, antialiased=True)
        
        # Add grains as 3D scatter points
        for grain_id, grain_data in self.visualizer.grains.items():
            g_theta, g_phi = grain_data['position']
            
            # Convert to Cartesian coordinates
            g_x = (major_radius + minor_radius * np.cos(g_phi)) * np.cos(g_theta)
            g_y = (major_radius + minor_radius * np.cos(g_phi)) * np.sin(g_theta)
            g_z = minor_radius * np.sin(g_phi)
            
            # Calculate size based on properties
            awareness = grain_data['awareness']
            activation = grain_data['activation']
            
            # Base size on absolute value of awareness and activation
            size = 20 + 100 * abs(awareness) * abs(activation) * 0.2  # Scale down for 3D
            
            # Get polarity for coloring
            polarity = grain_data['polarity']
            
            # Choose color based on polarity
            if polarity > 0.2:
                color = 'blue'  # Structure/order
            elif polarity < -0.2:
                color = 'red'   # Decay/chaos
            else:
                color = 'green' # Neutral
            
            # Draw grain
            ax.scatter(g_x, g_y, g_z, s=size, color=color, alpha=0.8, 
                      edgecolors='black', linewidths=0.5)
            
            # Add grain ID label for larger grains
            if size > 20:
                ax.text(g_x, g_y, g_z, f"{grain_id}", fontsize=8, 
                       ha='center', va='center', color='white')
        
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(field_data)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label(self.visualizer.field_titles.get(field_name, field_name.capitalize()))
        
        # Configure axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set up realistic aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Add title
        ax.set_title(f'3D Torus Visualization: {self.visualizer.field_titles.get(field_name, field_name.capitalize())} (t={self.visualizer.time:.2f})')
        
        return fig, ax

# Example usage code:
"""
# Initialize the base visualizer with your manifold
base_visualizer = CollapseVisualizerBase(resolution=100)
base_visualizer.update_state(manifold)

# Create the map visualizer
map_visualizer = CollapseGeometryMapVisualizer(base_visualizer)

# Create unwrapped torus map
fig1, ax1 = map_visualizer.create_unwrapped_torus_map(field_name='awareness')
plt.savefig('unwrapped_torus_map.png')

# Create Cartesian projection map
fig2, ax2 = map_visualizer.create_cartesian_projection_map(field_name='awareness')
plt.savefig('cartesian_projection_map.png')

# Create dual visualization
fig3 = map_visualizer.create_dual_map_visualization(field_name='polarity')
plt.savefig('dual_map_visualization.png')

# Create field comparison
fig4 = map_visualizer.create_field_comparison_visualization()
plt.savefig('field_comparison.png')

# Create 3D torus visualization
fig5, ax5 = map_visualizer.create_3d_torus_visualization(field_name='awareness')
plt.savefig('3d_torus_visualization.png')
"""