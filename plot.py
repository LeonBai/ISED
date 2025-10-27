import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_embedding(embeddings,
                   dim=2,
                   data=None,
                   idx_order=None,
                   r_cmap='cool',
                   l_cmap='viridis',
                   time_cmap='coolwarm',
                   s=10,
                   figsize=(6, 6),
                   dpi=150,
                   view_init_azim=45,
                   view_init_elev=45):
    """
    Generates a 2D or 3D scatter plot of embeddings with a white background and no axes.

    This function operates in multiple modes:
    - `dim`: Toggles between 2D and 3D visualization.
    - `data`: If provided, colors points based on 'left'/'right' groupings.
              If None, colors points based on their sequence (time).

    Args:
        embeddings (np.ndarray): Array of the embedded data points to plot.
        dim (int, optional): The dimension of the plot (2 or 3). Defaults to 2.
        data (dict, optional): If provided, a dictionary with a 'position' key
                               for grouping and coloring. Defaults to None.
        idx_order (tuple, optional): Indices for the axes. If None, defaults to
                                     (0, 1) for 2D and (0, 1, 2) for 3D.
        r_cmap (str, optional): Colormap for 'right' points. Defaults to 'cool'.
        l_cmap (str, optional): Colormap for 'left' points. Defaults to 'viridis'.
        time_cmap (str, optional): Colormap for time-based coloring. Defaults to 'plasma'.
        s (float, optional): Marker size. Defaults to 10.
        figsize (tuple, optional): Figure size. Defaults to (6, 6).
        dpi (int, optional): Figure resolution. Defaults to 150.
        view_init_azim (int, optional): Azimuthal viewing angle for 3D plots.
        view_init_elev (int, optional): Elevation viewing angle for 3D plots.

    Returns:
        tuple: A tuple containing the matplotlib figure and axes objects (fig, ax).
    """
    if dim not in [2, 3]:
        raise ValueError("The 'dim' parameter must be 2 or 3.")

    # Set default index order if not provided
    if idx_order is None:
        idx_order = tuple(range(dim))
    elif len(idx_order) != dim:
        raise ValueError(f"Length of 'idx_order' ({len(idx_order)}) must match 'dim' ({dim}).")

    # --- Figure and Axes Setup ---
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('white')

    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(azim=view_init_azim, elev=view_init_elev)
    else: # dim == 2
        ax = fig.add_subplot(111)

    ax.patch.set_facecolor('white')

    # --- Plotting Logic ---
    if data is not None:
        # Mode 1: Plot with Left/Right data
        num_points = min(len(data), len(embeddings))
        r_ind = data[:num_points, 1] == 1
        l_ind = data[:num_points, 2] == 1
        r_c = data[:num_points][r_ind, 0]
        l_c = data[:num_points][l_ind, 0]

        # Unpack indices
        indices = [embeddings[..., i] for i in idx_order]

        # Scatter plots
        ax.scatter(*[indices[i][r_ind] for i in range(dim)], c=r_c, cmap=r_cmap, s=s, alpha=0.8)
        ax.scatter(*[indices[i][l_ind] for i in range(dim)], c=l_c, cmap=l_cmap, s=s, alpha=0.8)
    else:
        # Mode 2: Plot as a single time series
        time_colors = np.arange(len(embeddings))
        indices = [embeddings[..., i] for i in idx_order]
        ax.scatter(*indices, c=time_colors, cmap=time_cmap, s=s, alpha=0.8)

    # Remove all axis elements
    ax.axis('off')

    return fig, ax

