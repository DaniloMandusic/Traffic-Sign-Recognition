import math

def get_subplot_dims(num_samples, max_cols=5, aspect_ratio=(4, 3)):
    """
    Calculate number of rows and columns for subplots
    and suggested figure size based on number of samples.

    Args:
        num_samples (int): Number of plots to display.
        max_cols (int): Maximum columns allowed.
        aspect_ratio (tuple): Width:Height ratio of each subplot.

    Returns:
        rows (int), cols (int), fig_size (tuple)
    """
    if num_samples <= 0:
        return 0, 0, (0, 0)

    cols = min(num_samples, max_cols)
    rows = math.ceil(num_samples / cols)

    # Width and height of figure
    width_per_ax, height_per_ax = aspect_ratio
    fig_width = cols * width_per_ax
    fig_height = rows * height_per_ax

    return rows, cols, (fig_width, fig_height)