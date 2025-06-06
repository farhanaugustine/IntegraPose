# File: app_config.py

# --- Constants for UI Styling ---
BASE_FONT_SIZE = 10
TITLE_FONT_SIZE = 12
LABEL_FONT_SIZE = 10
BUTTON_FONT_SIZE = 10
ENTRY_FONT_SIZE = 10
TOOLTIP_FONT_SIZE = 9  # This was also used in ToolTip class, good to have it centralized

# --- Constants for Matplotlib Plot Styling ---
# These will be used by plotting utilities and potentially by the main app
# when configuring Matplotlib's global rcParams.
PLOT_AXIS_LABEL_FONTSIZE = 9
PLOT_TITLE_FONTSIZE = 11
PLOT_TICK_LABEL_FONTSIZE = 8
PLOT_LEGEND_FONTSIZE = 'small' # Matplotlib interprets this relative to other font sizes

# Default Matplotlib rcParams update dictionary
# The main application can import this and call plt.rcParams.update(MATPLOTLIB_RCPARAMS)
MATPLOTLIB_RCPARAMS = {
    'font.size': PLOT_AXIS_LABEL_FONTSIZE, # General font size for plots
    'axes.titlesize': PLOT_TITLE_FONTSIZE,
    'axes.labelsize': PLOT_AXIS_LABEL_FONTSIZE,
    'xtick.labelsize': PLOT_TICK_LABEL_FONTSIZE,
    'ytick.labelsize': PLOT_TICK_LABEL_FONTSIZE,
    'legend.fontsize': PLOT_LEGEND_FONTSIZE,
    'figure.titlesize': PLOT_TITLE_FONTSIZE + 2 # For fig.suptitle
}

# --- Other Application-Wide Constants (Examples, add as needed) ---
# Example: Default FPS if not specified elsewhere or if video metadata is missing
DEFAULT_VIDEO_FPS = 30.0

# Example: Default visibility score if data is missing it and user assumes visible
DEFAULT_ASSUMED_VISIBILITY = 2.0

# Example: Small epsilon for division by zero avoidance in calculations
EPSILON = 1e-9

# Add any other constants that are used across multiple files or
# that you want to easily configure from one place.