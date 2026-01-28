"""
Professional styling configurations for analytics graphs and reports.

Provides consistent, professional visual styling for all generated
graphs and reports following corporate presentation standards.
"""

from typing import Dict, Any
import matplotlib.pyplot as plt
import matplotlib as mpl

# Professional color palette (colorblind-friendly)
colors = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#ff7f0e',    # Orange
    'tertiary': '#2ca02c',     # Green
    'quaternary': '#d62728',   # Red
    'quinary': '#9467bd',      # Purple
    'senary': '#8c564b',       # Brown
    'septenary': '#e377c2',    # Pink
    'octonary': '#7f7f7f',     # Gray
    'nonary': '#bcbd22',       # Olive
    'denary': '#17becf',       # Cyan
    'background': '#ffffff',   # White
    'grid': '#e0e0e0',         # Light gray
    'text': '#333333',         # Dark gray
    'accent': '#0066cc',       # Accent blue
    'warning': '#ff6b6b',      # Warning red
    'success': '#51cf66',      # Success green
    'info': '#339af0'          # Info blue
}

# Professional font configurations
fonts = {
    'title': {
        'family': 'DejaVu Sans',
        'size': 16,
        'weight': 'bold'
    },
    'subtitle': {
        'family': 'DejaVu Sans',
        'size': 12,
        'weight': 'normal'
    },
    'axis_label': {
        'family': 'DejaVu Sans',
        'size': 10,
        'weight': 'normal'
    },
    'tick_label': {
        'family': 'DejaVu Sans',
        'size': 9,
        'weight': 'normal'
    },
    'legend': {
        'family': 'DejaVu Sans',
        'size': 9,
        'weight': 'normal'
    },
    'annotation': {
        'family': 'DejaVu Sans',
        'size': 8,
        'weight': 'normal',
        'style': 'italic'
    }
}

# Graph styling configurations
graph_styles = {
    'line_chart': {
        'linewidth': 2.5,
        'markersize': 6,
        'alpha': 0.8,
        'marker_style': 'o'
    },
    'bar_chart': {
        'bar_width': 0.8,
        'alpha': 0.8,
        'edge_color': 'white',
        'edge_width': 1
    },
    'pie_chart': {
        'explode_distance': 0.05,
        'shadow': True,
        'start_angle': 90,
        'text_props': {
            'fontsize': 9,
            'fontweight': 'bold'
        }
    },
    'scatter': {
        'markersize': 50,
        'alpha': 0.7,
        'edge_color': 'white',
        'edge_width': 1
    }
}

# Layout configurations
layouts = {
    'figure_size': {
        'small': (8, 6),
        'medium': (10, 7),
        'large': (12, 8),
        'wide': (14, 6),
        'tall': (8, 10)
    },
    'margins': {
        'tight': {'pad': 1.0},
        'standard': {'pad': 1.5},
        'spacious': {'pad': 2.0}
    },
    'grid': {
        'alpha': 0.3,
        'linestyle': '--',
        'linewidth': 0.5
    }
}


def apply_professional_style(style_name: str = "default"):
    """
    Apply professional styling to matplotlib.

    Args:
        style_name: Name of the style to apply
    """
    # Set professional font
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'normal',
        'legend.framealpha': 0.9,
        'legend.fancybox': True,
        'legend.shadow': True
    })

    # Set color palette
    plt.rcParams.update({
        'axes.prop_cycle': plt.cycler('color', [
            colors['primary'], colors['secondary'], colors['tertiary'],
            colors['quaternary'], colors['quinary'], colors['senary'],
            colors['septenary'], colors['octonary'], colors['nonary'],
            colors['denary']
        ])
    })

    # Set grid styling
    plt.rcParams.update({
        'axes.grid': True,
        'grid.alpha': layouts['grid']['alpha'],
        'grid.linestyle': layouts['grid']['linestyle'],
        'grid.linewidth': layouts['grid']['linewidth']
    })

    # Set figure styling
    plt.rcParams.update({
        'figure.facecolor': colors['background'],
        'axes.facecolor': colors['background'],
        'savefig.facecolor': colors['background'],
        'savefig.edgecolor': 'none',
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight'
    })


def get_color_palette(n_colors: int = None) -> list:
    """
    Get a professional color palette.

    Args:
        n_colors: Number of colors needed (default: all colors)

    Returns:
        List of color hex codes
    """
    color_values = [
        colors['primary'], colors['secondary'], colors['tertiary'],
        colors['quaternary'], colors['quinary'], colors['senary'],
        colors['septenary'], colors['octonary'], colors['nonary'],
        colors['denary']
    ]

    if n_colors is None:
        return color_values
    else:
        # Cycle through colors if more needed
        return (color_values * ((n_colors // len(color_values)) + 1))[:n_colors]


def get_figure_size(style: str = "medium") -> tuple:
    """
    Get figure size for different chart styles.

    Args:
        style: Size style ('small', 'medium', 'large', 'wide', 'tall')

    Returns:
        Tuple of (width, height) in inches
    """
    return layouts['figure_size'].get(style, layouts['figure_size']['medium'])


# Professional style template
professional_style = {
    'title_font': fonts['title'],
    'subtitle_font': fonts['subtitle'],
    'axis_font': fonts['axis_label'],
    'tick_font': fonts['tick_label'],
    'legend_font': fonts['legend'],
    'annotation_font': fonts['annotation'],
    'colors': colors,
    'layouts': layouts,
    'graph_styles': graph_styles
}