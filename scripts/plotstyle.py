"""Shared matplotlib style for all figures (colors validated for colour-blind safety)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Categorical slots 1-3 of the reference palette, fixed per engine (never re-ordered).
# Aqua is below 3:1 contrast on the surface, so engine lines/bars are always direct-labelled.
ENGINE_COLOR = {'OPENAI': '#2a78d6', 'GEMINI': '#eb6834', 'PERPLEXITY': '#1baf7a'}
SINGLE = '#2a78d6'                      # one-series charts
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'


def apply():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
        'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.labelsize': 9, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': 2, 'legend.frameon': False, 'legend.fontsize': 8,
        'font.family': 'sans-serif', 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    })


apply()
