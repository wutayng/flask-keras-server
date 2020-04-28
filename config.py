import matplotlib.pyplot as plt

IMG_RESOLUTION = 256

# Heatmap Plot Colormaps
CMAP = plt.get_cmap('viridis')
CMAP.set_under('k', alpha=0)
VMIN = 0.01 # Colormap Lower Cutoff for Displaying Heamaps