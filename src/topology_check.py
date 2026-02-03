
import numpy as np
from scipy.ndimage import label

def get_cluster_sizes(mask):
    """Counts distinct connected 'islands' using 4-connectivity."""
    structure = np.array([[0,1,0], [1,1,1], [0,1,0]])
    labeled_array, num_features = label(mask, structure=structure)
    sizes = np.bincount(labeled_array.ravel())
    return sizes[1:] # Ignore background (0s)
