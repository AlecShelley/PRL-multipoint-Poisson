import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle

from numba import jit
from collections import defaultdict
from functools import lru_cache
import itertools
from itertools import combinations
import math
import hashlib

import scipy.special as sp
from scipy.spatial import ConvexHull, QhullError
from sklearn.decomposition import PCA

if __name__ == "__main__":
    import doctest
    doctest.testmod()