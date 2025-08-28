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



####### helper functions and 2d color plots #######
def rate(d, r):
    """Calculates the arrival rate of a rigid-motion invariant Poisson hyperplane process in d dimensions
    such that the arrival rate of hyperplanes hitting a line segment is independant of dimension.
    
    d: int, the dimension of the space
    r: float, the radius of the enveloping ball of the Poisson hyperplane process
       
    returns: float, the rate of the Poisson hyperplane process
    >>> rate(1, 1)
    2
    >>> rate(2, 1)  # doctest: +ELLIPSIS
    3.14...
    >>> rate(3, 1)  # doctest: +ELLIPSIS
    4.0...
    """
    
    if d == 1: #the formula won't work for d=1
        return 2*r
    else: #rate is 2r/lambda_d from the paper
        return 2*np.sqrt(np.pi) * sp.gamma(d/2 + 1/2) / sp.gamma(d/2) * r
    

def sample_from_ball(d, n):
    """ Sample n points from interior of unit d-ball by uniformly sampling angle and then offset

    d: int, the dimension of the space
    n: int, the number of points to sample
    returns: np.array, shape (n, d), the sampled points

    """
    points = np.random.normal(size=(n, d))
    scales = np.random.uniform(size=(n, 1))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    unit_ball_points = points / norms * scales
    return unit_ball_points

@jit(nopython=True)
def hyperplane_partition(points, gridpoints):
    """Returns a unique hash for each grid point for unlimited number of hyperplanes
    
    points: np.array, shape (n,d), the points defining the hyperplanes
    gridpoints: np.array, shape (N,d), the points at which to evaluate the hyperplanes

    returns: np.array, the region hashes for the gridpoints
    """
    num_hyperplanes = points.shape[0]
    num_regions = gridpoints.shape[0]
    region_hashes = np.zeros((num_regions, (num_hyperplanes + 63) // 64), dtype=np.int64)
    #region hashes assigns each gridpoint to a region based on the hyperplanes
    #each set of 64 hyperplanes forms a bitwise hash map for each gridpoint
    for i, point in enumerate(points):
        gridpoints_cont = np.ascontiguousarray(gridpoints - point)
        point_cont = np.ascontiguousarray(point)
        signs = np.sign(np.dot(gridpoints_cont, point_cont)).astype(np.int8)
        bucket, offset = divmod(i, 64)
        region_hashes[:, bucket] += (signs > 0).astype(np.int64) << offset
    
    return region_hashes

def _hash_tuple_to_unit_interval(tup, salt=0):
    """Deterministic float in [0,1) from an integer tuple."""
    m = hashlib.blake2b(digest_size=8)
    m.update(np.array([salt], dtype=np.int64).tobytes())
    m.update(np.array(tup, dtype=np.int64).tobytes())
    val = int.from_bytes(m.digest(), 'little')
    return (val & ((1 << 53) - 1)) / float(1 << 53)  # map to [0,1)

def hyperplane_colorer_2D(points, gridpoints, colorcutoffs, salt=12345):
    """
    Returns color indices for a 2D ball partitioned by hyperplanes.
    Color is deterministic per-region using a hash of the region's bit-signature,
    so changing N (grid resolution) won't reshuffle colors.
    """
    region_hashes = hyperplane_partition(points, gridpoints)

    # Convert per-point multi-int signatures to tuples (hashable)
    region_hashes_tuples = [tuple(row) for row in region_hashes]

    # Build a stable mapping: region tuple -> color index
    color_lookup = {}
    def color_index_for_region(t):
        if t not in color_lookup:
            u = _hash_tuple_to_unit_interval(t, salt=salt)   # deterministic in [0,1)
            color_lookup[t] = np.digitize(u, colorcutoffs)   # 0..len(colorcutoffs)
        return color_lookup[t]

    regions = np.fromiter(
        (color_index_for_region(t) for t in region_hashes_tuples),
        dtype=np.int32,
        count=len(region_hashes_tuples)
    )
    return regions



frozen_lake_colors = ['#043765', "#edf8fe", 'green', 'purple', 'orange',\
                                              'yellow', 'pink', 'cyan', 'magenta', 'navy']
#meat_colors = ['#d22c2c', '#f4d5ce', 'green', 'purple', 'orange', 'yellow', \
# 'pink', 'cyan', 'magenta', 'navy']
def plot_hyperplanes_color_2D(r, N=100, colorcutoffs=np.array([0.5]), cmap_list = frozen_lake_colors,\
                                figsize = (6,6), preview_dpi=100):
    """
    Plots the Poisson hyperplane process in a ball in 2D, colored by region.
    
    r: float, The radius of the enveloping ball of the Poisson hyperplane process
    N: int, the number of points in the grid in each direction (default 100)
    colorcutoffs: np.array, the cutoffs for coloring the regions from a uniform [0,1] variable
    cmap_list: list, the colors corresponding to the cutoffs
    figsize: tuple, the size of the figure
    preview_dpi: int, the dpi of the displayed figure

    returns: figure, the figure object
    """
    if len(colorcutoffs) > len(cmap_list) - 1:
        raise ValueError(
            f"Too many colors: got {len(colorcutoffs)} cutoffs but only "
            f"{len(cmap_list) - 1} colors available.")

    cmap = mcolors.ListedColormap(cmap_list)

    d = 2
    n = np.random.poisson(rate(d, r))
    points = sample_from_ball(d, n) * r

    x = np.linspace(-r, r, N)
    y = np.linspace(-r, r, N)
    xx, yy = np.meshgrid(x, y)
    gridpoints = np.c_[xx.ravel(), yy.ravel()]

    # Use the updated safe colorer
    values = hyperplane_colorer_2D(points, gridpoints, colorcutoffs).reshape(N, N)

    fig, ax = plt.subplots(figsize = figsize, dpi=preview_dpi)
    #the first set of colors is for fig. 1.a., the second set is for fig. 1.b.
    bounds = np.linspace(-0.5, len(cmap.colors) - 0.5, len(cmap.colors) + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    cax = ax.imshow(values, extent=(-r, r, -r, r), origin='lower', cmap=cmap, norm=norm, interpolation='nearest', resample=False)

    # Create a circular clip path
    clip_circle = Circle((0, 0), r, transform=ax.transData)
    for artist in ax.get_children():
        artist.set_clip_path(clip_circle)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()

    return fig

if __name__ == "__main__":
    res = 300
    rad = 20
    cutoff = .65

    fig1 = plot_hyperplanes_color_2D(rad, res, colorcutoffs=np.array([cutoff]))

### The rest of the code is for calculating the multipoint functions ###
@lru_cache(maxsize=None)
def generate_all_connectivity_tuples(n):
    """Generates all possible connectivity tuples for n points, sorted by lexicographical order.

    n: int, the number of elements (points)

    returns: list, the list of all connectivity tuples for the set of n points, sorted lexicographically

    >>> generate_all_connectivity_tuples(3)
    [((0,), (1,), (2,)), ((0,), (1, 2)), ((0, 1), (2,)), ((0, 1, 2),), ((0, 2), (1,))]
    """

    def partitions(set_):
        """Generate all partitions of a set, ensuring sorted partitions"""
        if len(set_) == 1:
            yield [tuple(set_)]
            return
        first = set_[0]
        for smaller in partitions(set_[1:]):
            for n, subset in enumerate(smaller):
                yield smaller[:n] + [(first,) + subset] + smaller[n+1:]
            yield [(first,)] + smaller

    elements = tuple(range(n))
    
    # Generate all connectivity tuples
    all_connectivity_tuples = [
        tuple(sorted(partition, key=lambda x: (x[0], x))) for partition in partitions(elements)
    ]
    
    # Sort the outer list of tuples
    all_connectivity_tuples.sort(key=lambda partition: (sorted(partition), partition))
    
    return all_connectivity_tuples

def allowed_tuples_colors(tuples,colors, last_color_unknown = False):
    """Returns the allowed connectivity tuples for points with the given colors. Different colors can't be connected.

    tuples: list of tuples, the list of connectivity tuples to filter
    colors: np.array, the colors of the points
    last_color_unknown: bool, whether the last color is unknown and should be ignored (default False)

    returns: list, the list of allowed connectivity tuples

    >>> all_tuples = generate_all_connectivity_tuples(3)
    >>> colors1 = np.array([1, 1, 2])
    >>> result1 = allowed_tuples_colors(all_tuples, colors1)
    >>> result1.sort()
    >>> print(result1)
    [((0,), (1,), (2,)), ((0, 1), (2,))]

    >>> colors2 = np.array([1, 2, 1])
    >>> result2 = allowed_tuples_colors(all_tuples, colors2)
    >>> result2.sort()
    >>> print(result2)
    [((0,), (1,), (2,)), ((0, 2), (1,))]

    >>> colors3 = np.array([1, 1, 1])
    >>> result3 = allowed_tuples_colors(all_tuples, colors3)
    >>> result3.sort()
    >>> print(result3)
    [((0,), (1,), (2,)), ((0,), (1, 2)), ((0, 1), (2,)), ((0, 1, 2),), ((0, 2), (1,))]

    >>> colors4 = np.array([1, 2])
    >>> result4 = allowed_tuples_colors(all_tuples, colors4, last_color_unknown=True)
    >>> result4.sort()
    >>> print(result4)
    [((0,), (1,), (2,)), ((0,), (1, 2)), ((0, 2), (1,))]
    """
    if last_color_unknown:  # don't filter based on the last point's color
        final_point = len(colors) #not -1 since colors is for all the points which actually do have a color assigned
        allowed_tuples = [
            tup for tup in tuples
            if all(len(set(colors[i] for i in component if i != final_point)) == 1 
                for component in tup if len(component) > 1 or final_point not in component)
        ]
        return allowed_tuples
    
    # Filter and collect valid tuples using a list comprehension
    allowed_tuples = [
        tup for tup in tuples
        if all(len(set(colors[i] for i in component)) == 1 for component in tup)
    ]

    return allowed_tuples


@lru_cache(maxsize = None) #cache the results of this function to avoid recalculating. must converet argument to hashable type
def graph_cutter(num_points, cuts):
    """Returns the connectivity tuple for a series of graph cuts.
    
    num_points: int, the number of points
    cuts: tuple of tuples, each tuple contains indices of points which are cut from every point OUTSIDE the tuple

    returns: tuple of tuples, the connectivity tuple

    >>> graph_cutter(5, ((0, 1), (2, 3)))
    ((0, 1), (2, 3), (4,))

    >>> graph_cutter(5, ((0, 1), (1, 2)))
    ((0,), (1,), (2,), (3, 4))
    """
    
    vertices = list(range(num_points))
    remaining = set(vertices)  # Use a set to track remaining vertices
    connectivity_tuple = []

    for v in vertices:
        if v not in remaining:  # Skip if already processed
            continue
        cc = [v]
        remaining.remove(v)  # Mark v as processed

        for w in list(remaining):  # Iterate over remaining vertices
            if all((v in cut) == (w in cut) for cut in cuts):  # Check if v and w belong to same side of all cuts
                cc.append(w)
                remaining.remove(w)  # Mark w as processed

        connectivity_tuple.append(cc)  # Keep as list for now

    return tuple(map(tuple, connectivity_tuple))  # Convert only once at the end


def hitrate_1d(points):
    """Calculates the Poisson rate of hyperplanes hitting the convex hull of points in 1D.

    points: np.array, shape (n,), the points to hit

    returns: float, the rate of hyperplanes hitting the convex hull

    >>> hitrate_1d(np.array([-2, -1, 0, 1, 2]))
    4
    """

    return max(points)-min(points) #hitrate is the length of segment


def hitrate_2d(points):
    """Calculates the Poisson rate of hyperplanes hitting the convex hull of points in 2D.

    points: np.array, shape (n,2), the points to hit

    returns: float, the rate of hyperplanes hitting the convex hull

    >>> import numpy as np
    >>> hitrate_2d(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))
    2.0

    >>> hitrate_2d(np.array([[0, 0], [1, 0]]))
    1.0

    >>> hitrate_2d(np.array([[0,0],[0,0],[0,0],[0,0]]))
    0.0
    """
    if len(points) == 2:
        return np.linalg.norm(points[1] - points[0])
    elif len(points) == 3:
        return np.sum([np.linalg.norm(points[i] - points[(i+1)%3]) for i in range(3)]) / 2
    try:
        hull = ConvexHull(points)
        perimeter = hull.area
        return perimeter / 2
    except QhullError:
        # Degenerate in 2D → project to 1D and return segment length
        centered = points - np.mean(points, axis=0)
        if np.allclose(centered, 0):
            return 0.0  # all points coincide, no hitrate
        direction = PCA(n_components=1).fit(centered).components_[0]
        projected = np.dot(centered, direction)
        return np.ptp(projected)


def dihedral_angle(norm1, norm2):
    """Calculates the dihedral angle between two normal vectors.

    Parameters:
    norm1 (np.array): Normal vector of the first face.
    norm2 (np.array): Normal vector of the second face.

    Returns:
    float: The dihedral angle in radians.

    >>> dihedral_angle(np.array([0, -1, 0]), np.array([1, 1, 1]))
    2.1862760354652844
    """

    cos_theta = np.dot(norm1, norm2) / (np.linalg.norm(norm1) * np.linalg.norm(norm2))
    # Ensure the cosine is within valid range due to potential numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return angle

def ensure_outward_facing(norm, point_on_face, centroid):
    """Ensure that the normal vector is outward-facing."""
    to_centroid = centroid - point_on_face
    if np.dot(norm, to_centroid) > 0:
        return -norm  # Flip the normal to point outward
    return norm

def hitrate_3d(points):
    """
    Calculates the Poisson rate of hyperplanes hitting the convex hull of points in 3D.

    Parameters:
    points (np.array): 3D array of points to hit, shape (n, 3).

    Returns:
    float: The rate of hyperplanes hitting the convex hull.

    Examples:
    >>> hitrate_3d(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.1, 0.1, 0.1]]))
    2.2262549897645005

    >>> hitrate_3d(np.array([[0, 0, 0], [1, 0, 0]]))
    1.0

    >>> hitrate_3d(np.array([[0, 0, 0], [1, 0, 0], [0,0,0]]))
    1.0

    >>> three_d_degenerate = hitrate_3d(np.array([[0, 0,5], [1, 0,5], [0, 1,5], [1, 1,5]]))
    >>> two_d = hitrate_2d(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))
    >>> three_d_degenerate == two_d
    True
    """

    if len(points) == 2:
        return np.linalg.norm(points[1] - points[0])
    elif len(points) == 3:
        return np.sum([np.linalg.norm(points[i] - points[(i+1)%3]) for i in range(3)]) / 2
    try:
        hull = ConvexHull(points)
    except QhullError:
        # Points are degenerate: project to 2D plane and try the 2D function
        centered = points - np.mean(points, axis=0)
        if np.allclose(centered, 0):
            return 0.0
        projected = PCA(n_components=2).fit_transform(centered)
        return hitrate_2d(projected)

    edge_contributions = []
    centroid = np.mean(points, axis=0)

    for simplex in hull.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = tuple(sorted((simplex[i], simplex[j])))
                if edge not in [e[0] for e in edge_contributions]:
                    adjacent_faces = [face for face in hull.simplices if edge[0] in face and edge[1] in face]
                    if len(adjacent_faces) < 2:
                        continue  # skip boundary edges
                    face1_edge1 = points[adjacent_faces[0][1]] - points[adjacent_faces[0][0]]
                    face1_edge2 = points[adjacent_faces[0][2]] - points[adjacent_faces[0][0]]
                    face2_edge1 = points[adjacent_faces[1][1]] - points[adjacent_faces[1][0]]
                    face2_edge2 = points[adjacent_faces[1][2]] - points[adjacent_faces[1][0]]

                    norm1 = ensure_outward_facing(np.cross(face1_edge1, face1_edge2), points[adjacent_faces[0][0]], centroid)
                    norm2 = ensure_outward_facing(np.cross(face2_edge1, face2_edge2), points[adjacent_faces[1][0]], centroid)

                    edge_length = np.linalg.norm(points[edge[1]] - points[edge[0]])
                    angle = dihedral_angle(norm1, norm2)
                    edge_contributions.append((edge, edge_length * angle))

    return sum(contribution for edge, contribution in edge_contributions) / (2 * np.pi)

def slash_rates_1d(points):
    """Returns the rates of each single hyperplane partition of the points in 1D.

    points: sorted np.array, shape (n,), the points to partition

    returns: dict, the rates of each partition

    >>> points = np.array([0, 1, 2, 5])
    >>> slash_rates_1d(points)
    {(0,): 1, (0, 1): 1, (0, 1, 2): 3}
    """

    if not np.array_equal(points, np.sort(points)):
        raise ValueError("Points must be sorted")

    rates = {}
    #for each segment between points, the rate is the length of the segment
    for i in range(len(points)-1):
        connected_component = (tuple(range(0,i+1),))
        rates[connected_component] = points[i+1] - points[i]
    
    return rates

def slash_rates(points):
    """Returns the rates of each single hyperplane partition of the points.

    points: np.array, shape (n,d), the points to partition

    returns: dict, the rates of each partition

    >>> points2d = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    >>> result = slash_rates(points2d)
    >>> result_rounded = {k: round(v, 6) for k, v in result.items()}
    >>> print(len(result_rounded))
    7
    >>> print(result_rounded[(0,)])
    0.292893
    >>> print(result_rounded[(1,)])
    0.292893
    >>> print(result_rounded[(0, 2)])
    0.414214

    >>> points2d = np.array([[0, 0], [1, 0], [0, 1]])
    >>> result = slash_rates(points2d)
    >>> print(len(result))
    3

    >>> points2d = np.array([[0, 0], [1, 0]])
    >>> result = slash_rates(points2d)
    >>> print(len(result))
    1

    >>> points2d = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [.1,.1]])
    >>> result2d = slash_rates(points2d)
    >>> print(len(result2d))
    15
    >>> points3d = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [.1,.1, 0]])
    >>> result3d = slash_rates(points3d)
    >>> print(all([np.allclose(result2d[k],result3d[k]) for k in result2d]))
    True
    """

    dimension = points.shape[1] #this block makes the function general for any dimension
    if dimension == 1:
        return slash_rates_1d(points)
    elif dimension == 2:
        hitrate = hitrate_2d
    elif dimension == 3:
        hitrate = hitrate_3d

    n = len(points)

    #loop over all possible partitions of the points into two disjoint sets
    #start with just one point in the first set and the rest in the second set

    if n == 2:
        projection = (points[1]-points[0])/np.linalg.norm(points[1] - points[0])
        sorted_points = np.dot(points, projection)
        return slash_rates_1d(sorted_points)


    rates = defaultdict(int)
    whole_hitrate = hitrate(points)

    for i in range(n): #single point partitions base case
        rest_hitrate = hitrate(np.delete(points, i, axis=0))
        rates[(i,)] = whole_hitrate - rest_hitrate

    if n == 3:
        return rates
    
    def subset_rates(subset):
        m = len(subset)
        if m == 1:
            return rates[subset] #already calculated in base case
        
        rest_points = np.delete(points, list(subset), axis=0)
        rest_hitrate = hitrate(rest_points)
        
        rate = whole_hitrate - rest_hitrate #next we need to subtract the subset rates
        for sub_size in range(1,m):
            for sub_subset in combinations(subset, sub_size):
                rate -= rates[sub_subset] #subtracting off rate of smaller subsets
        
        rates[subset] = rate #can modify mutable dictionary in function scope
        return rate
    
    #calculate the rates of all other partitions up to the middle one which is an edge case
    for size in range(2,n//2):
        for subset in combinations(range(n), size):
            subset_rates(subset)
    
    if n % 2 == 0: #have to be careful with middle partitions so as to not be redundant
        for subset in combinations(range(n), n//2):
            complement = tuple(sorted(set(range(n)) - set(subset)))
            #only keep the subset if its first element is less than the complement's first element
            if subset < complement:
                subset_rates(subset)
    else: #n is odd
        for subset in combinations(range(n), n//2):
            subset_rates(subset)
    return rates


def color_from_partitions(partitions, colors, num_points, color_dist):
    """Returns the color probability distribution for the final point given the partitions 
    and colors of the other points.
    
    partitions: tuple of tuples, the single cut connectivity tuples of the points
    colors: tuple, length (num_points-1), the colors of the points
    num_points: int, the number of points, including the final point whose color is unknown
    color_dist: tuple, the probabilities of each color

    returns: np.array, the calculated color probabilities for the final point

    >>> partitions = ((0,), (1,), (0, 1)) #the complete partition is ((0,), (1,), (2, 3)), so must be color 1
    >>> colors = (1, 2, 1)
    >>> num_points = 4
    >>> color_dist = (.2,.2,.2,.2,.2)
    >>> color_from_partitions(partitions, colors, num_points, color_dist)
    array([0., 1., 0., 0., 0.])

    >>> partitions = ((0,), (0,3)) #the complete partition is ((0,), (1,2), (3)), so uniform distribution
    >>> colors = (1, 2, 2)
    >>> num_points = 4
    >>> color_dist = (.4,.6)
    >>> got = color_from_partitions(partitions, colors, num_points, color_dist)
    >>> expected = np.array([0.4, 0.6])
    >>> np.allclose(got, expected)
    True

    >>> partitions = ()
    >>> colors = (1, 1, 1)
    >>> got = color_from_partitions(partitions, colors, num_points, color_dist)
    >>> expected = np.array([0., 1.])
    >>> np.allclose(got, expected)
    True
    """
    num_colors = len(color_dist)
    color_probs = np.zeros(num_colors)

    #find which point, if any, points[-1] is connected to.
    #points[-1] is connected to points[i] if each partition has ((num_points-1) in partition) == (i in partition)
    connected_point = None
    for i in range(num_points-1):
        if all(((num_points-1) in partition) == (i in partition) for partition in partitions):
            connected_point = i
            break
    if not connected_point == None:
        color_probs[colors[connected_point]] = 1
    else: #if points[-1] is isolated, the color is uniformly distributed
        color_probs = np.array(color_dist)
    return color_probs

def color_distribution(points, colors, color_dist):
    """Returns the color probability distribution for points[-1].
    
    points: np.array, shape (n,d). points[:-1] are the colored points, points[-1] is the point whose color is unknown
    colors: tuple of ints, shape (n-1). Colors of points[:-1].
    color_dist: tuple, the probabilities of each color

    returns: np.array, the calculated color distribution for the final point
    >>> points = np.array([[0, 1], [1, 0], [0, 0]])
    >>> colors = (0, 1)
    >>> color_dist = (.5,.5)
    >>> got = color_distribution(points, colors, color_dist)
    >>> expected = np.array([0.5, 0.5])
    >>> print(np.allclose(got, expected))
    True

    >>> points3d = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    >>> got3d = color_distribution(points3d, colors, color_dist)
    >>> print(np.allclose(got, got3d))
    True
    """
    n = len(points)
    rates = slash_rates(points)
    partitions = list(rates.keys())
    all_connectivity = generate_all_connectivity_tuples(n)
    allowed_partitions = allowed_tuples_colors(all_connectivity, colors, last_color_unknown=True)
    #allowed_partitions = allowed_tuples_geometric(allowed_partitions, points)
    #^^ the above line might be redundant, tests indicate it give the same result with minimal time difference
    num_p = len(partitions)
    ret = np.zeros(len(color_dist))

    # Precompute exp(-rates[partition]) for each partition
    exp_rates = {partition: np.exp(-rates[partition]) for partition in partitions}

    #calculate the probability of each member of the superset of all partitions
    #from there, use color_from_partitions to calculate the color distribution, then add it to the final distribution
    power_set = itertools.chain.from_iterable(itertools.combinations(partitions, r) for r in range(num_p+1))
    probcount = 0
    for subset in power_set: #remember, to convert a rate to a probability, do P(not happen) = e^(-rate)
        partition = graph_cutter(n, subset)
        if not partition in allowed_partitions:
            continue
        # Memoized product calculations
        subset_prob = math.prod([1 - exp_rates[partition] for partition in subset])
        complement_prob = math.prod([exp_rates[partition] for partition in partitions if partition not in subset])
        subset_prob *= complement_prob
        
        subset_colors = color_from_partitions(subset, colors, n, color_dist)
        ret += subset_prob * subset_colors    
        probcount += subset_prob
    ret = ret / probcount  # normalize
    ret = np.clip(ret, 0, None)  # remove any tiny negatives
    ret /= np.sum(ret)  # re-normalize after clipping
    return ret

def color_sample(points, colors, color_dist):
    """Samples the color of points[-1].
    
    points: np.array, shape (n,d). points[:-1] are the colored points, points[-1] is the point whose color is unknown
    colors: tuple of ints, shape (n-1). Colors of points[:-1].
    color_dist: tuple, the probabilities of each color

    returns: np.array, the calculated color distribution for the final point
    """
    n = len(points)
    rates = slash_rates(points)
    partitions = list(rates.keys())
    all_connectivity = generate_all_connectivity_tuples(n)
    allowed_partitions = allowed_tuples_colors(all_connectivity, colors, last_color_unknown=True)
    #allowed_partitions = allowed_tuples_geometric(allowed_partitions, points)
    #^^ the above line might be redundant, tests indicate it give the same result with minimal time difference
    num_p = len(partitions)
    ret = np.zeros(len(color_dist))

    # Precompute exp(-rates[partition]) for each partition
    exp_rates = {partition: np.exp(-rates[partition]) for partition in partitions}

    #calculate the probability of each member of the superset of all partitions
    #from there, use color_from_partitions to calculate the color distribution, then add it to the final distribution
    power_set = itertools.chain.from_iterable(itertools.combinations(partitions, r) for r in range(num_p+1))
    probcount = 0
    for subset in power_set: #remember, to convert a rate to a probability, do P(not happen) = e^(-rate)
        partition = graph_cutter(n, subset)
        if not partition in allowed_partitions:
            continue
        # Memoized product calculations
        subset_prob = math.prod([1 - exp_rates[partition] for partition in subset])
        complement_prob = math.prod([exp_rates[partition] for partition in partitions if partition not in subset])
        subset_prob *= complement_prob
        
        subset_colors = color_from_partitions(subset, colors, n, color_dist)
        ret += subset_prob * subset_colors    
        probcount += subset_prob
    ret = ret / probcount  # normalize
    ret = np.clip(ret, 0, None)  # remove any tiny negatives
    ret /= np.sum(ret)  # re-normalize after clipping
    return ret

### The following hepler functions are for the Monte Carlo plots

def monte_carlo_hyperplane_partitions(d, r, gridpoints, num_samples):
    """Returns Monte-Carlo distribution of connectivity tuples of hyperplane partitions of gridpoints.
    d: int, the dimension of the space
    r: float, the radius of the ball
    gridpoints: np.array, shape (n,d), the points to partition
    num_samples: int, the number of samples to take
    """

    connectivity_counts = defaultdict(int)  # Store the counts of each connectivity
    num_points = len(gridpoints)
    
    for _ in range(num_samples):
        # Sample hyperplanes
        n_hyperplanes = np.random.poisson(rate(d, r))
        hyperplanes = sample_from_ball(d, n_hyperplanes) * r
        
        # Determine regions using the safe partition function
        region_hashes = hyperplane_partition(hyperplanes, gridpoints)
        
        # Convert region hashes to tuples for hashability
        region_hashes_tuples = [tuple(row) for row in region_hashes]
        
        # Create connectivity tuple with sorted components
        connectivity_components = defaultdict(list)
        for i, region_hash in enumerate(region_hashes_tuples):
            connectivity_components[region_hash].append(i)
        
        # Convert to connectivity tuple (sorted by components)
        connectivity_tuple = tuple(tuple(sorted(component)) for component in connectivity_components.values())
        
        # Increment the count of this connectivity tuple
        connectivity_counts[connectivity_tuple] += 1
    
    # Convert counts to probabilities
    connectivity_distribution = {k: v / num_samples for k, v in connectivity_counts.items()}
    return connectivity_distribution

def plot_convergence_all_partitions_mc(d, gridpoints, samples_array):
    """Plots Monte Carlo convergence of probabilities of each connectivity graph
    d: dimension of the hyperplane
    gridpoints: list of points to generate the hyperplanes
    samples_array: array of number of samples to run the simulation
    analytic_probs: dictionary of analytic probabilities for each graph (to compare to Monte Carlo results)
    """
    r = max([np.linalg.norm(p) for p in gridpoints])  # Maximum distance from origin
    all_partition_probs = defaultdict(lambda: np.zeros(len(samples_array)))  # Empty array for nonexistent keys
    possible_partitions = generate_all_connectivity_tuples(len(gridpoints))

    cumulative_counts = defaultdict(int)

    for i, num_samples in enumerate(np.diff(np.insert(samples_array, 0, 0))):
        partition_probs = monte_carlo_hyperplane_partitions(d, r, gridpoints, num_samples)

        # Accumulate counts for each partition
        for partition, probability in partition_probs.items():
            cumulative_counts[partition] += probability * num_samples

        # Calculate probabilities at this step
        total_count = sum(cumulative_counts.values())
        for partition in possible_partitions:
            all_partition_probs[partition][i] = cumulative_counts[partition] / total_count

    return all_partition_probs

def monte_carlo_convergence_with_error_bars(d, gridpoints, samples_array, num_runs, analytic_probs=None):
    """Wrapper function that runs the convergence multiple times and plots the average with error bars."""
    # Initialize dictionaries to store sums and squared sums for averaging and std deviation
    all_partition_probs_sum = defaultdict(lambda: np.zeros(len(samples_array)))
    all_partition_probs_sq_sum = defaultdict(lambda: np.zeros(len(samples_array)))
    possible_partitions = generate_all_connectivity_tuples(len(gridpoints))

    # Run the Monte Carlo convergence multiple times
    for run in range(num_runs):
        all_partition_probs = plot_convergence_all_partitions_mc(d, gridpoints, samples_array)

        # Accumulate sum and squared sum for each partition
        for partition in possible_partitions:
            all_partition_probs_sum[partition] += all_partition_probs[partition]
            all_partition_probs_sq_sum[partition] += all_partition_probs[partition] ** 2

    # Calculate the mean and standard deviation for each partition
    all_partition_probs_mean = defaultdict(lambda: np.zeros(len(samples_array)))
    all_partition_probs_std = defaultdict(lambda: np.zeros(len(samples_array)))

    for partition in possible_partitions:
        mean = all_partition_probs_sum[partition] / num_runs
        variance = (all_partition_probs_sq_sum[partition] / num_runs) - (mean ** 2)
        stddev = np.sqrt(variance)

        all_partition_probs_mean[partition] = mean
        all_partition_probs_std[partition] = stddev

    # Plot the mean with error bars
    fig, ax = plt.subplots(figsize=(12, 8))
    for partition, mean_probs in all_partition_probs_mean.items():
        std_probs = all_partition_probs_std[partition]
        ax.errorbar(samples_array, mean_probs, yerr=std_probs, marker='o', label=f"H = {partition}", markersize=2, capsize=3)

    # Plot analytic probabilities as horizontal lines
    if analytic_probs:
        label_added = False
        for partition, analytic_prob in analytic_probs.items():
            if not label_added:
                ax.axhline(y=analytic_prob, linestyle='--', color='grey', label='Analytic Solution')
                label_added = True  # Set the flag to True after adding the label
            else:
                ax.axhline(y=analytic_prob, linestyle='--', color='grey')

    ax.set_xlabel('Number of Samples', fontsize=20)
    ax.set_ylabel('Probability of Graph H', fontsize=20)
    ax.set_title(f'Connectivity Graphs of Points {list(map(list, gridpoints))}', fontsize=26)
    ax.legend(loc='upper left', borderaxespad=0., fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=12)  # Minor ticks
    ax.set_xscale('log')

    # Add inset for log error
    if analytic_probs:
        inset_ax = fig.add_axes([0.51, 0.55, 0.25, 0.25], facecolor='white')
        inset_ax.patch.set_alpha(.8)  # Adjust transparency of the inset background
        for partition, mean_probs in all_partition_probs_mean.items():
            if partition in analytic_probs:
                analytic_prob = analytic_probs[partition]
                log_error = np.log10(np.abs(mean_probs - analytic_prob))
                inset_ax.plot(samples_array, log_error, label=f"H = {partition}")
        
        inset_ax.set_xscale('log')
        inset_ax.set_title(r'$\log_{10}$ Error', fontsize=20)
        inset_ax.tick_params(axis='both', which='major', labelsize=14)
        for spine in inset_ax.spines.values():
            spine.set_edgecolor('black')  # Ensure spine color stands out
            spine.set_linewidth(1.2)     # Make spines slightly thicker
            spine.set_zorder(5)          # Draw spines above background
        inset_ax.text(0.5, 0.05, 'Samples', fontsize=20, transform=inset_ax.transAxes,
                                ha='center', va='bottom', color='black', bbox=dict(facecolor='white', \
                                                                                alpha=0.8, edgecolor='none'))

    return fig, ax, all_partition_probs_mean, all_partition_probs_std

def hyperplane_colorer_3D(points, gridpoints, colorcutoffs):
    region_hashes = hyperplane_partition(points, gridpoints)
    
    # Use tuple-based representation instead of strings
    region_hashes_tuples = [tuple(row) for row in region_hashes]
    unique_dict = {}
    inverse_indices = []
    counter = 0
    
    # Manual uniqueness detection
    for row in region_hashes_tuples:
        if row not in unique_dict:
            unique_dict[row] = counter
            counter += 1
        inverse_indices.append(unique_dict[row])
    
    unique_regions = list(unique_dict.keys())
    num_unique_regions = len(unique_regions)

    # Assign random colors
    colors = np.random.uniform(size=num_unique_regions)
    color_indices = np.digitize(colors, colorcutoffs)

    # Map color indices to gridpoints
    regions = np.array([color_indices[inverse_indices[i]] for i in range(len(gridpoints))], dtype=np.int32)
    return regions

def colors_mc(d, gridpoints, color_dist, colors, samples_array):
    """Returns the CPF of gridpoints[-1] having the color corresponding to colorcutoffs[-1],
         given the colors of the other points
        d: dimension of the hyperplane
        r: radius to generate the hyperplanes
        gridpoints: numpy array of points to make the distribution of colors for
        color_dist: tuple, probability distribution of colors
        colors: tuple of colors for points[:-1]
        samples_array: array of number of samples to run the simulation, make with np.logspace!

        returns: list of dictionaries of color probabilities for each color in colors, for each sample size"""
    
    colorcutoffs = np.cumsum(color_dist)[:-1] #cumulative sum of color distribution to get cutoff values for each color
    r = max(np.linalg.norm(p) for p in gridpoints)*1.25 #maximum distance from origin
    if d == 2:
        hyperplane_colorer = hyperplane_colorer_2D
    elif d == 3:
        hyperplane_colorer = hyperplane_colorer_3D

            
    conditional_count = 0 #this counts the number of times colors[::-1] is the colors of gridpoints[::-1]

    color_counts = np.zeros(len(color_dist)) #array to store the number of times each color is the color of gridpoints[-1]
    color_probs = np.zeros((len(color_dist), len(samples_array))) #array to store the probability of each color for each sample size

    for i, num_samples in enumerate(np.diff(np.insert(samples_array, 0, 0))):
        for _ in range(num_samples):
            n = np.random.poisson(rate(d,r)) #number of hyperplanes
            points = sample_from_ball(d, n)*r #sample points defining hyperplanes

            color_values = hyperplane_colorer(points, gridpoints, colorcutoffs)
            if np.all(color_values[:-1] == colors):
                conditional_count += 1
                color_counts[color_values[-1]] += 1 #one new recorded instance of the color of gridpoints[-1]
        color_probs[:,i] = color_counts/conditional_count if conditional_count > 0 else 0

    return color_probs

def plot_mc_colors_with_errorbars(d, gridpoints, color_dist, colors,\
                                  samples_array, num_runs, analytic_probs=False):
    """Wrapper function that runs the convergence multiple times and plots the average with error bars.
    d: dimension of the hyperplane
    gridpoints: numpy array of points, all but the last are conditioned on with colors from "colors"
    color_dist: tuple, the probability distribution of colors
    colors: tuple of colors for points[:-1]
    samples_array: array of number of samples to run the simulation
    num_runs: number of epochs, used to calculate the mean and standard deviation"""

    # Run the Monte Carlo convergence multiple times
    color_probs_sum = np.zeros((len(color_dist), len(samples_array)))
    color_probs_sq_sum = np.zeros((len(color_dist), len(samples_array)))
    for _ in range(num_runs):
        color_probs = colors_mc(d, gridpoints, color_dist, colors, samples_array)
        color_probs_sum += color_probs
        color_probs_sq_sum += color_probs**2

    # Calculate the mean and standard deviation for each color
    color_probs_mean = color_probs_sum / num_runs
    variance = (color_probs_sq_sum / num_runs) - (color_probs_mean ** 2)
    stdev = np.sqrt(variance)

    # Calculate analytic probabilities if requested
    analytic_color_dist = None
    if analytic_probs:
        analytic_color_dist = color_distribution(gridpoints, colors, color_dist)

    # Plot the mean with error bars
    fig, ax = plt.subplots(figsize=(12, 8))
    for color in range(len(color_dist)):
        ax.errorbar(samples_array, color_probs_mean[color], yerr=stdev[color],
                     marker='o', label=f"Color {color}", markersize=2, capsize=3)

    if analytic_probs:
        label_added = False
        for color in range(len(color_dist)):
            if not label_added:
                ax.axhline(y=analytic_color_dist[color], linestyle='--', color='grey', label='Analytic Solution')
                label_added = True  # Add the label only once
            else:
                ax.axhline(y=analytic_color_dist[color], linestyle='--', color='grey')

    ax.set_xlabel('Number of Samples', fontsize=20)
    ax.set_ylabel('Probability of Color', fontsize=20)
    ax.set_title(f'Color of {list(map(list,gridpoints))[-1]} Given {list(map(list,gridpoints))[:-1]}', fontsize=26)
    ax.legend(loc='upper left', borderaxespad=0., fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)  # Major ticks
    ax.tick_params(axis='both', which='minor', labelsize=12)  # Minor ticks
    ax.set_xscale('log')

    # Add inset for log error
    if analytic_probs:
        inset_ax = fig.add_axes([0.5, 0.4, 0.25, 0.25], facecolor="white", alpha = .3)  # [left, bottom, width, height]
        inset_ax.patch.set_alpha(.4)  # Adjust transparency of the inset background
        for color in range(len(color_dist)):
            log_error = np.log10(np.abs(analytic_color_dist[color] - color_probs_mean[color]))
            inset_ax.plot(samples_array, log_error, label=f"Color {color}")
        
        inset_ax.set_xscale('log')
        inset_ax.set_ylabel(r'$\log_{10}$ Error', fontsize=20)
        inset_ax.tick_params(axis='both', which='major', labelsize=14)
        inset_ax.text(0.5, 0.05, 'Samples', fontsize=20, transform=inset_ax.transAxes,
                        ha='center', va='bottom', color='black', bbox=dict(facecolor='white', \
                                                                           alpha=0.8, edgecolor='none'))

    return fig, ax, color_probs_mean, stdev

if __name__ == "__main__":
    import doctest
    doctest.testmod()