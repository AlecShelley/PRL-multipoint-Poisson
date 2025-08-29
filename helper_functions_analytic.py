"""
Analytic utilities for Poisson hyperplane model.

This module provides functions for calculating conditional probability functions
for color in the Poisson hyperplane model.
It includes utilities for:
- Calculating the dimension-dependent rate which defines the Poisson model
- Connectivity tuple generation and filtering
- Hit rate calculations for convex hulls
- Graph cutting and connectivity analysis
- Calculating conditional probability functions for color in the Poisson hyperplane model
"""
import itertools
import math
from collections import defaultdict
from functools import lru_cache

import numpy as np
import scipy.special as sp
from scipy.spatial import ConvexHull, QhullError # pylint: disable=no-name-in-module
from sklearn.decomposition import PCA


def rate(dimension, radius):
    """Calculate the arrival rate of a rigid-motion invariant Poisson hyperplane process.
    
    Parameters
    ----------
    dimension : int
        The dimension of the space
    radius : float
        The radius of the enveloping ball of the Poisson hyperplane process
        
    Returns
    -------
    float
        The rate of the Poisson hyperplane process
        
    Examples
    --------
    >>> rate(1, 1)
    2
    >>> rate(2, 1)  # doctest: +ELLIPSIS
    3.14...
    >>> rate(3, 1)  # doctest: +ELLIPSIS
    4.0...
    """
    if dimension == 1:  # the formula won't work for d=1
        return 2 * radius
    # rate is 2r/lambda_d from the paper
    return 2 * np.sqrt(np.pi) * sp.gamma(dimension/2 + 1/2) / sp.gamma(dimension/2) * radius


### The rest of the code is for calculating the multipoint functions ###
@lru_cache(maxsize=None)
def generate_all_connectivity_tuples(num_points):
    """Generate all possible connectivity tuples for n points.
    
    Parameters
    ----------
    num_points : int
        The number of elements (points)
        
    Returns
    -------
    list
        The list of all connectivity tuples for the set of n points, 
        sorted lexicographically
        
    Examples
    --------
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
            for idx, subset in enumerate(smaller):
                yield smaller[:idx] + [(first,) + subset] + smaller[idx+1:]
            yield [(first,)] + smaller

    elements = tuple(range(num_points))

    # Generate all connectivity tuples
    all_connectivity_tuples = [
        tuple(sorted(partition, key=lambda x: (x[0], x)))
        for partition in partitions(elements)
    ]

    # Sort the outer list of tuples
    all_connectivity_tuples.sort(key=lambda partition: (sorted(partition), partition))

    return all_connectivity_tuples


def allowed_tuples_colors(tuples, colors, last_color_unknown=False):
    """Return the allowed connectivity tuples for points with the given colors.
    
    Different colors can't be connected.
    
    Parameters
    ----------
    tuples : list of tuples
        The list of connectivity tuples to filter
    colors : np.array
        The colors of the points
    last_color_unknown : bool, optional
        Whether the last color is unknown and should be ignored (default False)
        
    Returns
    -------
    list
        The list of allowed connectivity tuples
        
    Examples
    --------
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
        final_point = len(colors)  # not -1 since colors is for all the points which
                                   # actually do have a color assigned
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


@lru_cache(maxsize=None)  # cache the results of this function to avoid recalculating.
                          # must convert argument to hashable type
def graph_cutter(num_points, cuts):
    """Return the connectivity tuple for a series of graph cuts.
    
    Parameters
    ----------
    num_points : int
        The number of points
    cuts : tuple of tuples
        Each tuple contains indices of points which are cut from 
        every point OUTSIDE the tuple
        
    Returns
    -------
    tuple of tuples
        The connectivity tuple
        
    Examples
    --------
    >>> graph_cutter(5, ((0, 1), (2, 3)))
    ((0, 1), (2, 3), (4,))

    >>> graph_cutter(5, ((0, 1), (1, 2)))
    ((0,), (1,), (2,), (3, 4))
    """
    vertices = list(range(num_points))
    remaining = set(vertices)  # Use a set to track remaining vertices
    connectivity_tuple = []

    for vertex in vertices:
        if vertex not in remaining:  # Skip if already processed
            continue
        connected_component = [vertex]
        remaining.remove(vertex)  # Mark vertex as processed

        for other_vertex in list(remaining):  # Iterate over remaining vertices
            if all((vertex in cut) == (other_vertex in cut) for cut in cuts):
                # Check if vertex and other_vertex belong to same side of all cuts
                connected_component.append(other_vertex)
                remaining.remove(other_vertex)  # Mark other_vertex as processed

        connectivity_tuple.append(connected_component)  # Keep as list for now

    return tuple(map(tuple, connectivity_tuple))  # Convert only once at the end


def hitrate_1d(points):
    """Calculate the Poisson rate of hyperplanes hitting the convex hull of points in 1D.
    
    Parameters
    ----------
    points : np.array, shape (n,)
        The points to hit
        
    Returns
    -------
    float
        The rate of hyperplanes hitting the convex hull
        
    Examples
    --------
    >>> hitrate_1d(np.array([-2, -1, 0, 1, 2]))
    4
    """
    return max(points)-min(points) #hitrate is the length of segment


def hitrate_2d(points):
    """Calculate the Poisson rate of hyperplanes hitting the convex hull of points in 2D.
    
    Parameters
    ----------
    points : np.array, shape (n,2)
        The points to hit
        
    Returns
    -------
    float
        The rate of hyperplanes hitting the convex hull
        
    Examples
    --------
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
        return np.sum([np.linalg.norm(points[i] - points[(i+1) % 3]) for i in range(3)]) / 2
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
    """Calculate the dihedral angle between two normal vectors.
    
    Parameters
    ----------
    norm1 : np.array
        Normal vector of the first face
    norm2 : np.array
        Normal vector of the second face
        
    Returns
    -------
    float
        The dihedral angle in radians
        
    Examples
    --------
    >>> dihedral_angle(np.array([0, -1, 0]), np.array([1, 1, 1]))
    2.1862760354652844
    """
    cos_theta = np.dot(norm1, norm2) / (np.linalg.norm(norm1) * np.linalg.norm(norm2))
    # Ensure the cosine is within valid range due to potential numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return angle


def ensure_outward_facing(norm, point_on_face, centroid):
    """Ensure that the normal vector is outward-facing.
    
    Parameters
    ----------
    norm : np.array
        The normal vector to check
    point_on_face : np.array
        A point on the face
    centroid : np.array
        The centroid of the shape
        
    Returns
    -------
    np.array
        The outward-facing normal vector
    """
    to_centroid = centroid - point_on_face
    if np.dot(norm, to_centroid) > 0:
        return -norm  # Flip the normal to point outward
    return norm


def hitrate_3d(points):
    """Calculate the Poisson rate of hyperplanes hitting the convex hull of points in 3D.
    
    Parameters
    ----------
    points : np.array, shape (n, 3)
        3D array of points to hit
        
    Returns
    -------
    float
        The rate of hyperplanes hitting the convex hull
        
    Examples
    --------
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
        return np.sum([np.linalg.norm(points[i] - points[(i+1) % 3]) for i in range(3)]) / 2
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
                    adjacent_faces = [face for face in hull.simplices
                                    if edge[0] in face and edge[1] in face]
                    if len(adjacent_faces) < 2:
                        continue  # skip boundary edges
                    face1_edge1 = points[adjacent_faces[0][1]] - points[adjacent_faces[0][0]]
                    face1_edge2 = points[adjacent_faces[0][2]] - points[adjacent_faces[0][0]]
                    face2_edge1 = points[adjacent_faces[1][1]] - points[adjacent_faces[1][0]]
                    face2_edge2 = points[adjacent_faces[1][2]] - points[adjacent_faces[1][0]]

                    norm1 = ensure_outward_facing(
                        np.cross(face1_edge1, face1_edge2),
                        points[adjacent_faces[0][0]], centroid)
                    norm2 = ensure_outward_facing(
                        np.cross(face2_edge1, face2_edge2),
                        points[adjacent_faces[1][0]], centroid)

                    edge_length = np.linalg.norm(points[edge[1]] - points[edge[0]])
                    angle = dihedral_angle(norm1, norm2)
                    edge_contributions.append((edge, edge_length * angle))

    return sum(contribution for edge, contribution in edge_contributions) / (2 * np.pi)


def slash_rates_1d(points):
    """Return the rates of each single hyperplane partition of the points in 1D.
    
    Parameters
    ----------
    points : np.array, shape (n,)
        Sorted array of points to partition
        
    Returns
    -------
    dict
        The rates of each partition
        
    Examples
    --------
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
    """Return the rates of each single hyperplane partition of the points.
    
    Parameters
    ----------
    points : np.array, shape (n,d)
        The points to partition
        
    Returns
    -------
    dict
        The rates of each partition
        
    Examples
    --------
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
    dimension = points.shape[1]  # this block makes the function general for any dimension
    if dimension == 1:
        return slash_rates_1d(points)
    elif dimension == 2:
        hitrate_func = hitrate_2d
    elif dimension == 3:
        hitrate_func = hitrate_3d
    else:
        raise ValueError(f"Unsupported dimension: {dimension}")

    num_points = len(points)

    # loop over all possible partitions of the points into two disjoint sets
    # start with just one point in the first set and the rest in the second set

    if num_points == 2:
        projection = (points[1] - points[0]) / np.linalg.norm(points[1] - points[0])
        sorted_points = np.dot(points, projection)
        return slash_rates_1d(sorted_points)

    rates = defaultdict(int)
    whole_hitrate = hitrate_func(points)

    for i in range(num_points):  # single point partitions base case
        rest_hitrate = hitrate_func(np.delete(points, i, axis=0))
        rates[(i,)] = whole_hitrate - rest_hitrate

    if num_points == 3:
        return rates

    def subset_rates(subset):
        subset_size = len(subset)
        if subset_size == 1:
            return rates[subset]  # already calculated in base case

        rest_points = np.delete(points, list(subset), axis=0)
        rest_hitrate = hitrate_func(rest_points)

        my_rate = whole_hitrate - rest_hitrate  # next we need to subtract the subset rates
        for sub_size in range(1, subset_size):
            for sub_subset in itertools.combinations(subset, sub_size):
                my_rate -= rates[sub_subset]  # subtracting off rate of smaller subsets

        rates[subset] = my_rate  # can modify mutable dictionary in function scope
        return my_rate

    # calculate the rates of all other partitions up to the middle one which is an edge case
    for size in range(2, num_points//2):
        for subset in itertools.combinations(range(num_points), size):
            subset_rates(subset)

    if num_points % 2 == 0:  # have to be careful with middle partitions so as to not be redundant
        for subset in itertools.combinations(range(num_points), num_points//2):
            complement = tuple(sorted(set(range(num_points)) - set(subset)))
            # only keep the subset if its first element is less than the complement's first element
            if subset < complement:
                subset_rates(subset)
    else:  # num_points is odd
        for subset in itertools.combinations(range(num_points), num_points//2):
            subset_rates(subset)
    return rates


def color_from_partitions(partitions, colors, num_points, color_dist):
    """Return the color probability distribution for the final point given the partitions
    and colors of the other points.
    
    Parameters
    ----------
    partitions : tuple of tuples
        The single cut connectivity tuples of the points
    colors : tuple
        Length (num_points-1), the colors of the points
    num_points : int
        The number of points, including the final point whose color is unknown
    color_dist : tuple
        The probabilities of each color
        
    Returns
    -------
    np.array
        The calculated color probabilities for the final point
        
    Examples
    --------
    >>> partitions = ((0,), (1,), (0, 1))  # complete: ((0,), (1,), (2, 3)), color 1
    >>> colors = (1, 2, 1)
    >>> num_points = 4
    >>> color_dist = (.2,.2,.2,.2,.2)
    >>> color_from_partitions(partitions, colors, num_points, color_dist)
    array([0., 1., 0., 0., 0.])

    >>> partitions = ((0,), (0,3))  # complete: ((0,), (1,2), (3)), uniform
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

    # find which point, if any, points[-1] is connected to.
    # points[-1] is connected to points[i] if each partition has
    # ((num_points-1) in partition) == (i in partition)
    connected_point = None
    for i in range(num_points-1):
        if all(((num_points-1) in partition) == (i in partition)
                for partition in partitions):
            connected_point = i
            break
    if connected_point is not None:
        color_probs[colors[connected_point]] = 1
    else:  # if points[-1] is isolated, color is uniform
        color_probs = np.array(color_dist)
    return color_probs


def color_distribution(points, colors, color_dist):
    """Return the color probability distribution for points[-1].
    
    Should not be called with greater than 3 dimensions or more than 5 points.
    
    Parameters
    ----------
    points : np.array, shape (n,d)
        points[:-1] are the colored points,
        points[-1] is the point whose color is unknown
    colors : tuple of ints, shape (n-1)
        Colors of points[:-1]
    color_dist : tuple
        The probabilities of each color
        
    Returns
    -------
    np.array
        The calculated color distribution for the final point
        
    Examples
    --------
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
    if len(points) > 5 or points.shape[1] > 3:
        raise ValueError("color_distribution is not supported for "
        "more than 5 points or 3 dimensions. The run time is 2^(2^number of points), "
        "so it wont run for 6 points.")
    num_points = len(points)
    rates = slash_rates(points)
    partitions = list(rates.keys())
    all_connectivity = generate_all_connectivity_tuples(num_points)
    allowed_partitions = allowed_tuples_colors(all_connectivity, colors, last_color_unknown=True)

    num_p = len(partitions)
    ret = np.zeros(len(color_dist))

    # Precompute exp(-rates[partition]) for each partition
    exp_rates = {partition: np.exp(-rates[partition]) for partition in partitions}

    # calculate the probability of each member of the superset of all partitions
    # from there, use color_from_partitions to calculate the color distribution,
    # then add it to the final distribution
    power_set = itertools.chain.from_iterable(
        itertools.combinations(partitions, r) for r in range(num_p+1))
    probcount = 0
    for subset in power_set:  # remember, to convert a rate to a probability,
                               # do P(not happen) = e^(-rate)
        partition = graph_cutter(num_points, subset)
        if partition not in allowed_partitions:
            continue
        # Memoized product calculations
        subset_prob = math.prod([1 - exp_rates[partition] for partition in subset])
        complement_prob = math.prod([
            exp_rates[partition] for partition in partitions if partition not in subset])
        subset_prob *= complement_prob

        subset_colors = color_from_partitions(subset, colors, num_points, color_dist)
        ret += subset_prob * subset_colors
        probcount += subset_prob
    ret = ret / probcount  # normalize
    ret = np.clip(ret, 0, None)  # remove any tiny negatives
    ret /= np.sum(ret)  # re-normalize after clipping
    return ret


if __name__ == "__main__":
    import doctest
    doctest.testmod()
