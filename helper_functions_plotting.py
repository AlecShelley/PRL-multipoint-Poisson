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

from helper_functions_analytic import * 

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