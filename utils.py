import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import os
import re

def smooth_ts(ts, window_size=5):
    if window_size < 2:
        return np.array(ts)
    window = np.ones(window_size) / window_size
    smoothed_ts = np.convolve(ts, window, mode='same')

    return smoothed_ts

def get_mean_distance(dtw_matrix):
    return dtw_matrix[-1][-1] / len(find_dtw_path(dtw_matrix))

def normalize_series(s):
    """
    Normalizes a time series to the range [0, 1].

    Args:
        s (list): The time series to normalize.

    Returns:
        list: The normalized time series.
    """
    min_s = min(s)
    max_s = max(s)
    if max_s - min_s == 0:
        return [0.0] * len(s)
    return [(x - min_s) / (max_s - min_s) for x in s]

def sanitize_filename(title):
    """
    Cleans a string to make it a valid filename.
    Replaces spaces with underscores and removes invalid characters.
    """
    title = title.replace(".", "_")
    # Replace spaces and certain separators with underscores
    filename = re.sub(r'[ \t=(),]+', '_', title.lower())
    # Remove all non-alphanumeric characters (except underscores)
    filename = re.sub(r'[\\/:*?"<>|]+', '', filename)
    # Strip leading/trailing underscores
    filename = filename.strip('_')
    
    if not filename:
        filename = "plot"
        
    return filename + ".png"

def plot_time_series(*series, ref_serie=None, title='Time Series Comparison', save_dir=None):
    """
    Plots one or more time series on the same graph.

    Args:
        series (list): A list of time series to plot.
        title (str, optional): The title of the plot.
        save_dir (str, optional): If provided, saves the plot to this directory
                                  instead of showing it.
    """
    
    if not series:
        print("No time series provided to plot.")
        return

    plt.figure(figsize=(12, 6))
    
    for i, s in enumerate(series):
        plt.plot(range(len(s)), s, '-', label=f's{i+1}')
    
    if ref_serie is not None:
        plt.plot(range(len(ref_serie)), ref_serie, 'k--', label='Reference Series')
    
    plt.title(title)
    plt.xlabel('Time Step (Index)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if save_dir:
        file_name = sanitize_filename(title)
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def compute_dtw_matrix(s1, s2):
    """
    Compute the Dynamic Time Warping (DTW) matrix between two time series.

    Args:
        s1 (list): The first time series.
        s2 (list): The second time series.

    Returns:
        list: The DTW matrix.
    """
    print("Computing standard DTW matrix")
    n, m = len(s1), len(s2)
    dtw_matrix = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    dtw_matrix[0][0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # print(f"Computing dtw_matrix[{i}][{j}]")
            cost = abs(s1[i - 1] - s2[j - 1])
            min_prev_cost = min(dtw_matrix[i - 1][j],
                                dtw_matrix[i][j - 1],
                                dtw_matrix[i - 1][j - 1])
            dtw_matrix[i][j] = cost + min_prev_cost
    return dtw_matrix

def compute_dtw_matrix_restricted(s1, s2, window):
    """
    Compute the Dynamic Time Warping (DTW) with local search restriction matrix between two time series

    Args:
        s1 (list): The first time series.
        s2 (list): The second time series.
        window (int): The size of the window for local restriction.

    Returns:
        list: The DTW matrix.
    """
    print(f"Computing restricted DTW matrix with window size {window}")
    n, m = len(s1), len(s2)
    dtw_matrix = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    window = max(window, abs(n - m))
    dtw_matrix[0][0] = 0
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window + 1)):
            # print(f"Computing dtw_matrix[{i}][{j}]")
            cost = abs(s1[i - 1] - s2[j - 1])
            min_prev_cost = min(dtw_matrix[i - 1][j],
                                dtw_matrix[i][j - 1],
                                dtw_matrix[i - 1][j - 1])
            dtw_matrix[i][j] = cost + min_prev_cost
    return dtw_matrix

def find_dtw_path(dtw_matrix):
    """
    Finds the optimal warping path in a DTW matrix.

    Args:
        dtw_matrix (list): The DTW matrix.

    Returns:
        list: The optimal warping path as a list of (i, j) tuples.
    """
    n = len(dtw_matrix) - 1
    m = len(dtw_matrix[0]) - 1
    path = []
    i, j = n, m
    path.append((i, j))
    while i > 0 or j > 0:
        if i == 0:
            j = j - 1
        elif j == 0:
            i = i - 1
        else:
            neighbors = {
                'diag': dtw_matrix[i - 1][j - 1],
                'up': dtw_matrix[i - 1][j],
                'left': dtw_matrix[i][j - 1]
            }
            min_cost = min(neighbors.values())
            if neighbors['diag'] == min_cost:
                i, j = i - 1, j - 1
            elif neighbors['up'] == min_cost:
                i, j = i - 1, j
            else:
                i, j = i, j - 1
        path.append((i, j))
    path.reverse()
    return path


def plot_dtw_matrix_with_path(dtw_matrix, path, title='DTW Cost Matrix', save_path=None):
    """
    Plots the DTW cost matrix with the optimal warping path overlaid.

    Args:
        dtw_matrix (list): The DTW matrix.
        path (list): The optimal warping path.
        title (str, optional): The title of the plot. Defaults to 'DTW Cost Matrix'.
        save_path (str, optional): If provided, saves the plot to this path. Defaults to None.
    """
    matrix = np.array(dtw_matrix)
    matrix[matrix == float('inf')] = np.nan
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, interpolation='nearest', cmap='viridis_r', origin='lower')
    fig.colorbar(im, ax=ax, label='Cumulative Cost')
    path_j = [p[1] for p in path]
    path_i = [p[0] for p in path]
    ax.plot(path_j, path_i, color='red', 
            marker='o', markersize=4, 
            linewidth=2, label='Optimal Path')
    ax.set_title(title)
    ax.set_xlabel('Sequence 2 Index (j)')
    ax.set_ylabel('Sequence 1 Index (i)')
    # ax.set_xticks(np.arange(matrix.shape[1]))
    # ax.set_yticks(np.arange(matrix.shape[0]))
    ax.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_dtw_alignment(s1, s2, method, window=None, title="DTW Alignment", save_dir=None, dtw_matrix=None):
    """
    Calculates and plots the DTW alignment between two time series.
    
    Args:
        s1 (list): The first time series.
        s2 (list): The second time series.
        method (str): 'standard' for standard DTW, 'restricted' for restricted DTW.
        window (int, optional): The window size for restricted DTW. Required if method is 'restricted'.
        title (str, optional): The title of the plot.
        save_dir (str, optional): If provided, saves the plot to this directory.
    """
    
    path = []
    title_suffix = ""
    
    if method == 'standard':
        if dtw_matrix is None:
            dtw_matrix = compute_dtw_matrix(s1, s2)
        path = find_dtw_path(dtw_matrix)
        title_suffix = "(Standard DTW)"
        
    elif method == 'restricted':
        if window is None:
            raise ValueError("A 'window' size must be provided for the 'restricted' method.")
        if dtw_matrix is None:
            dtw_matrix = compute_dtw_matrix_restricted(s1, s2, window)
        path = find_dtw_path(dtw_matrix)
        title_suffix = f"(Restricted DTW, w={window})"
        
    else:
        raise ValueError("Method must be 'standard' or 'restricted'.")

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    fig.suptitle(f'{title} {title_suffix} | DTW Distance: {dtw_matrix[-1][-1]}', fontsize=16)

    ax1.plot(s1, 'bo-', label='s1')
    ax1.set_title('Sequence 1')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax2.plot(s2, 'mo-', label='s2')
    ax2.set_title('Sequence 2')
    ax2.set_ylabel('Value')
    ax2.set_xlabel('Time Step (Index)')
    ax2.legend()
    for i, j in path:
        if i == 0 or j == 0:
            continue
        x1, y1 = i - 1, s1[i - 1]
        x2, y2 = j - 1, s2[j - 1]
        con = ConnectionPatch(
            xyA=(x1, y1), xyB=(x2, y2),
            coordsA=ax1.transData, coordsB=ax2.transData,
            linestyle='--', color='gray', alpha=0.7
        )
        fig.add_artist(con)

    plt.subplots_adjust(top=0.9)
    plt.tight_layout()
    
    if save_dir:
        file_name = file_name = sanitize_filename(title)
        
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_dtw_path(s1, s2, title="DTW Path", save_dir=None, dtw_matrix=None):
    """
    Plots the DTW cost matrix with the optimal warping path for standard DTW.

    Args:
        s1 (list): The first time series.
        s2 (list): The second time series.
        title (str, optional): The title of the plot. Defaults to "DTW Path".
        save_dir (str, optional): If provided, saves the plot to this directory. Defaults to None.
    """

    if dtw_matrix is None:
        dtw_matrix = compute_dtw_matrix(s1, s2)

    path_full = find_dtw_path(dtw_matrix)

    n, m = len(s1), len(s2)
    print("\n--- Standard DTW Distance---")
    print(f"Standard DTW Distance: {dtw_matrix[n][m]}")
    # print(f"Path: {path_full}")
    
    save_path = None
    if save_dir:
        file_name = sanitize_filename(title)
        save_path = os.path.join(save_dir, file_name)
        
    plot_dtw_matrix_with_path(dtw_matrix, path_full, title, save_path)


def plot_dtw_path_restricted(s1, s2, window_size, title="DTW Path Restricted", save_dir=None, dtw_matrix=None):
    """
    Plots the DTW with local search restriction cost matrix with the optimal warping path for restricted DTW.

    Args:
        s1 (list): The first time series.
        s2 (list): The second time series.
        window_size (int): The size of the restricted window.
        title (str, optional): The title of the plot. Defaults to "DTW Path Restricted".
        save_dir (str, optional): If provided, saves the plot to this directory. Defaults to None.
    """
    print("\n--- Restricted DTW Distance---")
    if dtw_matrix is None:
        dtw_matrix = compute_dtw_matrix_restricted(s1, s2, window_size)
    path_restricted = find_dtw_path(dtw_matrix)

    n, m = len(s1), len(s2)
    print(f"Restricted DTW Distance (w={window_size}): {dtw_matrix[n][m]}")
    # print(f"Path: {path_restricted}")

    save_path = None
    if save_dir:
        file_name = sanitize_filename(title)
        save_path = os.path.join(save_dir, file_name)

    plot_dtw_matrix_with_path(dtw_matrix, path_restricted, title, save_path)