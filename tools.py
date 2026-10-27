# Copyright (C) 2026    Thomas GUFFROY

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import arctic




# ----- Tools related to transformations between coordinate systems -----
def get_rotation_matrix(roll, pitch, yaw):
    """
    Compute the rotation matrix from roll, pitch, yaw angles (in radians)
    from the array coordinate system to the room coordinate system.

    Parameters:
        roll, pitch, yaw: rotation angles in radians

    Returns:
        R: 3x3 rotation matrix
    """

    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])
    return R




def rotate_and_translate(points, R, t):
    """
    Transform 3D point(s) from room frame to microphone frame.

    Parameters
    ----------
    points : array-like, shape (3,) or (3, N)
        Points in room frame.
    R : array-like, shape (3, 3)
        Rotation matrix (mic → room).
    t : array-like, shape (3,) or (1,3)
        Translation vector (mic position in room frame).

    Returns
    -------
    transformed_points : ndarray, shape (3,) or (3, N)
        Points expressed in microphone frame.
    """
    points = np.asarray(points)
    R = np.asarray(R)
    t = np.asarray(t).reshape(3,)  # ensure (3,)

    # Single point
    if points.ndim == 1:
        return R.T @ (points - t)

    # Multiple points (3, N)
    return R.T @ (points - t[:, None])


def rotate_and_translate_back(points, R, t):
    """
    Transform 3D point(s) from microphone frame back to room frame.

    Parameters
    ----------
    points : array-like, shape (3,) or (3, N)
        Points in microphone frame.
    R : array-like, shape (3, 3)
        Rotation matrix (mic → room).
    t : array-like, shape (3,) or (1,3)
        Translation vector (mic position in room frame).

    Returns
    -------
    transformed_points : ndarray, shape (3,) or (3, N)
        Points expressed in room frame.
    """
    points = np.asarray(points)
    R = np.asarray(R)
    t = np.asarray(t).reshape(3,)  # ensure (3,)

    # Single point
    if points.ndim == 1:
        return R @ points + t

    # Multiple points (3, N)
    return R @ points + t[:, None]




def compute_relative_transformation(R1, t1, R2, t2):
    """
    Compute the relative transformation (rotation and translation) from frame 1 to frame 2.

    Parameters:
        R1, R2: 3x3 numpy arrays representing rotation matrices of frame 1 and frame 2
        t1, t2: 3D translation vectors of frame 1 and frame 2

    Returns:
        R21: 3x3 numpy array representing the rotation from frame 1 to frame 2
        t21: 3D translation vector of frame 1 origin expressed in frame 2 coordinates
    """

    R1 = np.asarray(R1)
    R2 = np.asarray(R2)
    t1 = np.asarray(t1).reshape(3,)
    t2 = np.asarray(t2).reshape(3,)

    R21 = R2.T @ R1
    t21 = R2.T @ (t1 - t2)

    return R21, t21





# ----- Tools related to point cloud generation -----
def add_gaussian_noise(point_cloud, anchor_idx, sigma=0.01, mean=0.0, seed=None):
    """
    Add Gaussian noise with variance increasing with distance to the closest point to origin.

    Parameters:
        point_cloud (np.ndarray): Shape (D, N)
        anchor_idx (int): Index of the anchor point
        sigma (float): Maximum standard deviation (for farthest point)
        mean (float): Mean of the Gaussian noise
        seed (int or None): Seed for reproducibility

    Returns:
        np.ndarray: Noisy point cloud of same shape (D, N)
    """
    point_cloud = np.asarray(point_cloud)
    D, N = point_cloud.shape

    rng = np.random.default_rng(seed)

    # Step 1: find reference point (closest to origin)
    ref_point = point_cloud[:, anchor_idx].reshape(D, 1)

    # Step 2: compute distances to reference point
    distances = np.linalg.norm(point_cloud - ref_point, axis=0)

    # Step 3: normalize distances to [0, 1]
    if np.max(distances) > 0:
        distances_norm = distances / np.max(distances)
    else:
        distances_norm = distances  # all zeros case

    # Step 4: compute per-point sigma
    sigmas = sigma * distances_norm  # shape (N,)

    # Step 5: generate noise with per-point std
    noise = rng.normal(loc=mean, scale=1.0, size=(D, N))
    noise *= sigmas  # broadcast (D, N) * (N,)

    return point_cloud + noise



def remove_random_points(point_cloud, base_ratio=0.1, max_ratio=0.9,
                               exclude_idx=None, seed=None):
    """
    Remove points with probability increasing with distance to origin.

    Parameters:
        point_cloud (np.ndarray): Shape (D, N)
        base_ratio (float): Minimum removal probability (near origin)
        max_ratio (float): Maximum removal probability (far away)
        exclude_idx (int or None): Index to never remove
        seed (int or None): Seed for reproducibility

    Returns:
        np.ndarray: Filtered point cloud
        np.ndarray: Removed indices (relative to input)
    """
    point_cloud = np.asarray(point_cloud)
    D, N = point_cloud.shape

    if not (0 <= base_ratio <= max_ratio <= 1):
        raise ValueError("Require 0 <= base_ratio <= max_ratio <= 1")

    rng = np.random.default_rng(seed)

    # Step 1: distance to origin
    distances = np.linalg.norm(point_cloud, axis=0)

    # Step 2: normalize to [0, 1]
    if np.max(distances) > 0:
        d_norm = distances / np.max(distances)
    else:
        d_norm = distances

    # Step 3: map to removal probabilities
    probs = base_ratio + (max_ratio - base_ratio) * d_norm

    # Step 4: sample removal
    random_vals = rng.random(N)
    remove_mask = random_vals < probs

    # Ensure excluded index is kept
    if exclude_idx is not None:
        remove_mask[exclude_idx] = False

    # Build outputs
    keep_mask = ~remove_mask
    removed_indices = np.where(remove_mask)[0]

    return point_cloud[:, keep_mask], removed_indices



def generate_point_clouds(
    room_dim,
    src_pos,
    R1,
    t1,
    R2,
    t2,
    max_order_ism=5,
    sigma_noise=0.3,
    base_ratio_1=0.05,
    max_ratio_1=0.6,
    base_ratio_2=0.05,
    max_ratio_2=0.7,
    max_distance_to_mics=12,
    seed_noise=42
):

    # Generate perfect image sources in the room's coordinate system
    room_perfect = pra.ShoeBox(
        room_dim,
        max_order=max_order_ism,
        ray_tracing=False,
        use_rand_ism=False,
    )

    room_perfect.add_source(src_pos)
    room_perfect.add_microphone(t1)
    room_perfect.add_microphone(t2)
    room_perfect.image_source_model()
    img_sources_perfect = room_perfect.sources[0].images

    # Add noise to image source positions
    img1 = rotate_and_translate(img_sources_perfect, R1, t1)
    idx_src_1 = arctic.closest_point_to_origin(img1.T)  # index of true source in img1
    img1 = add_gaussian_noise(img1, anchor_idx=idx_src_1, sigma=sigma_noise, seed=seed_noise)

    # Create second cloud by rotating and translating, and add noise
    img2 = rotate_and_translate(img_sources_perfect, R2, t2)
    idx_src_2 = arctic.closest_point_to_origin(img2.T)  # index of true source in img2
    img2 = add_gaussian_noise(img2, anchor_idx=idx_src_2, sigma=sigma_noise, seed=seed_noise+1)  # Different seed for different noise

    # Initial indices
    indices_1 = np.arange(img1.shape[1])
    indices_2 = np.arange(img2.shape[1])

    # --- Random removal (img1) ---
    img1, idx_rmv_local_1 = remove_random_points(img1, base_ratio=base_ratio_1, max_ratio=max_ratio_1, exclude_idx=idx_src_1, seed=seed_noise)
    idx_rmv_1 = indices_1[idx_rmv_local_1]  # map back to original
    indices_1 = np.delete(indices_1, idx_rmv_local_1)

    # --- Random removal (img2) ---
    img2, idx_rmv_local_2 = remove_random_points(img2, base_ratio=base_ratio_2, max_ratio=max_ratio_2, exclude_idx=idx_src_2, seed=seed_noise+1)
    idx_rmv_2 = indices_2[idx_rmv_local_2]
    indices_2 = np.delete(indices_2, idx_rmv_local_2)

    # --- Distance filtering (img1) ---
    distances = np.linalg.norm(img1.T - t1, axis=1)
    mask = distances <= max_distance_to_mics

    idx_too_far_1 = indices_1[~mask]
    img1 = img1.T[mask].T
    indices_1 = indices_1[mask]   # update tracking

    # --- Distance filtering (img2) ---
    distances = np.linalg.norm(img2.T - t2, axis=1)
    mask = distances <= max_distance_to_mics

    idx_too_far_2 = indices_2[~mask]
    img2 = img2.T[mask].T
    indices_2 = indices_2[mask]   # update tracking

    # Final removed indices (all in ORIGINAL indexing)
    num_original = img_sources_perfect.shape[1]
    removed_indices_1 = np.concatenate([idx_too_far_1, idx_rmv_1])
    removed_indices_2 = np.concatenate([idx_too_far_2, idx_rmv_2])

    # --- True matches (ground truth) ---
    map_orig_to_img2 = -np.ones(num_original, dtype=int)
    map_orig_to_img2[indices_2] = np.arange(len(indices_2))
    matches_1_to_2 = map_orig_to_img2[indices_1]   # shape = (img1 points,)

    # Keep only valid matches
    valid_mask = matches_1_to_2 != -1

    # Indices in img1
    idx1 = np.where(valid_mask)[0]

    # Corresponding indices in img2
    idx2 = matches_1_to_2[valid_mask]

    # List of tuples
    matches_pairs = list(zip(idx1, idx2))

    return img1, img2, removed_indices_1, removed_indices_2, matches_pairs, img_sources_perfect



# ----- Tools related to plots and prints -----
def plot_point_clouds(pc1, pc2, show_indices=False, indices=None, labels=("Point Cloud 1", "Point Cloud 2")):
    """
    Plot two 3D point clouds using Plotly (interactive),
    with point indices displayed as labels.
    Also plots the origin (0,0,0) with a distinct style.
    """

    import numpy as np
    import plotly.graph_objects as go

    pc1 = np.asarray(pc1)
    pc2 = np.asarray(pc2)

    N = pc1.shape[1]
    M = pc2.shape[1]

    # Create index labels
    if show_indices:
        if indices is not None:
            text1 = [str(i) for i in indices[:, 0]]
            text2 = [str(i) for i in indices[:, 1]]
        else:
            text1 = [str(i) for i in range(N)]
            text2 = [str(i) for i in range(M)]
    
    else:
        text1 = [''] * N
        text2 = [''] * M

    # --- Point cloud 1 ---
    trace1 = go.Scatter3d(
        x=pc1[0, :],
        y=pc1[1, :],
        z=pc1[2, :],
        mode='markers+text',
        name=labels[0],
        marker=dict(size=3),
        text=text1,
        textposition='top center'
    )

    # --- Point cloud 2 ---
    trace2 = go.Scatter3d(
        x=pc2[0, :],
        y=pc2[1, :],
        z=pc2[2, :],
        mode='markers+text',
        name=labels[1],
        marker=dict(size=3),
        text=text2,
        textposition='top center'
    )

    # --- Origin point ---
    origin_trace = go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers+text',
        name='Origin',
        marker=dict(
            size=8,              # larger than others
            color='red',         # distinct color
            symbol='diamond'     # different shape
        ),
        text=['(0,0,0)'],
        textposition='bottom center'
    )

    # Compute equal aspect ratio
    all_points = np.hstack((pc1, pc2, np.array([[0], [0], [0]])))
    center = all_points.mean(axis=1)
    max_range = (all_points.max(axis=1) - all_points.min(axis=1)).max() * 1.5  # add some margin

    # Layout
    layout = go.Layout(
        width=900,
        height=800,
        scene=dict(
            xaxis=dict(title='X', range=[center[0] - max_range/2, center[0] + max_range/2]),
            yaxis=dict(title='Y', range=[center[1] - max_range/2, center[1] + max_range/2]),
            zaxis=dict(title='Z', range=[center[2] - max_range/2, center[2] + max_range/2]),
            aspectmode='cube'
        ),
        legend=dict(x=0, y=1)
    )

    fig = go.Figure(data=[trace1, trace2, origin_trace], layout=layout)
    fig.show()




def plot_room_with_sources(
    room_dim,
    source_pos,
    mic_positions,
    image_sources=None,
    point_size=5,
    zoom=1.0,
    show_background=True
):
    """
    Plot a shoebox room as a wireframe with a sound source and microphone arrays.
    Optionally plot image sources.

    Parameters
    ----------
    room_dim : tuple/list of 3 floats
        Dimensions of the room (length_x, width_y, height_z)

    source_pos : array-like of 3 floats
        Position of the main sound source [x, y, z]

    mic_positions : list of array-like
        List of microphone array positions. Each is [x, y, z]

    image_sources : np.ndarray or None, optional
        Array of shape (3, M) for M image sources.
        If None or empty, image sources are not plotted.

    point_size : int
        Marker size for points

    zoom : float
        Camera zoom factor

    show_background : bool
        Whether to show the axis background, grid, labels, and zero lines.
    """

    room_dim = np.asarray(room_dim)
    source_pos = np.asarray(source_pos)
    mic_positions = np.asarray(mic_positions)

    # Convert image sources only if provided
    if image_sources is not None:
        image_sources = np.asarray(image_sources)

    fig = go.Figure()

    # ------------------------------------------------------------
    # 1. Draw room as black wireframe
    # ------------------------------------------------------------
    L, W, H = room_dim

    vertices = np.array([
        [0, 0, 0],
        [L, 0, 0],
        [L, W, 0],
        [0, W, 0],
        [0, 0, H],
        [L, 0, H],
        [L, W, H],
        [0, W, H]
    ])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7)   # verticals
    ]

    for i, j in edges:
        fig.add_trace(go.Scatter3d(
            x=[vertices[i, 0], vertices[j, 0]],
            y=[vertices[i, 1], vertices[j, 1]],
            z=[vertices[i, 2], vertices[j, 2]],
            mode="lines",
            line=dict(color="black", width=3),
            showlegend=False
        ))

    # ------------------------------------------------------------
    # 2. Main sound source
    # ------------------------------------------------------------
    fig.add_trace(go.Scatter3d(
        x=[source_pos[0]],
        y=[source_pos[1]],
        z=[source_pos[2]],
        mode="markers",
        marker=dict(
            size=point_size * 2,
            color="red"
        ),
        name="Sound Source"
    ))

    # ------------------------------------------------------------
    # 3. Microphone arrays
    # ------------------------------------------------------------
    fig.add_trace(go.Scatter3d(
        x=mic_positions[:, 0],
        y=mic_positions[:, 1],
        z=mic_positions[:, 2],
        mode="markers",
        marker=dict(
            size=point_size * 1.5,
            color="blue",
            symbol="diamond"
        ),
        name="Mic Arrays"
    ))

    # ------------------------------------------------------------
    # 4. Image sources (only if provided and non-empty)
    # ------------------------------------------------------------
    has_image_sources = (
        image_sources is not None
        and image_sources.size > 0
    )

    if has_image_sources:
        fig.add_trace(go.Scatter3d(
            x=image_sources[0],
            y=image_sources[1],
            z=image_sources[2],
            mode="markers",
            marker=dict(
                size=point_size,
                color="green",
                opacity=0.6
            ),
            name="Image Sources"
        ))

    # ------------------------------------------------------------
    # 5. Determine axis ranges
    # ------------------------------------------------------------
    points = [
        np.asarray(source_pos).reshape(3, 1),
        mic_positions.T
    ]

    # Only include image sources in axis ranges when they exist
    if has_image_sources:
        points.append(image_sources)

    all_points = np.hstack(points)

    x_min, x_max = np.min(all_points[0]), np.max(all_points[0])
    y_min, y_max = np.min(all_points[1]), np.max(all_points[1])
    z_min, z_max = np.min(all_points[2]), np.max(all_points[2])

    # Always include the room boundaries
    x_min = min(x_min, 0)
    y_min = min(y_min, 0)
    z_min = min(z_min, 0)

    x_max = max(x_max, room_dim[0])
    y_max = max(y_max, room_dim[1])
    z_max = max(z_max, room_dim[2])

    # ------------------------------------------------------------
    # 6. Layout
    # ------------------------------------------------------------
    zoom_factor = 1.0 / zoom

    fig.update_layout(
        title=(
            "Room with Source, Microphones and Image Sources"
            if has_image_sources
            else "Room with Source and Microphones"
        ),
        width=900,
        height=800,
        scene=dict(
            aspectmode="data",
            camera=dict(
                eye=dict(
                    x=zoom_factor,
                    y=zoom_factor,
                    z=zoom_factor
                )
            ),
            xaxis=dict(
                showbackground=show_background,
                showgrid=show_background,
                zeroline=show_background,
                showticklabels=show_background,
                title="X" if show_background else "",
                range=[x_min - 1, x_max + 1]
            ),
            yaxis=dict(
                showbackground=show_background,
                showgrid=show_background,
                zeroline=show_background,
                showticklabels=show_background,
                title="Y" if show_background else "",
                range=[y_min - 1, y_max + 1]
            ),
            zaxis=dict(
                showbackground=show_background,
                showgrid=show_background,
                zeroline=show_background,
                showticklabels=show_background,
                title="Z" if show_background else "",
                range=[z_min - 1, z_max + 1]
            ),
        )
    )

    fig.show()





def plot_transform_comparison(R1, R2, t1, t2, labels=None, scale=1.0):
    """
    Compare two rigid transformations (R1, t1) and (R2, t2).

    Left plot: rotation matrices (axes)
    Right plot: translation vectors

    Parameters
    ----------
    R1, R2 : (3,3) arrays
        Rotation matrices
    t1, t2 : (3,) arrays
        Translation vectors
    scale : float
        Length of axes
    """

    R1 = np.asarray(R1)
    R2 = np.asarray(R2)
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)

    origin = np.zeros(3)
    basis = np.eye(3)

    # Rotated bases
    R1_basis = R1 @ basis
    R2_basis = R2 @ basis

    colors = ['r', 'g', 'b']
    labels = labels if labels is not None else ['R1', 'R2',  't1', 't2']

    fig = plt.figure(figsize=(12, 6))

    # ==================================================
    # 1. Rotation comparison
    # ==================================================
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_title("Rotation Comparison")

    # Original axes (reference)
    for i in range(3):
        # Original axes (faint reference)
        v = basis[:, i]
        ax1.plot(
            [-v[0]*scale, v[0]*scale],
            [-v[1]*scale, v[1]*scale],
            [-v[2]*scale, v[2]*scale],
            color=colors[i],
            alpha=0.3
        )

    # R1 axes
    for i in range(3):
        ax1.quiver(*origin,
                   *(R1_basis[:, i] * scale),
                   color=colors[i],
                   linewidth=1.5,
                   label=f'{labels[0]}' if i == 0 else None)

    # R2 axes
    for i in range(3):
        ax1.quiver(*origin,
                   *(R2_basis[:, i] * scale),
                   color=colors[i],
                   linestyle='dashed',
                   linewidth=1.5,
                   label=f'{labels[1]}' if i == 0 else None)

    ax1.set_xlim([-scale, scale])
    ax1.set_ylim([-scale, scale])
    ax1.set_zlim([-scale, scale])

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax1.legend()

    # ==================================================
    # 2. Translation comparison
    # ==================================================
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title("Translation Comparison")

    # Plot translation vectors
    ax2.quiver(*origin, *t1, color='black', linewidth=1.5, label=labels[2])
    ax2.quiver(*origin, *t2, color='orange', linewidth=1.5, label=labels[3])

    # Also show endpoints
    ax2.scatter(*t1, color='black')
    ax2.scatter(*t2, color='orange')

    # Scale limits based on max translation
    max_range = max(np.linalg.norm(t1), np.linalg.norm(t2), scale)

    ax2.set_xlim([-max_range, max_range])
    ax2.set_ylim([-max_range, max_range])
    ax2.set_zlim([-max_range, max_range])

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    ax2.legend()

    plt.tight_layout()
    plt.show()




def compute_metrics(R21, R21_true, t21, t21_true):
    """
    Compute metrics for the estimated relative transformation.

    Parameters:
        R21: Estimated rotation matrix (3x3)
        t21: Estimated translation vector (3,)
        R21_true: Ground truth rotation matrix (3x3)
        t21_true: Ground truth translation vector (3,)
    
    Returns:
        metrics: dict containing:
            - 'geodesic_distance': Rotation error in degrees
            - 'angular_error': Angular error between the two translation vectorsin degrees
            - 'inter-array_distance_error': Difference in distance between the two microphone arrays
            - 'euclidean_distance_error': Translation error (Euclidean distance)
    """

    def compute_geodesic_distance(R1, R2):
        # Relative rotation matrix
        R = R1.T @ R2
        trace_R = np.trace(R)

        cos_theta = (trace_R - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        theta = np.arccos(cos_theta)

        return np.degrees(theta)

    def compute_angular_error(t1, t2):
        t1_norm = t1 / np.linalg.norm(t1)
        t2_norm = t2 / np.linalg.norm(t2)

        dot_product = np.clip(np.dot(t1_norm, t2_norm), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)

        return np.degrees(angle_rad)
    
    def compute_inter_array_distance_error(t1, t2):
        dist1 = np.linalg.norm(t1)
        dist2 = np.linalg.norm(t2)

        return abs(dist1 - dist2)
    
    def compute_euclidean_distance_error(t1, t2):
        return np.linalg.norm(t1 - t2)


    metrics = {
        'geodesic_distance': compute_geodesic_distance(R21, R21_true),
        'angular_error': compute_angular_error(t21, t21_true),
        'inter_array_distance_error': compute_inter_array_distance_error(t21, t21_true),
        'euclidean_distance_error': compute_euclidean_distance_error(t21, t21_true)
    }

    return metrics



def print_performance(R21, R21_true, t21, t21_true, matches, matches_true):
    """
    Show performance metrics for estimated relative transformation.

    Parameters:
        R21: Estimated rotation matrix (3x3)
        t21: Estimated translation vector (3,)
        R21_true: Ground truth rotation matrix (3x3)
        t21_true: Ground truth translation vector (3,)
        matches: Estimated matches
        matches_true: Ground truth matches
    """

    metrics = compute_metrics(R21, R21_true, t21, t21_true)

    # Main metrics
    print("PERFORMANCE METRICS")
    print(f"    Geodesic Distance:                        {metrics['geodesic_distance']:.3f}°")
    print(f"    Angular Error:                            {metrics['angular_error']:.3f}°")
    print(f"    Inter-Array Distance Error:               {metrics['inter_array_distance_error']:.3f} m")
    print(f"    Euclidean Distance Error:                 {metrics['euclidean_distance_error']:.3f} m")

    # Matches
    print(f"    Number of matches found:                  {len(matches)} out of {len(matches_true)} possible matches")
    print(f"    Number of correct matches:                {sum(1 for (i, j) in matches if (i, j) in matches_true)}")





def compute_metrics_multi(data, geodesic_distance_max=10, euclidean_distance_error_max=0.5):
    """
    Compute a metric from the raw experiment data.
    """

    geodesic_distances = np.asarray(data["geodesic_distances"])
    angular_errors = np.asarray(data["angular_errors"])
    euclidean_distance_errors = np.asarray(data["euclidean_distance_errors"])
    inter_array_distance_errors = np.asarray(data["inter_array_distance_errors"])

    nb_matches = np.asarray(data["nb_matches"])
    matches_precisions = np.asarray(data["matches_precisions"]) if data["matches_precisions"] is not None else None
    matches_recalls = np.asarray(data["matches_recalls"]) if data["matches_recalls"] is not None else None


    nb_rooms = data["nb_rooms"]

    # Compute alignment statistics
    nb_correct_alignments = np.sum((geodesic_distances < geodesic_distance_max) & (euclidean_distance_errors < euclidean_distance_error_max))
    correct_alignments_ratio = nb_correct_alignments / nb_rooms

    nb_incorrect_alignments = nb_rooms - nb_correct_alignments
    incorrect_alignments_ratio = nb_incorrect_alignments / nb_rooms

    nb_failure_alignments = np.sum(np.isnan(geodesic_distances))
    failure_alignments_ratio = nb_failure_alignments / nb_rooms

    nb_flipped_alignments = np.sum((geodesic_distances > 180-geodesic_distance_max))
    flipped_alignments_ratio = nb_flipped_alignments / nb_rooms

    nb_wrong_alignments = nb_rooms - nb_correct_alignments - nb_failure_alignments - nb_flipped_alignments
    wrong_alignments_ratio = nb_wrong_alignments / nb_rooms

    nb_matches_mean = np.nanmean(nb_matches) if nb_matches.size > 0 else None
    matches_precision_mean = np.nanmean(matches_precisions) if matches_precisions is not None and matches_precisions.size > 0 else None
    matches_recall_mean = np.nanmean(matches_recalls) if matches_recalls is not None and matches_recalls.size > 0 else None

    metrics = {
        # Alignments statistics
        "nb_correct_alignments": nb_correct_alignments,
        "nb_incorrect_alignments": nb_incorrect_alignments,
        "nb_failure_alignments": nb_failure_alignments,
        "nb_flipped_alignments": nb_flipped_alignments,
        "nb_wrong_alignments": nb_wrong_alignments,
        "correct_alignments_ratio": correct_alignments_ratio,
        "incorrect_alignments_ratio": incorrect_alignments_ratio,
        "failure_alignments_ratio": failure_alignments_ratio,
        "flipped_alignments_ratio": flipped_alignments_ratio,
        "wrong_alignments_ratio": wrong_alignments_ratio,

        # Averaged metrics
        "geodesic_distance_mean": np.nanmean(geodesic_distances),
        "angular_error_mean": np.nanmean(angular_errors),
        "euclidean_distance_error_mean": np.nanmean(euclidean_distance_errors),
        "inter_array_distance_error_mean": np.nanmean(inter_array_distance_errors),

        "geodesic_distance_mean_correct_alignments": np.mean(geodesic_distances[(geodesic_distances < geodesic_distance_max) & (euclidean_distance_errors < euclidean_distance_error_max)]),
        "angular_error_mean_correct_alignments": np.mean(angular_errors[(geodesic_distances < geodesic_distance_max) & (euclidean_distance_errors < euclidean_distance_error_max)]),
        "euclidean_distance_error_mean_correct_alignments": np.mean(euclidean_distance_errors[(geodesic_distances < geodesic_distance_max) & (euclidean_distance_errors < euclidean_distance_error_max)]),
        "inter_array_distance_error_mean_correct_alignments": np.mean(inter_array_distance_errors[(geodesic_distances < geodesic_distance_max) & (euclidean_distance_errors < euclidean_distance_error_max)]),

        # Matches statistics
        "nb_matches_mean": nb_matches_mean,
        "matches_precision_mean": matches_precision_mean,
        "matches_recall_mean": matches_recall_mean,
    }

    return metrics




def print_performance_multi(data, geodesic_distance_max=10, euclidean_distance_error_max=0.5):
    """
    Show performance metrics for estimated relative transformation,
    for multiple experiments (e.g., multiple rooms).

    Parameters:
        data: dict containing lists of metrics for multiple experiments
        geodesic_distance_max: threshold for considering an alignment correct (in degrees)
        euclidean_distance_error_max: threshold for considering an alignment correct (in meters)
    """

    metrics = compute_metrics_multi(data, geodesic_distance_max, euclidean_distance_error_max)

    # Main metrics
    print(f"PERFORMANCE METRICS ON {data['nb_rooms']} ROOMS")
    print(f"    Number of correct alignments:             {metrics['nb_correct_alignments']} ({metrics['correct_alignments_ratio']*100:.2f}%)")
    print(f"    Number of incorrect alignments:           {metrics['nb_incorrect_alignments']} ({metrics['incorrect_alignments_ratio']*100:.2f}%)")
    print(f"    Number of flipped alignments:             {metrics['nb_flipped_alignments']} ({metrics['flipped_alignments_ratio']*100:.2f}%)")

    print(f"    Average number of matches:                {metrics['nb_matches_mean']:.2f}")
    if metrics['matches_precision_mean'] is not None and metrics['matches_recall_mean'] is not None:
        print(f"    Average match precision:                  {metrics['matches_precision_mean']*100:.2f}%")
        print(f"    Average match recall:                     {metrics['matches_recall_mean']*100:.2f}%")

    print(f"\nFor the {metrics['nb_correct_alignments']} correct alignments only (geodesic distance < {geodesic_distance_max}°, euclidean distance error < {euclidean_distance_error_max} m):")

    print(f"    Average Geodesic Distance:               {metrics['geodesic_distance_mean_correct_alignments']:.3f}°")
    print(f"    Average Angular Error:                   {metrics['angular_error_mean_correct_alignments']:.3f}°")
    print(f"    Average Inter-Array Distance Error:      {metrics['inter_array_distance_error_mean_correct_alignments']:.3f} m")
    print(f"    Average Euclidean Distance Error:        {metrics['euclidean_distance_error_mean_correct_alignments']:.3f} m")