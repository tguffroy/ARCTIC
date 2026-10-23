import heapq

import numpy as np
from scipy.optimize import linear_sum_assignment



def closest_point_to_origin(points):
    """
    Estimate the closest point to origin in a point cloud.

    Parameters
    ----------
    points : ndarray, shape (N,3)

    Returns
    -------
    idx : int
        Index of the closest point to the origin.
    """
    distances = np.linalg.norm(points, axis=1)
    idx = np.argmin(distances)
    return idx




def is_triangle_ambiguous(ab, ac, bc, tol=0.1):
    """
    Check if a triangle is ambiguous (collinear or isosceles).

    Parameters
    ----------
    ab, ac, bc : float
        Lengths of the sides of the triangle.
    tol : float between 0 and 1
        Tolerance for collinearity and isosceles checks.
        A low tolerance means more triangles will be considered ambiguous.
    Returns
    -------
    ambiguous : bool
        True if the triangle is ambiguous.
    """

    L = max(ab, ac, bc)
    threshold = 1 - tol

    e_col = min(abs(ab + ac - bc), abs(ab + bc - ac), abs(ac + bc - ab)) / L
    if e_col < threshold:
        return True

    e_iso = min(abs(ab - ac), abs(ab - bc), abs(ac - bc)) / L
    if e_iso < threshold:
        return True


def is_degenerate(points, tol=0.1):
    """
    Check if a point configuration is degenerate.

    Parameters
    ----------
    points : ndarray, shape (N,3)
        Point cloud to check.
    tol : float between 0 and 1
        Tolerance for rank estimation.
        A low tolerance means more configurations will be considered degenerate.

    Returns
    -------
    degenerate : bool
        True if the configuration is degenerate.
    """

    n = points.shape[0]
    threshold = 1 - tol

    if n < 3:
        return True

    centered = points - points.mean(axis=0)

    s = np.linalg.svd(centered, compute_uv=False)

    # normalize by characteristic size
    s /= s[0] + 1e-12

    if n == 3:
        return s[1] < threshold

    return s[2] < threshold




def kabsch_triangles(
    X,
    Y,
    use_anchor=True,
    anchor_idx1=None,
    anchor_idx2=None
):
    """
    Exact rigid transform from three corresponding points.

    Parameters
    ----------
    X : (3,3) array
        Source triangle.
    Y : (3,3) array
        Target triangle.
    use_anchor : bool
        If True, translation is computed from the anchor points.
        If False, translation is computed from centroids.
    anchor_idx1 : int or None
        Anchor index in X.
    anchor_idx2 : int or None
        Anchor index in Y.

    Returns
    -------
    R : (3,3) array
        Rotation matrix.
    t : (3,) array
        Translation vector.
    """

    if X.shape != (3, 3) or Y.shape != (3, 3):
        raise ValueError("X and Y must both contain exactly 3 points.")

    # --------------------------------------------------
    # Build orthonormal frame on X
    # --------------------------------------------------
    x0, x1, x2 = X

    ex1 = x1 - x0
    ex1 /= np.linalg.norm(ex1)

    ex2 = x2 - x0
    ex2 -= np.dot(ex2, ex1) * ex1
    ex2 /= np.linalg.norm(ex2)

    ex3 = np.cross(ex1, ex2)

    EX = np.column_stack((ex1, ex2, ex3))

    # --------------------------------------------------
    # Build orthonormal frame on Y
    # --------------------------------------------------
    y0, y1, y2 = Y

    ey1 = y1 - y0
    ey1 /= np.linalg.norm(ey1)

    ey2 = y2 - y0
    ey2 -= np.dot(ey2, ey1) * ey1
    ey2 /= np.linalg.norm(ey2)

    ey3 = np.cross(ey1, ey2)

    EY = np.column_stack((ey1, ey2, ey3))

    # --------------------------------------------------
    # Rotation
    # --------------------------------------------------
    R = EY @ EX.T

    # --------------------------------------------------
    # Translation
    # --------------------------------------------------
    if not use_anchor:
        x_ref = X.mean(axis=0)
        y_ref = Y.mean(axis=0)

    else:
        if anchor_idx1 is None or anchor_idx2 is None:
            raise ValueError("anchor_idx1 and anchor_idx2 must be provided when use_anchor=True")

        x_ref = X[anchor_idx1]
        y_ref = Y[anchor_idx2]

    t = y_ref - R @ x_ref

    return R, t




def kabsch(
    X,
    Y,
    use_anchor=True,
    anchor_idx1=None,
    anchor_idx2=None
):
    """
    Kabsch algorithm for rigid transformation estimation between two point clouds.

    Parameters
    ----------
    X : (N,3) array
        Source point cloud.
    Y : (N,3) array
        Target point cloud.
    use_anchor : bool
        If True, translation is computed from the anchor points.
        If False, translation is computed from centroids.
    anchor_idx1 : int or None
        Anchor index in X.
    anchor_idx2 : int or None
        Anchor index in Y.

    Returns
    -------
    R : (3,3) array
        Rotation matrix.
    t : (3,) array
        Translation vector.
    """

    n = X.shape[0]

    # Fast path for triangles
    if n == 3:
        if anchor_idx1 is None or anchor_idx2 is None:
            anchor_idx1 = closest_point_to_origin(X)
            anchor_idx2 = closest_point_to_origin(Y)
            
        return kabsch_triangles(
            X, Y,
            use_anchor=use_anchor,
            anchor_idx1=anchor_idx1,
            anchor_idx2=anchor_idx2
        )

    # ---------- centroids ----------
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)

    Xc = X - X_mean
    Yc = Y - Y_mean

    # ---------- covariance ----------
    H = Xc.T @ Yc

    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    # ---------- translation ----------
    if not use_anchor:
        t = Y_mean - R @ X_mean
    else:

        if anchor_idx1 is None or anchor_idx2 is None:
            anchor_idx1 = closest_point_to_origin(X)
            anchor_idx2 = closest_point_to_origin(Y)

        t = Y[anchor_idx2] - R @ X[anchor_idx1]

    return R, t



def arctic(
    X, Y,
    d1=None,
    d2=None,
    anchor_idx1=None,
    anchor_idx2=None,
    epsilon=0.8,
    epsilon_delta=None,
    alpha=1.0,
    n_iter=500,
    early_stop=np.inf,
    use_anchor=True,
    verbose=False
):
    """
    Find the best rigid transform between two point clouds, using anchored congruent triangles.

    Parameters
    ----------
    X, Y : arrays (N,3), (M,3)
        Point clouds
    d1, d2 : arrays (N,), (M,)
        Radial distances to anchor points (if None, will be computed)
    anchor_idx1, anchor_idx2 : int or None
        Indices of anchor points in X and Y (if None, will be estimated)
    epsilon : float
        Minimum distance after transformation for two points to match (in meters)
    epsilon_delta : float or None
        Optional relaxed threshold for anchor distance consistency (in meters).
        If None, set to epsilon.
    alpha : float between 0 and 1
        Tolerance for degeneracy check
    n_iter : int
        Number of iterations (max number of triangles to evaluate)
    early_stop : int
        Stop after this many iterations without improvement
    use_anchor : bool
        If True, the final estimated translation is computed from the anchor point
        If False, the final estimated translation is computed from the centroids of the matched points
    verbose : bool
        If True, print progress information

    Returns
    -------
    best_R : (3,3) array
        Estimated rotation matrix
    best_t : (3,) array
        Estimated translation vector
    best_matches : list of (i,j) index pairs of matches
    best_triangle : tuple of two lists of 3 indices
        Indices of the vertices of the best triangle-pair in X and Y
    """

    N, M = len(X), len(Y)

    if epsilon_delta is None:
        epsilon_delta = epsilon

    # --------------------------------------------------
    # Anchor indices
    # --------------------------------------------------
    if anchor_idx1 is None or anchor_idx2 is None:
        anchor_idx1 = closest_point_to_origin(X)
        anchor_idx2 = closest_point_to_origin(Y)

    # --------------------------------------------------
    # Anchor distances
    # --------------------------------------------------
    if d1 is None:
        d1 = np.linalg.norm(X - X[anchor_idx1], axis=1)

    if d2 is None:
        d2 = np.linalg.norm(Y - Y[anchor_idx2], axis=1)

    # --------------------------------------------------
    # Build candidate correspondences
    # --------------------------------------------------
    candidate_pairs = []

    for i in range(N):
        for j in range(M):

            # Avoid anchor points themselves
            if i == anchor_idx1 or j == anchor_idx2:
                continue

            e_anchor = abs(d1[i] - d2[j])

            if e_anchor <= epsilon_delta:
                candidate_pairs.append((i, j))

    if len(candidate_pairs) < 2:
        raise ValueError("ARCTIC: Not enough candidate correspondences.")

    # --------------------------------------------------
    # Build triangle list
    # --------------------------------------------------
    triangle_pairs = []

    # Precompute pairwise distances for efficiency
    DX = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    DY = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)

    for a in range(len(candidate_pairs)):
        i, j = candidate_pairs[a]

        for b in range(a + 1, len(candidate_pairs)):
            k, l = candidate_pairs[b]

            # avoid duplicated vertices
            if i == k or j == l:
                continue

            # Degeneracy check
            if is_triangle_ambiguous(d1[i], d1[k], DX[i, k], tol=alpha):
                continue
            if is_triangle_ambiguous(d2[j], d2[l], DY[j, l], tol=alpha):
                continue

            # Compute triangle error
            e3 = abs(DX[i, k] - DY[j, l])
            if e3 > epsilon_delta:
                continue

            e1 = abs(d1[i] - d2[j])
            e2 = abs(d1[k] - d2[l])

            triangle_error = np.sqrt((e1**2 + e2**2 + e3**2) / 3)

            # Append the triangle pair as vertice indices, an its associated triangle error
            triangle_pairs.append(( ([anchor_idx1, i, k], [anchor_idx2, j, l]), triangle_error ))

    if len(triangle_pairs) == 0:
        raise ValueError("ARCTIC: No geometrically consistent triangles found.")

    if verbose:
        print(f"ARCTIC: Found {len(triangle_pairs)} triangle hypotheses.")

    # Sort triangles by ascending triangle error (best first)
    triangle_pairs = heapq.nsmallest(n_iter, triangle_pairs, key=lambda x: x[1])

    # --------------------------------------------------
    # Evaluate triangle hypotheses
    # --------------------------------------------------
    best_matches = []
    best_residuals = None
    best_X_in = None
    best_Y_in = None
    cur_idx = -1

    # Initialize cost matrix and row/column indices for assignment problem
    cost = np.empty((N, M))
    
    # Find candidate matches in Y based on anchor distance consistency
    candidate_js = [np.where(np.abs(d1[i] - d2) <= epsilon_delta)[0] for i in range(N)]
    
    max_iter = min(n_iter, len(triangle_pairs))
    step = 0

    for idx in range(max_iter):

        step += 1

        # Extract triangle subsets
        Xs = X[triangle_pairs[idx][0][0]]
        Ys = Y[triangle_pairs[idx][0][1]]

        # Fast triangle alignment
        R, t = kabsch_triangles(
            Xs,
            Ys,
            use_anchor=True,
            anchor_idx1=0,
            anchor_idx2=0
        )

        # Fill cost matrix with large values
        BIG_COST = 1e12
        cost.fill(BIG_COST)

        # Initialize valid rows indices for assignment problem
        valid_rows = np.zeros(N, dtype=bool)

        # Compute transformed points
        X_trans = (R @ X.T).T + t

        # Build assignment cost matrix
        for i in range(N):

            if i == anchor_idx1:
                continue

            js = candidate_js[i]
            js = js[js != anchor_idx2]

            if js.size == 0:
                continue

            dists = np.linalg.norm(Y[js] - X_trans[i], axis=1)

            valid = dists < epsilon
            if np.any(valid):
                valid_rows[i] = True
                cost[i, js[valid]] = dists[valid]

        # Keep only columns with at least one valid correspondence and different from anchor
        valid_cols = np.any(cost < BIG_COST, axis=0)
        valid_cols[anchor_idx2] = False

        # Recover original indices
        rows_reduced = np.flatnonzero(valid_rows)
        cols_reduced = np.flatnonzero(valid_cols)

        # Skip if the reduced cost matrix is too small to find a better solution than the current best
        if min(rows_reduced.size, cols_reduced.size) < len(best_matches)-1:  # -1 because we always include the anchor
            continue

        reduced_cost = cost[np.ix_(rows_reduced, cols_reduced)]

        # Solve LAP on the reduced matrix
        row_ind, col_ind = linear_sum_assignment(reduced_cost)

        # Build consensus set
        matches = [(anchor_idx1, anchor_idx2)]
        residuals = [0.0]

        for r, c in zip(row_ind, col_ind):            

            # Skip matches with cost above the threshold
            if reduced_cost[r, c] >= epsilon:
                continue
            
            matches.append((rows_reduced[r], cols_reduced[c]))
            residuals.append(reduced_cost[r, c])  # add back the offset to get the actual distance

        # Compute residuals
        mean_residual = (np.mean(residuals) if residuals else np.inf)
        best_mean_residual = (np.mean(best_residuals) if best_residuals is not None else np.inf)

        # Best hypothesis is the one with the most matches,
        # and in case of a tie, the one with the lowest mean residual
        if (
            len(matches) > len(best_matches)
            or
            (
                len(matches) == len(best_matches)
                and mean_residual < best_mean_residual
            )
        ):

            X_in = np.array([X[ii] for ii, _ in matches])
            Y_in = np.array([Y[jj] for _, jj in matches])

            # Degeneracy check before updating the best model
            if (not is_degenerate(X_in, tol=alpha) and not is_degenerate(Y_in, tol=alpha)):
                best_matches = matches
                best_residuals = residuals.copy()
                best_X_in = X_in
                best_Y_in = Y_in
                cur_idx = idx
                step = 0

        # Early stopping
        if step >= early_stop:
            if verbose:
                print(f"ARCTIC: Early stopping after {idx+1} iterations with no improvement.")
            break

    # --------------------------------------------------
    # Compute final transformation
    # --------------------------------------------------

    if len(best_matches) == 0:
        raise ValueError("ARCTIC: No matches found. Try increasing the threshold or number of iterations.")

    if verbose:
        print(f"ARCTIC: Best triangle pair index {cur_idx} with {len(best_matches)} matches and mean residual {np.mean(best_residuals):.4f} m.")

    # Anchor-based Kabsch on all matches found
    best_R, best_t = kabsch(best_X_in, best_Y_in, use_anchor=use_anchor)

    return best_R, best_t, best_matches, np.array(triangle_pairs[cur_idx][0]) if cur_idx >= 0 else None



