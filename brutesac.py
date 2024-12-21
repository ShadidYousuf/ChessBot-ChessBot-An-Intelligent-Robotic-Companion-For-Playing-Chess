import time
from functools import wraps

import cv2
import numpy as np
from scipy.spatial import Delaunay


def timed(f):
    """
    Decorator to measure the execution time of a function.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{f.__name__} took {elapsed * 1e3:.2f} ms to finish")
        return result

    return wrapper


def get_all_quads(tri):
    """
    Generate all possible unique quads from Delaunay triangles.

    Parameters:
        tri (Delaunay): Delaunay triangulation object.

    Returns:
        np.ndarray: Array of quads as indices of points in `tri.points`.
    """
    pairings = set()
    quads = []

    for i, neighbors in enumerate(tri.neighbors):
        for k in range(3):  # Each triangle has up to 3 neighbors
            nk = neighbors[k]
            if nk != -1:  # Valid neighbor exists
                pair = (i, nk)
                reverse_pair = (nk, i)
                if reverse_pair not in pairings:
                    pairings.add(pair)
                    b = tri.simplices[i]
                    d = tri.simplices[nk]
                    nk_vtx = (set(d) - set(b)).pop()
                    insert_mapping = [2, 3, 1]
                    b = np.insert(b, insert_mapping[k], nk_vtx)
                    quads.append(b)
    return np.array(quads)


def count_hits(given_pts, x_offset, y_offset):
    """
    Count matching points in a unity grid with given offsets.

    Parameters:
        given_pts (list): List of integer points.
        x_offset (int): Offset along the x-axis.
        y_offset (int): Offset along the y-axis.

    Returns:
        int: Number of matches.
    """
    pt_set = set(tuple(pt) for pt in given_pts)
    X, Y = np.meshgrid(np.arange(7) + x_offset, np.arange(7) + y_offset)
    matches = sum(1 for x, y in zip(X.flatten(), Y.flatten()) if (x, y) in pt_set)
    return matches


def get_best_board_matchup(given_pts):
    """
    Find the best offset for a chess grid that maximizes point matches.

    Parameters:
        given_pts (np.ndarray): Input points.

    Returns:
        tuple: Best score and the corresponding offset.
    """
    best_score = 0
    best_offset = None

    for i in range(7):
        for j in range(7):
            score = count_hits(given_pts, i - 6, j - 6)
            if score > best_score:
                best_score = score
                best_offset = [i - 6, j - 6]

    return best_score, best_offset


def score_quad(quad, pts, prev_best_score=0):
    """
    Score a quad by calculating its alignment with an ideal grid.

    Parameters:
        quad (np.ndarray): Quad to score.
        pts (np.ndarray): Input points.
        prev_best_score (int): Previous best score for comparison.

    Returns:
        tuple: Score, error score, transformation matrix, and offset.
    """
    ideal_quad = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), ideal_quad)
    pts_warped = cv2.perspectiveTransform(np.expand_dims(pts.astype(float), 0), M)[0]

    pts_warped_int = pts_warped.round().astype(int)
    M_refined, _ = cv2.findHomography(pts, pts_warped_int, cv2.RANSAC)
    M_refined = M_refined if M_refined is not None else M
    pts_warped = cv2.perspectiveTransform(
        np.expand_dims(pts.astype(float), 0), M_refined
    )[0]
    pts_warped_int = pts_warped.round().astype(int)

    score, offset = get_best_board_matchup(pts_warped_int)
    if score < prev_best_score:
        return score, None, None, None

    error_score = np.sum(np.linalg.norm((pts_warped - pts_warped_int), axis=1))
    return score, error_score, M, offset


def brutesac_chessboard(xcorner_pts):
    """
    RANSAC-based brute force search for the best chessboard quad.

    Parameters:
        xcorner_pts (np.ndarray): Detected x-corner points.

    Returns:
        tuple: Best transformation matrix, quad, offset, score, and error score.
    """
    tri = Delaunay(xcorner_pts)
    quads = get_all_quads(tri)

    best_score, best_error_score, best_M = 0, None, None
    best_quad, best_offset = None, None

    for quad in xcorner_pts[quads]:
        score, error_score, M, offset = score_quad(quad, xcorner_pts, best_score)
        if score > best_score or (
            score == best_score and error_score < best_error_score
        ):
            best_score, best_error_score, best_M = score, error_score, M
            best_quad, best_offset = quad, offset
            if best_score > len(xcorner_pts) * 0.8:
                break

    return best_M, best_quad, best_offset, best_score, best_error_score


def refine_homography(pts, M, best_offset):
    """
    Refine the homography matrix using inliers within the chessboard region.

    Parameters:
        pts (np.ndarray): Input points.
        M (np.ndarray): Initial transformation matrix.
        best_offset (list): Best offset found for the chessboard grid.

    Returns:
        np.ndarray: Refined homography matrix.
    """
    pts_warped = cv2.perspectiveTransform(np.expand_dims(pts.astype(float), 0), M)[0]
    warped_offsets = pts_warped.round() - best_offset

    # Filter points within the chessboard bounds
    in_bounds = ~np.any((warped_offsets < 0) | (warped_offsets > 7), axis=1)
    warped_offsets = warped_offsets[in_bounds]
    inlier_pts = pts[in_bounds]

    # Compute refined homography using Least-Median robust method
    M_refined, _ = cv2.findHomography(inlier_pts, warped_offsets, cv2.LMEDS)
    return M_refined
