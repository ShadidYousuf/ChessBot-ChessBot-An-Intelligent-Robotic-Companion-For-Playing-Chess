import cv2
import numpy as np


def get_saddle(gray_img):
    """
    Calculate the saddle points and gradients of the given grayscale image.

    Parameters:
        gray_img (numpy.ndarray): Input grayscale image.

    Returns:
        tuple: Saddle measure (S), subpixel offsets (sub_s, sub_t), and gradients (gx, gy).
    """
    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1)
    gxx = cv2.Sobel(gx, cv2.CV_32F, 1, 0)
    gyy = cv2.Sobel(gy, cv2.CV_32F, 0, 1)
    gxy = cv2.Sobel(gx, cv2.CV_32F, 0, 1)

    # Calculate the saddle measure (negative for maxima)
    S = -gxx * gyy + gxy**2

    # Calculate subpixel offsets
    denom = gxx * gyy - gxy**2
    sub_s = np.divide(
        gy * gxy - gx * gyy, denom, out=np.zeros_like(denom), where=denom != 0
    )
    sub_t = np.divide(
        gx * gxy - gy * gxx, denom, out=np.zeros_like(denom), where=denom != 0
    )

    return S, sub_s, sub_t, gx, gy


def fast_nonmax_suppression(img, win=11):
    """
    Perform fast non-maximum suppression using dilation.

    Parameters:
        img (numpy.ndarray): Input image.
        win (int): Window size for non-maximum suppression.
    """
    element = np.ones((win, win), np.uint8)
    img_dilate = cv2.dilate(img, element)
    peaks = cv2.compare(img, img_dilate, cv2.CMP_EQ)
    img[peaks == 0] = 0


def clip_bounding_points(pts, img_shape, win_size=10):
    """
    Clip points near the image boundaries.

    Parameters:
        pts (numpy.ndarray): Points in (x, y) coordinates.
        img_shape (tuple): Shape of the image (rows, cols).
        win_size (int): Minimum distance from the image boundary.

    Returns:
        numpy.ndarray: Filtered points within the valid region.
    """
    valid_mask = ~np.any(
        np.logical_or(
            pts <= win_size, pts[:, [1, 0]] >= np.array(img_shape) - win_size - 1
        ),
        axis=1,
    )
    return pts[valid_mask, :]


def get_final_saddle_points(img, win_size=10):
    """
    Detect saddle points in the image and refine them to subpixel accuracy.

    Parameters:
        img (numpy.ndarray): Input grayscale image.
        win_size (int): Minimum distance of saddle points from image boundaries.

    Returns:
        tuple: Saddle points, x-gradient (gx), and y-gradient (gy).
    """
    # Preprocess the image with a blur
    img = cv2.blur(img, (3, 3))

    # Calculate saddle points and gradients
    saddle, sub_s, sub_t, gx, gy = get_saddle(img)

    # Suppress non-maximum points
    fast_nonmax_suppression(saddle, win=11)

    # Filter out low-intensity points
    saddle[saddle < 10000] = 0
    sub_idxs = np.nonzero(saddle)

    # Extract points in (x, y) order
    spts = np.argwhere(saddle).astype(np.float64)[:, [1, 0]]

    # Add subpixel offsets
    subpixel_offsets = np.array([sub_s[sub_idxs], sub_t[sub_idxs]]).T
    spts += subpixel_offsets

    # Clip points near image boundaries
    spts = clip_bounding_points(spts, img.shape, win_size)

    return spts, gx, gy
