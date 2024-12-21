import cv2
import einops
import numpy as np
import torch
import torchvision.transforms.functional as tf

import brutesac
import utils
from model import CNNModel, get_patch_model
from saddle_points import get_final_saddle_points

patch_model = get_patch_model()
patch_model.eval()
patch_model.load_state_dict(torch.load("patch_model_v2.pt"))

model = CNNModel([32, 64, 1024])
model.eval()
model.load_state_dict(torch.load("model.pt"))


def pred_patches(patches):
    patches = einops.rearrange(patches, "NH NW H W C -> (NH NW) C H W")
    patches = torch.from_numpy(patches / 255).float()

    with torch.no_grad():
        logits = patch_model(2 * patches - 1)

    preds = logits.argmax(dim=-1).reshape(8, 8)
    return preds.numpy()


@torch.no_grad()
def predict_fn(inputs):
    tiles = inputs["x"]
    probs = []

    for tile in tiles:
        img = tile.astype(np.uint8)
        img = tf.to_tensor(img)

        pred = model(2 * img[None] - 1)
        probs.append(pred.softmax(dim=1))

    return {"probabilities": torch.cat(probs, dim=0).numpy()}


def classify_image(frame, win_size=10, prob_threshold=0.5):
    spts, gx, gy = get_final_saddle_points(frame, win_size=win_size)
    return spts[predict_on_image(spts, frame, predict_fn, win_size) > prob_threshold]


def predict_on_tiles(tiles, predict_fn):
    """
    Predict on a batch of tiles using the provided prediction function.

    Parameters:
        tiles (numpy.ndarray): Array of tiles to predict on.
        predict_fn (function): Function to make predictions.

    Returns:
        numpy.ndarray: Array of probabilities of tiles being an x-corner.
    """
    predictions = predict_fn({"x": tiles})

    # Extract the probability of the second class (x-corner).
    return np.array([p[1] for p in predictions["probabilities"]])


def predict_on_image(pts, img, predict_fn, win_size=10):
    """
    Predict probabilities for points in the image using a tile-based classifier.

    Parameters:
        pts (numpy.ndarray): Points in (x, y) coordinates.
        img (numpy.ndarray): Input image.
        predict_fn (function): Prediction function.
        win_size (int): Half-size of the tile window.

    Returns:
        numpy.ndarray: Probabilities for each point being an x-corner.
    """
    # Build tiles from the image
    tiles = get_tiles_from_image(pts, img, win_size=win_size)

    # Predict probabilities for the tiles
    probs = predict_on_tiles(tiles, predict_fn)

    return probs


def get_tiles_from_image(pts, img, win_size=10):
    """
    Extract tiles centered around points in the image.

    Parameters:
        pts (numpy.ndarray): Points in (x, y) coordinates.
        img (numpy.ndarray): Input image.
        win_size (int): Half-size of the tile window.

    Returns:
        numpy.ndarray: Extracted tiles of shape (N, 2*win_size+1, 2*win_size+1).
    """
    tile_size = 2 * win_size + 1
    tiles = np.zeros((len(pts), tile_size, tile_size))

    for i, pt in enumerate(np.round(pts).astype(np.int64)):
        tiles[i] = img[
            pt[1] - win_size : pt[1] + win_size + 1,
            pt[0] - win_size : pt[0] + win_size + 1,
        ]

    return tiles


def get_board_patches(src_image):
    src_gray = cv2.cvtColor(src_image, cv2.COLOR_RGB2GRAY)
    xpts = utils.classify_image(frame=src_gray)
    (
        raw_M,
        best_quad,
        best_offset,
        best_score,
        best_error_score,
    ) = brutesac.brutesac_chessboard(xpts)
    M_homog = brutesac.refine_homography(xpts, raw_M, best_offset)

    ideal_grid_pts = np.vstack(
        [np.array([0, 0, 1, 1, 0]) * 8 - 1, np.array([0, 1, 1, 0, 0]) * 8 - 1]
    ).T

    # Refined via homography of all valid points
    unwarped_ideal_chess_corners_homography = cv2.perspectiveTransform(
        np.expand_dims(ideal_grid_pts.astype(float), 0), np.linalg.inv(M_homog)
    )[0, :, :]

    a, b, c, d = unwarped_ideal_chess_corners_homography[:-1]
    candidates = [(a, b, c, d), (b, c, d, a), (c, d, a, b), (d, a, b, c)]

    for pt_A, pt_B, pt_C, pt_D in candidates:
        if np.all((pt_C - pt_A) > 0):
            if np.all((pt_D - pt_B) * (1, -1) < 0):
                pt_D, pt_B = pt_B, pt_D
            break

        # Here, I have used L2 norm. You can use L1 also.
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32(
        [[0, 0], [0, maxHeight - 1], [maxWidth - 1, maxHeight - 1], [maxWidth - 1, 0]]
    )

    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts, output_pts)

    out = cv2.warpPerspective(
        src_image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR
    )
    out = cv2.resize(out, (256, 256))

    patches = einops.rearrange(out, "(h h1) (w w1) c -> h w h1 w1 c", h=8, w=8)
    return patches


def get_grayscale_patches(patches):
    NH, NW, H, W, C = patches.shape
    patches_reshaped = patches.reshape(-1, H, W, C)
    grayscale_patches = np.array(
        [cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) for patch in patches_reshaped]
    )
    grayscale_patches = grayscale_patches.reshape(NH, NH, H, W)
    return grayscale_patches


def erode_and_dilate_patches(patches):
    NH, NW, H, W = patches.shape
    patches_reshaped = patches.reshape(-1, H, W)
    eroded = np.array(
        [cv2.erode(patch, np.ones((5, 5), np.uint8)) for patch in patches_reshaped]
    )
    dilated = np.array(
        [cv2.dilate(patch, np.ones((5, 5), np.uint8)) for patch in eroded]
    )
    dilated = dilated.reshape(NH, NW, H, W)
    return dilated
