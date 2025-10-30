# ChessBot: An Intelligent Robotic Chess Companion
# Technical Documentation

## ChessBot: An Intelligent Robotic Companion For Playing Chess

The ChessBot project is designed to create an intelligent robotic companion for playing chess. The architecture consists of several key components:

1. **Inverse Kinematics**: This module handles the inverse kinematics for the robotic arm, allowing it to move to specific positions based on the chessboard coordinates.
2. **Chessboard Detection**: The `detect.py` module uses OpenCV to detect the chessboard and its positions.
3. **Piece Detection**: The `saddle_points.py` module calculates saddle points and gradients to detect the presence of chess pieces on the board.
4. **Model Training**: The `train.ipynb` module trains a Convolutional Neural Network (CNN) to classify tiles as good or bad.
5. **Prediction**: The `utils.py` module contains functions to predict the presence of chess pieces and classify images.
6. **Main Execution**: The `main.py` and `main.ipynb` files contain the main logic for executing the chessbot.

### Setup & Installation

To set up and install the ChessBot project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/ChessBot-ChessBot-An-Intelligent-Robotic-Companion-For-Playing-Chess.git
   cd ChessBot-ChessBot-An-Intelligent-Robotic-Companion-For-Playing-Chess
   ``
2. **Download Pre-trained Models**:
   Ensure you have the pre-trained models (`patch_model_v2.pt` and `model.pt`) in the appropriate directory.



The project uses several Python modules and functions. Below is a brief overview of the key functions and their usage:

- **`brutesac.py`**:
  - `timed(f)`: A decorator to measure the execution time of a function.
  - `get_all_quads(tri)`: Generates all possible unique quads from Delaunay triangles.
  - `count_hits(given_pts, x_offset, y_offset)`: Counts matching points in a unity grid with given offsets.

- **`detect.py`**:
  - `detect_chessboard_and_positions(image_path)`: Detects the chessboard and its positions from an image.

- **`model.py`**:
  - `CNNModel(filter_sizes)`: Defines a CNN model for classification tasks.

- **`saddle_points.py`**:
  - `get_saddle(gray_img)`: Calculates saddle points and gradients of a grayscale image.
  - `fast_nonmax_suppression(img, win=11)`: Performs fast non-maximum suppression using dilation.
  - `clip_bounding_points(pts, img_shape, win_size=10)`: Clips points near the image boundaries.
  - `get_final_saddle_points(img, win_size=10)`: Gets the final saddle points from an image.

- **`train.ipynb`**:
  - Contains the training script for the CNN model.

- **`utils.py`**:
  - `pred_patches(patches)`: Predicts patches using the pre-trained model.
  - `predict_fn(inputs)`: Predicts probabilities for tiles.
  - `classify_image(frame, win_size=10, prob_threshold=0.5)`: Classifies an image based on saddle points.
  - `predict_on_tiles(tiles, predict_fn)`: Predicts on a batch of tiles.
  - `predict_on_image(pts, img, predict_fn, win_size=10)`: Predicts probabilities for points in an image.


### Configuration

Configuration settings are typically managed through environment variables or configuration files. Ensure the following environment variables are set:

- `STOCKFISH_PATH`: Path to the Stockfish executable.
- `MODEL_PATH`: Path to the pre-trained model files (`patch_model_v2.pt` and `model.pt`).



