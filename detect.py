import cv2
import numpy as np


def detect_chessboard_and_positions(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the chessboard size
    chessboard_size = (7, 7)  # Corners for 8x8 squares in a chessboard

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if not ret:
        print("Chessboard not detected.")
        return None

    # Refine corner positions for better accuracy
    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria=(cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.1),
    )

    # Draw the chessboard corners (optional, for visualization)
    cv2.drawChessboardCorners(image, chessboard_size, corners, ret)

    # Estimate the grid positions based on the detected corners
    board_positions = []
    square_size = int(
        (corners[7][0][0] - corners[0][0][0]) / 7
    )  # Approximate square size

    # Loop through 8x8 grid positions
    for i in range(8):
        row = []
        for j in range(8):
            # Calculate the top-left corner of each square
            top_left_x = int(corners[0][0][0] + j * square_size)
            top_left_y = int(corners[0][0][1] + i * square_size)
            bottom_right_x = top_left_x + square_size
            bottom_right_y = top_left_y + square_size

            # Extract the region of interest (ROI) for each square
            roi = gray[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            # Check if square is filled by detecting if the ROI is mostly empty or not
            # Threshold the ROI to detect presence of a piece
            _, binary_roi = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY_INV)
            filled_ratio = cv2.countNonZero(binary_roi) / (square_size * square_size)

            # Mark as filled if it exceeds a threshold (e.g., 10% of the square is non-zero)
            is_filled = filled_ratio > 0.1

            # Append the result for this position
            row.append(is_filled)

            # Draw a rectangle for visualization (optional)
            color = (0, 255, 0) if is_filled else (0, 0, 255)
            cv2.rectangle(
                image,
                (top_left_x, top_left_y),
                (bottom_right_x, bottom_right_y),
                color,
                2,
            )

        board_positions.append(row)

    # Display the result
    cv2.imshow("Chessboard Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return the 8x8 board positions grid
    return board_positions


# Usage
image_path = "images/2024-11-26-154227.jpg"
positions = detect_chessboard_and_positions(image_path)

# Print board positions grid
if positions:
    for row in positions:
        print(row)
        print(row)
