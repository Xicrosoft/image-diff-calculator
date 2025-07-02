#!/usr/bin/env python3
"""
Create test images for the image difference calculator tests
"""

import cv2
import numpy as np
import os

def create_test_images():
    """Create various test images for testing purposes"""
    test_images_dir = "test_images"

    # Create a solid red image (100x100)
    red_image = np.zeros((100, 100, 3), dtype=np.uint8)
    red_image[:, :] = [0, 0, 255]  # BGR format
    cv2.imwrite(os.path.join(test_images_dir, "red_100x100.png"), red_image)

    # Create a solid blue image (100x100)
    blue_image = np.zeros((100, 100, 3), dtype=np.uint8)
    blue_image[:, :] = [255, 0, 0]  # BGR format
    cv2.imwrite(os.path.join(test_images_dir, "blue_100x100.png"), blue_image)

    # Create an identical copy of red image
    cv2.imwrite(os.path.join(test_images_dir, "red_100x100_copy.png"), red_image)

    # Create a gradient image
    gradient_image = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        gradient_image[i, :] = [i * 2.55, i * 2.55, i * 2.55]
    cv2.imwrite(os.path.join(test_images_dir, "gradient_100x100.png"), gradient_image)

    # Create a checkerboard pattern
    checkerboard = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(0, 100, 10):
        for j in range(0, 100, 10):
            if (i // 10 + j // 10) % 2 == 0:
                checkerboard[i:i+10, j:j+10] = [255, 255, 255]
    cv2.imwrite(os.path.join(test_images_dir, "checkerboard_100x100.png"), checkerboard)

    # Create a different sized red image (50x50)
    red_small = np.zeros((50, 50, 3), dtype=np.uint8)
    red_small[:, :] = [0, 0, 255]
    cv2.imwrite(os.path.join(test_images_dir, "red_50x50.png"), red_small)

    # Create a slightly modified red image (one pixel different)
    red_modified = red_image.copy()
    red_modified[50, 50] = [0, 255, 0]  # One green pixel
    cv2.imwrite(os.path.join(test_images_dir, "red_100x100_modified.png"), red_modified)

    # Create a noisy version of red image
    red_noisy = red_image.copy().astype(np.float32)
    noise = np.random.normal(0, 10, red_noisy.shape)
    red_noisy = np.clip(red_noisy + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(test_images_dir, "red_100x100_noisy.png"), red_noisy)

    print("Test images created successfully in", test_images_dir)

if __name__ == "__main__":
    create_test_images()
