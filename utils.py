
"""Utility functions for image conversion and preprocessing."""

def convert(img):
    """Convert grayscale image to 3-channel RGB-like image."""
    new_img = []
    for row in img:
        t = []
        for val in row:
            t.append([val, val, val])
        new_img.append(t)
    return new_img

def rgb2gray(rgb):
    """Convert RGB image to grayscale."""
    try:
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    except (IndexError, TypeError):
        return rgb
