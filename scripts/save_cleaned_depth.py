"""
save_cleaned_depth.py

This script loads depth data from the "depth_uncover_cleaned_0to102.npy" file,
processes the uncovered depth data by splitting it into training, validation, and test sets
(using a 60:20:20 subject split), applies the necessary rotation and intensity normalization,
visualizes the results with both OpenCV and Matplotlib, and finally saves the processed arrays,
plots, and the global scaling factor as a text file -- all in the cleaned_depth_images directory.

Directory structure expected:
  Rootdir
  ├── depth_uncover_cleaned_0to102.npy
  ├── save_cleaned_depth.py
  └── cleaned_depth_images  
             └── x_ttv.npz  (created by this script)
"""

import os
import numpy as np
import cv2
import skimage.exposure
import matplotlib.pyplot as plt

def load_uncover_depth():
    """
    Load the uncovered depth data.
    
    Returns:
        uncover_depth (ndarray): Uncovered depth data.
    """
    uncover_depth = np.load("depth_uncover_cleaned_0to102.npy")
    return uncover_depth

def compute_global_max(uncover_depth):
    """
    Compute the global maximum value across all uncovered images.
    
    Args:
        uncover_depth (ndarray): Uncovered depth data.
    
    Returns:
        max_val (float): Global maximum value.
    """
    max_val = np.max(uncover_depth)
    return max_val

def process_uncover_depth(uncover_depth, split_indices):
    """
    Process the uncovered depth data by splitting subjects and applying a rotation.
    The data is reshaped to flatten the subject axis and a channel dimension is added.
    
    Args:
        uncover_depth (ndarray): The raw uncovered depth data.
        split_indices (dict): Dictionary with keys 'train', 'val', 'test'
                              and slice objects indicating subject ranges.
    
    Returns:
        tuple: Processed training, validation, and test arrays.
    """
    # Reshape: (num_subjects, num_images_per_subject, H, W) -> (-1, H, W)
    # Then add a channel dimension and rotate the images 270° (k=3).
    train = np.rot90(
        np.expand_dims(uncover_depth[split_indices['train']].reshape(-1,
            uncover_depth.shape[2], uncover_depth.shape[3]), axis=3),
        k=3, axes=(1,2)
    )
    val = np.rot90(
        np.expand_dims(uncover_depth[split_indices['val']].reshape(-1,
            uncover_depth.shape[2], uncover_depth.shape[3]), axis=3),
        k=3, axes=(1,2)
    )
    test = np.rot90(
        np.expand_dims(uncover_depth[split_indices['test']].reshape(-1,
            uncover_depth.shape[2], uncover_depth.shape[3]), axis=3),
        k=3, axes=(1,2)
    )
    return train, val, test

def visualize_cv2(sample_image, output_dir, output_prefix="dist_img"):
    """
    Visualize a depth image using OpenCV by resizing, intensity stretching,
    applying a color lookup table (LUT), and then saving and displaying the results.
    
    Args:
        sample_image (ndarray): 2D depth image (grayscale) for visualization.
        output_dir (str): Directory to save output images.
        output_prefix (str): Prefix for the saved output files.
    """
    print("Original image shape for CV2 visualization:", sample_image.shape)
    img = sample_image.copy()
    
    # Compute new size. cv2.resize expects (width, height).
    new_width = img.shape[1] * 3
    new_height = img.shape[0] * 12
    print("Resizing image to (width, height):", new_width, new_height)
    img_resized = cv2.resize(img, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Stretch intensity to full dynamic range [0, 255].
    stretch = skimage.exposure.rescale_intensity(img_resized, in_range='image', out_range=(0, 255)).astype(np.uint8)
    
    # Convert grayscale image to a 3-channel image.
    stretch_color = cv2.merge([stretch, stretch, stretch])
    
    # Define 6 colors for the LUT: red, orange, yellow, cyan, blue, violet.
    color1 = (0, 0, 255)     # Red
    color2 = (0, 165, 255)   # Orange
    color3 = (0, 255, 255)   # Yellow
    color4 = (255, 255, 0)   # Cyan
    color5 = (255, 0, 0)     # Blue
    color6 = (128, 64, 64)   # Violet
    colorArr = np.array([[color1, color2, color3, color4, color5, color6]], dtype=np.uint8)
    
    # Resize LUT to 256 values.
    lut = cv2.resize(colorArr, (256, 1), interpolation=cv2.INTER_LINEAR)
    
    # Apply LUT to the stretched image.
    result = cv2.LUT(stretch_color, lut)
    
    # Create a gradient image to visualize the LUT.
    grad = np.linspace(0, 255, 512, dtype=np.uint8)
    grad = np.tile(grad, (20, 1))
    grad = cv2.merge([grad, grad, grad])
    grad_colored = cv2.LUT(grad, lut)
    
    # Save the images.
    cv2.imwrite(os.path.join(output_dir, f'{output_prefix}_colorized.png'), result)
    cv2.imwrite(os.path.join(output_dir, f'{output_prefix}_lut.png'), grad_colored)
    
    # Display the images.
    cv2.imshow('RESULT', result)
    cv2.imshow('LUT', grad_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_matplotlib(image, output_dir, output_file="sample.png", threshold=0.7123):
    """
    Visualize a normalized depth image using Matplotlib. Applies a threshold
    to set low-intensity values to zero and uses the 'YlOrRd' colormap.
    
    Args:
        image (ndarray): Normalized depth image with shape (H, W, 1).
        output_dir (str): Directory to save the output image.
        output_file (str): Filename for the saved visualization.
        threshold (float): Threshold value below which pixels are set to zero.
    """
    sample = np.where(image < threshold, 0, image)[:, :, 0]
    fig = plt.figure(dpi=100)
    plt.imshow(sample, cmap='YlOrRd')
    plt.axis('off')
    full_output_path = os.path.join(output_dir, output_file)
    fig.savefig(full_output_path)
    plt.show()
    print("Matplotlib visualization saved as:", full_output_path)

def save_global_scaling_factor(output_dir, max_val):
    """
    Save the global scaling factor to a text file in the output directory.
    
    Args:
        output_dir (str): Directory to save the scaling factor text file.
        max_val (float): Global maximum value.
    """
    scaling_file_path = os.path.join(output_dir, "Scaling_factors_global.txt")
    with open(scaling_file_path, "w") as f:
        f.write("\n")
        f.write("depth arr scaling factor = " + str(max_val))
    print("Global scaling factor saved to:", scaling_file_path)

def main():
    """
    Main function to load, process, visualize, and save the uncovered depth data.
    Splits the subjects (102 total) into training (first 61 subjects), 
    validation (next 20 subjects), and test (remaining 21 subjects) sets
    based on a 60:20:20 split.
    """
    # Create the output directory.
    output_dir = "cleaned_depth_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the uncovered depth data.
    uncover_depth = load_uncover_depth()
    
    # Compute the global maximum value (before splitting).
    max_val = compute_global_max(uncover_depth)
    
    # Save the global scaling factor.
    save_global_scaling_factor(output_dir, max_val)
    
    # Define subject indices for a 60:20:20 split.
    # Total subjects = 102: Train = subjects 0-60 (61 subjects),
    # Val = subjects 61-80 (20 subjects), Test = subjects 81-101 (21 subjects).
    train_index = round(uncover_depth.shape[0] * 0.6)
    val_index = train_index + round(uncover_depth.shape[0] * 0.2)
    test_index = uncover_depth.shape[0]
    
    split_indices = {
        'train': slice(0, train_index),
        'val': slice(train_index, val_index),
        'test': slice(val_index, test_index)
    }
    
    # Process the uncovered depth data.
    x_train, x_val, x_test = process_uncover_depth(uncover_depth, split_indices)
    
    # Print shapes and intensity range before normalization.
    print("Before normalization (uncovered depth data):")
    print("x_train shape:", x_train.shape, "min:", np.min(x_train), "max:", np.max(x_train))
    print("x_val shape:", x_val.shape, "min:", np.min(x_val), "max:", np.max(x_val))
    print("x_test shape:", x_test.shape, "min:", np.min(x_test), "max:", np.max(x_test))
    
    # Normalize the uncovered depth arrays using the global maximum.
    x_train = x_train / max_val
    x_val = x_val / max_val
    x_test = x_test / max_val
    
    print("\nAfter normalization (values in [0, 1]):")
    print("x_train shape:", x_train.shape, "min:", np.min(x_train), "max:", np.max(x_train))
    print("x_val shape:", x_val.shape, "min:", np.min(x_val), "max:", np.max(x_val))
    print("x_test shape:", x_test.shape, "min:", np.min(x_test), "max:", np.max(x_test))
    
    # Visualization using OpenCV.
    # Recompute the training images (without normalization) to extract a sample.
    x_train_original = np.rot90(
        np.expand_dims(uncover_depth[split_indices['train']].reshape(-1,
            uncover_depth.shape[2], uncover_depth.shape[3]), axis=3),
        k=3, axes=(1,2)
    )
    sample_index = 410  # Example index within the training set.
    sample_cv2 = np.where(x_train_original[sample_index] < 1800, 0, x_train_original[sample_index])[:, :, 0]
    visualize_cv2(sample_cv2, output_dir, output_prefix="dist_img")
    
    # Visualization using Matplotlib (with normalized data).
    visualize_matplotlib(x_train[sample_index], output_dir, output_file="sample.png", threshold=0.7123)
    
    # Save the normalized uncovered depth arrays as a compressed .npz file.
    output_path = os.path.join(output_dir, "x_ttv.npz")
    np.savez_compressed(output_path, x_train=x_train, x_val=x_val, x_test=x_test)
    print("\nSaved processed arrays to:", output_path)

if __name__ == "__main__":
    main()
