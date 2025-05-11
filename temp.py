import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load MRI and Mask images
def load_images(mri_path, mask_path):
    mri = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return mri, mask

# Preprocessing
def preprocess_image(img):
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # Normalize to 0-255
    norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    return norm

# Apply segmentation using thresholding and morphology
def segment_tumor(preprocessed_img):
    _, thresh = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closing

# Visualize results
def visualize(mri, mask, result):
    plt.figure(figsize=(12, 4))
    titles = ['Original MRI', 'Ground Truth Mask', 'Segmented Tumor']
    images = [mri, mask, result]

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    mri_path = "C:/Users/aswin/Desktop/react_project/ticket/src/Component/brain.jpg"
    mask_path = "C:/Users/aswin/Desktop/react_project/ticket/src/Component/mask.png"

    mri, mask = load_images(mri_path, mask_path)
    preprocessed = preprocess_image(mri)
    result = segment_tumor(preprocessed)
    visualize(mri, mask, result)
