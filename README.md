# Fuzzy C-Means Clustering GUI

Course project for CE672: Machine Processing of Remotely Sensed Data, carried out in semester 2024-25-II at IIT Kanpur. The .py file is a Python-based GUI application for performing image segmentation using Fuzzy C-Means (FCM) clustering with selectable distance metrics (Euclidean or Mahalanobis). Designed to support both RGB and multispectral images, the tool visualizes segmentation results side-by-side with the original image.

## Features

- GUI built with Tkinter and Matplotlib
- Support for `.jpg`, `.png`, and `.tif` image formats
- Implements:
  - Standard FCM (Euclidean distance)
  - Enhanced FCM (Mahalanobis distance with optional PCA)
- Optionally includes spatial coordinates as features
- PCA-based dimensionality reduction for multispectral data
- Visual comparison of original and clustered images
