import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import rasterio
from PIL import Image, ImageTk
import os
from time import time
from sklearn.decomposition import PCA

class FuzzyCMeansEuclidean:
    """
    Simple Fuzzy C-Means clustering using Euclidean distance.
    """
    def __init__(self, n_clusters=3, max_iter=100, m=2, threshold=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.threshold = threshold
        self.centers = None
        self.membership = None
        self.runtime = 0
        self.iterations = 0

    def fit(self, X):
        start_time = time()
        n_samples, n_features = X.shape

        np.random.seed(42)
        U = np.random.rand(n_samples, self.n_clusters)
        U /= U.sum(axis=1, keepdims=True)

        prev_obj = float('inf')

        for iteration in range(self.max_iter):
            U_m = U ** self.m
            centers = (U_m.T @ X) / (U_m.sum(axis=0)[:, np.newaxis] + 1e-8) #

            distances = np.zeros((n_samples, self.n_clusters))
            for j in range(self.n_clusters):
                # Euclidean distance calculation
                distances[:, j] = np.sum((X - centers[j]) ** 2, axis=1)

            distances = np.fmax(distances, np.finfo(float).eps) #
            # Update membership U
            inv_dist_pow = distances ** (-2. / (self.m - 1))
            U = inv_dist_pow / inv_dist_pow.sum(axis=1, keepdims=True)

            obj = np.sum((U ** self.m) * distances) #
            if abs(prev_obj - obj) < self.threshold:
                break
            prev_obj = obj

        self.centers = centers
        self.membership = U
        self.iterations = iteration + 1
        self.runtime = time() - start_time

        return np.argmax(U, axis=1), U

class FuzzyCMeansMahalanobis:
    """
    Fuzzy C-Means clustering with Mahalanobis distance.
    """
    def __init__(self, n_clusters=5, max_iter=100, m=2, threshold=1e-4, use_pca=True, explained_variance_threshold=0.99, lambda_reg=1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.threshold = threshold
        self.use_pca = use_pca
        self.explained_variance_threshold = explained_variance_threshold #
        self.lambda_reg = lambda_reg #
        self.n_components = None
        self.centers = None
        self.membership = None
        self.runtime = 0
        self.iterations = 0
        self.cov_matrices = None
        self.alpha = None
        self.pca_model = None # Store PCA model if used

    def fit(self, X_orig):
        start_time = time()
        X = X_orig.copy() # Work on a copy

        if self.use_pca:
            # Determining n_components based on explained variance
            pca = PCA(n_components=min(X.shape))
            pca.fit(X)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            self.n_components = np.argmax(cumulative_variance >= self.explained_variance_threshold) + 1
            print(f"Using PCA: {self.n_components} components explain >= {self.explained_variance_threshold * 100:.1f}% variance.")
            self.pca_model = PCA(n_components=self.n_components)
            X = self.pca_model.fit_transform(X)
        else:
             print("PCA is not used.")
             self.n_components = X.shape[1] # Using original features


        # Normalization
        self.data_max = np.max(X, axis=0) + 1e-8
        X = X / self.data_max


        n_samples, n_features = X.shape
        np.random.seed(42)
        U = np.random.rand(n_samples, self.n_clusters)
        U = U / U.sum(axis=1, keepdims=True) #
        prev_obj = float('inf')

        for iteration in range(self.max_iter):
            U_m = U ** self.m
            centers = (U_m.T @ X) / (U_m.sum(axis=0)[:, np.newaxis] + 1e-8) #

            # Regularized covariance matrices
            cov_matrices = []
            for j in range(self.n_clusters):
                diff = X - centers[j]
                weighted_diff = U_m[:, j][:, None] * diff
                cov = (weighted_diff.T @ diff) / (U_m[:, j].sum() + 1e-8)
                # Regularization to ensure invertibility
                cov += np.eye(n_features) * 1e-3 # Adjusted regularization
                # Checking for positive semi-definite matrix, adding more regularization if needed
                min_eig = np.min(np.real(np.linalg.eigvals(cov)))
                if min_eig < 1e-6:
                   cov += np.eye(n_features) * (1e-6 - min_eig)
                cov_matrices.append(cov)

            # Mahalanobis distances
            distances = np.zeros((n_samples, self.n_clusters))
            determinants = np.zeros(self.n_clusters)
            for j in range(self.n_clusters):
                try:
                    inv_cov = np.linalg.inv(cov_matrices[j])
                    det_cov = np.linalg.det(cov_matrices[j])
                    if det_cov <= 1e-10: # Preventing log(0) or negative
                        det_cov = 1e-10
                    determinants[j] = det_cov

                    diff = X - centers[j]
                    distances[:, j] = np.sum(diff @ inv_cov * diff, axis=1) #

                except np.linalg.LinAlgError:
                    # Handling cases where matrix is still singular
                    print(f"Warning: Covariance matrix for cluster {j} is singular at iteration {iteration}. Using Euclidean distance as fallback for this cluster.")
                    # Fallback to Euclidean distance for this cluster
                    distances[:, j] = np.sum((X - centers[j]) ** 2, axis=1)
                    determinants[j] = 1.0 # Assigning a neutral determinant


            # Alpha
            alpha = np.sum(U, axis=0) / n_samples
            alpha = alpha / alpha.sum()


            # Updating membership matrix using Zhao et al. objective function
            distances = np.fmax(distances, np.finfo(float).eps)
            U_new = np.zeros_like(U)

            term1 = -distances / self.lambda_reg
            term2 = -np.log(np.fmax(determinants[None, :], 1e-10)) # Using stored determinants, adding None for broadcasting
            term3 = np.log(np.fmax(alpha[None, :], 1e-10))         # Using stored alpha, adding None for broadcasting

            exp_terms = np.exp(term1 + term2 + term3)

            # Potential overflows/NaNs in exp_terms
            exp_terms[np.isnan(exp_terms)] = 0.0
            exp_terms[np.isinf(exp_terms)] = np.finfo(float).max / self.n_clusters # Avoid inf sum

            row_sums = exp_terms.sum(axis=1, keepdims=True)
            # Division by zero if a row sum is zero
            zero_sum_rows = (row_sums <= 1e-10)
            U_new = exp_terms / (row_sums + zero_sum_rows) # Adding 1 where sum is zero to avoid NaN
            # For rows where sum was zero, assigning equal probability 
            U_new[zero_sum_rows.flatten(), :] = 1.0 / self.n_clusters


             # Objective function value
            obj = np.sum(U * distances) \
                + self.lambda_reg * np.sum(U * np.log(np.fmax(alpha[None, :], 1e-10))) \
                + self.lambda_reg * np.sum(U * np.log(np.fmax(determinants[None, :], 1e-10)))


            # Checking for convergence
            if abs(prev_obj - obj) < self.threshold:
                print(f"Converged after {iteration + 1} iterations.")
                break
            prev_obj = obj
            U = U_new

            if iteration == self.max_iter - 1:
                 print(f"Reached max iterations ({self.max_iter}).")


        self.centers = centers # Stores centers in normalized space
        self.membership = U
        self.iterations = iteration + 1
        self.runtime = time() - start_time
        self.cov_matrices = cov_matrices
        self.alpha = alpha

        return np.argmax(U, axis=1), U

def load_image_data(image_path, include_coords=True):
    """
    Loads image data (RGB or Multispectral) and optionally adds coordinates.
    Returns pixel data array, shape, and displayable image (RGB or first 3 bands).
    (Adapted from load_image and load_multispectral in CE672_Term_Paper_Code.ipynb)
    """
    display_img = None
    file_ext = os.path.splitext(image_path)[1].lower()

    if file_ext in ('.jpg', '.jpeg', '.png'): # RGB image
        img = cv2.imread(image_path)
        if img is None:
             raise ValueError(f"Could not read image file: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #
        display_img = img.copy() # Use for display

    elif file_ext in ('.tif', '.tiff'): # Multispectral image
        try:
            with rasterio.open(image_path) as src:
                img = np.transpose(src.read(), (1, 2, 0)) #
                # Creates a displayable image (first 3 bands or grayscale if fewer)
                if src.count >= 3:
                    display_img_raw = img[:, :, :3].astype(np.float32) #
                    # Normalizes for display if needed (e.g., if values > 255)
                    max_val = np.max(display_img_raw)
                    if max_val > 255.0:
                         display_img_raw = (display_img_raw / max_val * 255.0)
                    display_img = display_img_raw.astype(np.uint8)
                elif src.count >= 1:
                    display_img_raw = img[:, :, 0].astype(np.float32)
                    max_val = np.max(display_img_raw)
                    if max_val > 0: # Avoids division by zero
                        display_img_raw = (display_img_raw / max_val * 255.0)
                    display_img = cv2.cvtColor(display_img_raw.astype(np.uint8), cv2.COLOR_GRAY2RGB) # Convert to RGB for consistency
                else:
                     raise ValueError("TIFF file has no bands.")

        except rasterio.RasterioIOError as e:
            raise ValueError(f"Error reading TIFF file with rasterio: {e}")

    else:
        raise ValueError(f"Unsupported image format: {file_ext}. Supported: jpg, png, tif, tiff")

    if img is None:
        raise ValueError("Failed to load image.")
    if display_img is None:
         raise ValueError("Failed to create displayable image.")


    h, w = img.shape[:2]
    n_features = img.shape[2] if img.ndim == 3 else 1
    X = img.reshape(-1, n_features).astype(np.float32) #

    if include_coords:
        y, x = np.mgrid[0:h, 0:w] #
        # Normalizing coordinates to [0, 1]
        coords = np.stack((x / (w - 1) if w > 1 else 0, y / (h - 1) if h > 1 else 0), axis=-1).reshape(-1, 2)
        X = np.hstack((X, coords)) #
        print(f"Included normalized coordinates. Feature shape: {X.shape}")

    return X, (h, w), display_img

# --- Tkinter GUI Application ---

class FCM_GUI:
    def __init__(self, master):
        self.master = master
        master.title("Fuzzy C-Means Clustering")
        master.geometry("800x600")

        self.image_path = None
        self.original_image_pil = None
        self.clustered_image_pil = None
        self.data = None
        self.shape = None
        self.display_img_orig = None # Store the original image for display

        # --- Top Frame: Controls ---
        control_frame = ttk.Frame(master, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # File Selection
        self.select_button = ttk.Button(control_frame, text="Select Image", command=self.select_image)
        self.select_button.pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(control_frame, text="No image selected")
        self.file_label.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # Distance Metric Selection
        ttk.Label(control_frame, text="Distance:").pack(side=tk.LEFT, padx=(10, 2))
        self.distance_var = tk.StringVar(value="Euclidean")
        distance_options = ["Euclidean", "Mahalanobis"]
        distance_menu = ttk.OptionMenu(control_frame, self.distance_var, distance_options[0], *distance_options)
        distance_menu.pack(side=tk.LEFT, padx=2)

        # Number of Clusters
        ttk.Label(control_frame, text="Clusters:").pack(side=tk.LEFT, padx=(10, 2))
        self.n_clusters_var = tk.IntVar(value=5)
        self.n_clusters_spinbox = ttk.Spinbox(control_frame, from_=2, to=20, textvariable=self.n_clusters_var, width=5)
        self.n_clusters_spinbox.pack(side=tk.LEFT, padx=2)

        # Run Button
        self.run_button = ttk.Button(control_frame, text="Run Clustering", command=self.run_clustering, state=tk.DISABLED)
        self.run_button.pack(side=tk.LEFT, padx=10)

        # --- Bottom Frame: Image Display ---
        display_frame = ttk.Frame(master, padding="10")
        display_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Matplotlib Figure for plotting
        self.fig, self.axs = plt.subplots(1, 2, figsize=(8, 4)) # One row, two columns
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Initial placeholder titles
        self.axs[0].set_title("Original Image")
        self.axs[1].set_title("Clustered Image")
        self.axs[0].axis('off')
        self.axs[1].axis('off')

    def select_image(self):
        """Opens a file dialog to select an image."""
        filetypes = [
            ('Image files', '*.jpg *.jpeg *.png *.tif *.tiff'),
            ('All files', '*.*')
        ]
        filepath = filedialog.askopenfilename(title="Select Image File", filetypes=filetypes)
        if filepath:
            self.image_path = filepath
            self.file_label.config(text=os.path.basename(filepath))
            self.run_button.config(state=tk.NORMAL) # Enables run button
            # Loads and displays the original image immediately
            try:
                # We need coordinates for both methods potentially
                self.data, self.shape, self.display_img_orig = load_image_data(self.image_path, include_coords=True)
                self.display_images(original=self.display_img_orig, clustered=None) # Show original
            except Exception as e:
                messagebox.showerror("Error Loading Image", f"Failed to load image: {e}")
                self.image_path = None
                self.file_label.config(text="No image selected")
                self.run_button.config(state=tk.DISABLED)
                self.data = None
                self.shape = None
                self.display_img_orig = None


    def run_clustering(self):
        """Runs the selected FCM algorithm."""
        if not self.image_path or self.data is None:
            messagebox.showerror("Error", "Please select a valid image first.")
            return

        n_clusters = self.n_clusters_var.get()
        distance_metric = self.distance_var.get()

        # Disables button during processing
        self.run_button.config(state=tk.DISABLED)
        self.master.update_idletasks() # Updates GUI to show disabled state

        try:
            print(f"Starting clustering with {distance_metric} distance, {n_clusters} clusters...")
            start_run_time = time()

            if distance_metric == "Euclidean":
                 # Uses Euclidean FCM - coordinates were already added in load_image_data
                 fcm = FuzzyCMeansEuclidean(n_clusters=n_clusters, max_iter=100, threshold=1e-4) #
                 labels, _ = fcm.fit(self.data) #
            elif distance_metric == "Mahalanobis":
                 # Uses Mahalanobis FCM - coordinates were already added in load_image_data
                 # Decides on PCA based on features (e.g., > 3 features might benefit from PCA)
                 use_pca = self.data.shape[1] > 3 # Enable PCA if more than 3 features (e.g., RGB+coords or multispectral)
                 fcm = FuzzyCMeansMahalanobis(n_clusters=n_clusters, max_iter=100, threshold=1e-4, use_pca=use_pca, explained_variance_threshold=0.99) #
                 labels, _ = fcm.fit(self.data) #
            else:
                raise ValueError("Invalid distance metric selected")

            end_run_time = time()
            print(f"Clustering finished in {end_run_time - start_run_time:.2f} seconds.")

            # Reshapes labels to image dimensions
            clustered_result = labels.reshape(self.shape)

            # Displays results
            self.display_images(original=self.display_img_orig, clustered=clustered_result)

        except Exception as e:
            messagebox.showerror("Clustering Error", f"An error occurred during clustering: {e}")
            # Clears the clustered result display on error
            self.display_images(original=self.display_img_orig, clustered=None)
        finally:
            # Re-enable button
             if self.image_path: 
                 self.run_button.config(state=tk.NORMAL)


    def display_images(self, original, clustered):
        """Updates the matplotlib canvas with original and clustered images."""
        self.axs[0].clear()
        self.axs[1].clear()

        # Displays Original Image
        if original is not None:
            self.axs[0].imshow(original)
            self.axs[0].set_title("Original Image")
        else:
            self.axs[0].set_title("Original Image (None)")
        self.axs[0].axis('off')

        # Displays Clustered Image
        if clustered is not None:
            # Uses a qualitative colormap like 'tab10' or 'viridis'
            cmap = plt.get_cmap('tab10', self.n_clusters_var.get())
            self.axs[1].imshow(clustered, cmap=cmap, interpolation='nearest')
            self.axs[1].set_title(f"Clustered ({self.distance_var.get()})")
        else:
             self.axs[1].set_title("Clustered Image (None)")
        self.axs[1].axis('off')

        self.canvas.draw()

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = FCM_GUI(root)
    root.mainloop()