import numpy as np
import copy
from PIL import Image
from scipy.ndimage import rotate, median_filter, gaussian_filter
from scipy.signal import find_peaks
from scipy.optimize import minimize
import random


class PepperPotAnalyzer:
    """Class for analyzing pepper-pot images and calculating emittance using PAT-style algorithms"""

    def __init__(self):
        # Default parameters (updated to match PAT defaults)
        self.scaling = 0.048807  # Default scaling factor (mm/pixel) from PAT config
        self.offset = 0  # Default offset
        self.distance = 41  # Default L value (distance from pepper-pot to screen in mm) from PAT
        self.hole_diameter = 0.1  # Hole diameter in mm (from PAT)
        self.hole_space = 1.0  # Hole spacing in mm (from PAT)
        self.threshold = 0.2  # Default intensity threshold
        self.rotation_angle = 0  # Default rotation angle
        self.alpha = 0.57  # Default peak detection sensitivity
        self.xhole_positions = []  # X hole positions
        self.yhole_positions = []  # Y hole positions
        self.peak_size = 10  # Default peak size
        self.intensity_scale = 1.0  # Default intensity scale factor
        self.spot_intensitymin = 10  # Minimum intensity for spot detection (from PAT)
        self.spot_areamin = 5  # Minimum area for valid spots (from PAT)

        # Image data
        self.raw_image = None
        self.background_image = None
        self.processed_image = None
        self.cropped_image = None
        self.x_profile = None
        self.y_profile = None
        self.signal_mask = None  # Added: Binary mask for signal identification
        self.denoised_image = None  # Added: Denoised image
        self.fit_image = None  # Added: Curve-fitted image
        self.combined_image = None  # Added: Combined image

        # Results
        self.hole_coordinates = []
        self.clean_hole_data = []
        self.clean_hole_sizes = []
        self.emittance_results = {}

    def load_image(self, filepath):
        """Load an image from filepath"""
        try:
            img = Image.open(filepath)
            return np.array(img.convert('I'))
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def subtract_background(self, image, background):
        """Subtract background from image"""
        if image is None or background is None:
            return None

        subtracted = image - background

        # Clean negative values
        subtracted[subtracted < 0] = 0

        return subtracted

    def rotate_image(self, image, angle):
        """Rotate image by given angle"""
        if image is None:
            return None

        return rotate(image, angle=angle, reshape=False)

    def crop_image(self, image, x_min, x_max, y_min, y_max):
        """Crop image to given boundaries"""
        if image is None:
            return None

        # Ensure coordinates are within image bounds
        height, width = image.shape
        x_min = max(0, min(x_min, width - 1))
        x_max = max(0, min(x_max, width))
        y_min = max(0, min(y_min, height - 1))
        y_max = max(0, min(y_max, height))

        return image[y_min:y_max, x_min:x_max]

    def calculate_profiles(self, image):
        """Calculate X and Y intensity profiles"""
        if image is None:
            return None, None

        x_profile = np.sum(image, axis=0)
        y_profile = np.sum(image, axis=1)

        return x_profile, y_profile

    def get_std(self, image):
        """Calculate standard deviation of image"""
        if image is None:
            return 0

        return np.std(image)

    def get_background_level(self, image):
        """Calculate background level using edge pixels (PAT method)"""
        if image is None:
            return 0

        height, width = image.shape
        edge_pixels = []

        # Add top and bottom rows
        edge_pixels.extend(image[0, :])
        edge_pixels.extend(image[-1, :])

        # Add left and right columns (excluding corners)
        edge_pixels.extend(image[1:-1, 0])
        edge_pixels.extend(image[1:-1, -1])

        # Calculate average of edge pixels
        return np.mean(edge_pixels)

    # PAT-style median filtering
    def apply_median_filter(self, image):
        """Apply median filter to reduce salt-and-pepper noise (PAT step 1)"""
        if image is None:
            return None

        return median_filter(image, size=3)

    # PAT-style mean filtering
    def apply_mean_filter(self, image):
        """Apply mean filter for smoothing (PAT step 2)"""
        if image is None:
            return None

        return gaussian_filter(image, sigma=1)

    # PAT-style signal marking
    def signal_mark(self, image):
        """Mark signal parts of the image (PAT method)"""
        if image is None:
            return None, None

        # 1. Detect background
        background = self.get_background_level(image)

        # 2. Apply median filtering
        median_filtered = self.apply_median_filter(image)

        # 3. Apply mean filtering
        smoothed = self.apply_mean_filter(median_filtered)

        # 4. Create signal mask based on intensity threshold
        mask = np.zeros_like(smoothed, dtype=bool)
        mask[smoothed > (background + self.spot_intensitymin)] = True

        # 5. Apply area filtering (remove small spots)
        from scipy import ndimage
        labeled_mask, num_features = ndimage.label(mask)

        for i in range(1, num_features + 1):
            area = np.sum(labeled_mask == i)
            if area < self.spot_areamin:
                mask[labeled_mask == i] = False

        return mask, smoothed

    # PAT-style Gaussian curve fitting
    def gaussian_function(self, params, x, y):
        """2D Gaussian function for curve fitting"""
        A, x0, y0, sigma_x, sigma_y = params
        return A * np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))

    def fit_gaussian(self, region):
        """Fit a 2D Gaussian to a region (PAT-style curve fitting)"""
        if region is None or np.sum(region) == 0:
            return None, None

        height, width = region.shape
        y, x = np.mgrid[0:height, 0:width]
        x = x.flatten()
        y = y.flatten()
        z = region.flatten()

        # Initial guess
        max_idx = np.argmax(region)
        y_max, x_max = np.unravel_index(max_idx, region.shape)
        max_val = region[y_max, x_max]

        initial_guess = [max_val, x_max, y_max, width / 10, height / 10]

        # Define error function to minimize
        def error_function(params):
            model = self.gaussian_function(params, x, y)
            return np.sum((z - model) ** 2)

        # Calculate "fitting rate" (Overlap/Combination)
        def fitting_rate(params):
            model = np.zeros_like(region)
            for i in range(height):
                for j in range(width):
                    model[i, j] = self.gaussian_function(params, j, i)
                    if model[i, j] > 1:
                        model[i, j] = 1

            overlap = 0
            combination = 0

            for i in range(height):
                for j in range(width):
                    if region[i, j] > 0:
                        if model[i, j] < region[i, j]:
                            overlap += model[i, j]
                            combination += region[i, j]
                        else:
                            overlap += region[i, j]
                            combination += model[i, j]
                    else:
                        combination += model[i, j]

            if combination == 0:
                return 0
            return -overlap / combination  # Negative because we're minimizing

        try:
            # Use PAT's fitting rate instead of sum of squares
            result = minimize(fitting_rate, initial_guess, method='Nelder-Mead')
            fitted_params = result.x

            # Create fitted Gaussian
            fitted_data = np.zeros_like(region)
            for i in range(height):
                for j in range(width):
                    fitted_data[i, j] = self.gaussian_function(fitted_params, j, i)

            return fitted_data, fitted_params
        except:
            return None, None

    def denoise_image(self, image, signal_mask):
        """Denoise image based on signal mask (PAT method)"""
        if image is None or signal_mask is None:
            return None

        denoised = np.zeros_like(image)
        denoised[signal_mask] = image[signal_mask]
        return denoised

    def curve_fit_image(self, image, signal_mask):
        """Apply curve fitting to the image (PAT method)"""
        if image is None or signal_mask is None:
            return None

        # Label connected components
        from scipy import ndimage
        labeled_mask, num_spots = ndimage.label(signal_mask)

        fitted_image = np.zeros_like(image, dtype=float)

        # Process each spot
        for spot_idx in range(1, num_spots + 1):
            # Extract spot region
            spot_mask = (labeled_mask == spot_idx)
            y_indices, x_indices = np.where(spot_mask)

            if len(x_indices) == 0 or len(y_indices) == 0:
                continue

            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            # Extract region with padding
            pad = 5
            x_min_pad = max(0, x_min - pad)
            x_max_pad = min(image.shape[1] - 1, x_max + pad)
            y_min_pad = max(0, y_min - pad)
            y_max_pad = min(image.shape[0] - 1, y_max + pad)

            spot_region = image[y_min_pad:y_max_pad + 1, x_min_pad:x_max_pad + 1].copy()
            spot_mask_region = spot_mask[y_min_pad:y_max_pad + 1, x_min_pad:x_max_pad + 1]

            # Set non-signal pixels to zero for fitting
            spot_region[~spot_mask_region] = 0

            # Fit Gaussian
            fitted_region, params = self.fit_gaussian(spot_region)

            if fitted_region is not None:
                fitted_image[y_min_pad:y_max_pad + 1, x_min_pad:x_max_pad + 1] = fitted_region

        return fitted_image

    def combine_data(self, image, denoised_image, fitted_image, signal_mask):
        """Combine original and fitted data (PAT method)"""
        if image is None or denoised_image is None or fitted_image is None:
            return None

        combined = np.zeros_like(image, dtype=float)

        # Use original data for signal pixels, fitted data elsewhere
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if signal_mask[i, j]:
                    # If pixel is saturated, use fitted value
                    if image[i, j] >= 255:
                        combined[i, j] = fitted_image[i, j]
                    else:
                        combined[i, j] = image[i, j]
                else:
                    combined[i, j] = fitted_image[i, j]

        return combined

    def preprocess_image(self, image):
        """Full preprocessing pipeline (PAT workflow)"""
        if image is None:
            return None, None, None, None

        # 1. Signal mark
        signal_mask, smoothed = self.signal_mark(image)

        # 2. Denoise
        denoised = self.denoise_image(image, signal_mask)

        # 3. Curve fit
        fitted = self.curve_fit_image(image, signal_mask)

        # 4. Combine data
        combined = self.combine_data(image, denoised, fitted, signal_mask)

        return signal_mask, denoised, fitted, combined

    def analyze_holes(self, image, hole_coordinates, threshold=0.2, peak_size=10):
        """Analyze each hole in the image with PAT-style approach"""
        if image is None or not hole_coordinates:
            return [], []

        clean_hole_data = []
        clean_hole_sizes = []

        for hole_idx, (x_hole, y_hole) in enumerate(hole_coordinates):
            # Initial boundary box around hole
            dx = peak_size
            dy = peak_size

            x_min = max(0, x_hole - dx)
            x_max = min(image.shape[1] - 1, x_hole + dx)
            y_min = max(0, y_hole - dy)
            y_max = min(image.shape[0] - 1, y_hole + dy)

            # Extract region around hole
            img_region = image[y_min:y_max, x_min:x_max]

            # Apply PAT-style analysis to region
            region_mask, _, region_fitted, region_combined = self.preprocess_image(img_region)

            if region_mask is None or np.sum(region_mask) < self.spot_areamin:
                continue

            # Find boundaries of the spot in the mask
            y_indices, x_indices = np.where(region_mask)

            if len(x_indices) == 0 or len(y_indices) == 0:
                continue

            # Calculate region boundaries
            local_x_min, local_x_max = np.min(x_indices), np.max(x_indices)
            local_y_min, local_y_max = np.min(y_indices), np.max(y_indices)

            # Convert to global coordinates
            final_x_min = x_min + local_x_min
            final_x_max = x_min + local_x_max
            final_y_min = y_min + local_y_min
            final_y_max = y_min + local_y_max

            # Extract final region
            final_region = image[final_y_min:final_y_max + 1, final_x_min:final_x_max + 1]

            # Store results
            clean_hole_sizes.append([final_y_min, final_y_max + 1, final_x_min, final_x_max + 1])
            clean_hole_data.append(final_region)

        return clean_hole_data, clean_hole_sizes

    def calculate_emittance(self, hole_data, hole_sizes, x0_positions, y0_positions):
        """Calculate beam emittance using the pepper-pot method (PAT approach)"""
        if not hole_data or not hole_sizes:
            return {}

        # Initialize lists for calculations
        Xi_range_list = []
        Pi_Xrange_list = []
        Yi_range_list = []
        Pi_Yrange_list = []

        # Process each hole
        for idx, (hole_img, hole_size) in enumerate(zip(hole_data, hole_sizes)):
            # Extract hole coordinates
            y_min, y_max, x_min, x_max = hole_size

            # Create coordinate ranges scaled to mm
            x_range = (np.arange(x_min, x_max) * self.scaling) + self.offset
            y_range = (np.arange(y_min, y_max) * self.scaling) + self.offset

            # Get X and Y profiles
            x_profile = np.sum(hole_img, axis=0)
            y_profile = np.sum(hole_img, axis=1)

            # Ensure same length for x_range and x_profile
            min_len_x = min(len(x_range), len(x_profile))
            x_range = x_range[:min_len_x]
            x_profile = x_profile[:min_len_x]

            # Ensure same length for y_range and y_profile
            min_len_y = min(len(y_range), len(y_profile))
            y_range = y_range[:min_len_y]
            y_profile = y_profile[:min_len_y]

            # Store coordinate ranges and profiles
            Xi_range_list.append(x_range)
            Pi_Xrange_list.append(x_profile)
            Yi_range_list.append(y_range)
            Pi_Yrange_list.append(y_profile)

        # Merge all ranges and profiles
        try:
            Xi_merge = np.concatenate(Xi_range_list)
            Pi_Xmerge = np.concatenate(Pi_Xrange_list)
            Yi_merge = np.concatenate(Yi_range_list)
            Pi_Ymerge = np.concatenate(Pi_Yrange_list)
        except Exception as e:
            print(f"Merge error: {e}")
            return {}

        # Calculate <X> and <Y>
        x_bar = np.sum(Xi_merge * Pi_Xmerge) / np.sum(Pi_Xmerge)
        y_bar = np.sum(Yi_merge * Pi_Ymerge) / np.sum(Pi_Ymerge)

        # Calculate X' and Y' (angular divergence)
        Xpi_list = []
        Ypi_list = []
        XO_merge_list = []
        YO_merge_list = []

        # Calculate divergence for each hole (PAT method)
        for idx, (x_range, y_range) in enumerate(zip(Xi_range_list, Yi_range_list)):
            # Get reference position (x0, y0) for this hole
            x0 = x0_positions[min(idx, len(x0_positions) - 1)]
            y0 = y0_positions[min(idx, len(y0_positions) - 1)]

            # Calculate divergence (convert to mrad)
            x_divergence = ((x_range - x0) / self.distance) * 1000
            y_divergence = ((y_range - y0) / self.distance) * 1000

            Xpi_list.append(x_divergence)
            Ypi_list.append(y_divergence)

            # Create arrays of reference positions
            XO_merge_list.append(np.full_like(x_range, x0))
            YO_merge_list.append(np.full_like(y_range, y0))

        # Merge divergence arrays
        Xpi_merge = np.concatenate(Xpi_list)
        Ypi_merge = np.concatenate(Ypi_list)
        XO_merge = np.concatenate(XO_merge_list)
        YO_merge = np.concatenate(YO_merge_list)

        # Calculate <X'> and <Y'>
        xp_bar = np.sum(Xpi_merge * Pi_Xmerge) / np.sum(Pi_Xmerge)
        yp_bar = np.sum(Ypi_merge * Pi_Ymerge) / np.sum(Pi_Ymerge)

        # Calculate <X²> and <Y²>
        x_bar_sq = np.sum(((Xi_merge - x_bar) ** 2) * Pi_Xmerge) / np.sum(Pi_Xmerge)
        y_bar_sq = np.sum(((Yi_merge - y_bar) ** 2) * Pi_Ymerge) / np.sum(Pi_Ymerge)

        # Calculate <X'²> and <Y'²>
        Xpi_sq = np.sum(((Xpi_merge - xp_bar) ** 2) * Pi_Xmerge) / np.sum(Pi_Xmerge)
        Ypi_sq = np.sum(((Ypi_merge - yp_bar) ** 2) * Pi_Ymerge) / np.sum(Pi_Ymerge)

        # Calculate <XX'> and <YY'>
        xxp = np.sum(((Xi_merge - x_bar) * (Xpi_merge - xp_bar) * Pi_Xmerge)) / np.sum(Pi_Xmerge)
        yyp = np.sum(((Yi_merge - y_bar) * (Ypi_merge - yp_bar) * Pi_Ymerge)) / np.sum(Pi_Ymerge)

        # Calculate emittance (ε²)
        emit_x_sq = (x_bar_sq * Xpi_sq) - (xxp ** 2)
        emit_y_sq = (y_bar_sq * Ypi_sq) - (yyp ** 2)

        # Calculate RMS values
        x_rms = np.sqrt(np.sum(((Xi_merge - x_bar) ** 2) * Pi_Xmerge) / np.sum(Pi_Xmerge))
        y_rms = np.sqrt(np.sum(((Yi_merge - y_bar) ** 2) * Pi_Ymerge) / np.sum(Pi_Ymerge))

        # Store all results
        results = {
            'x_bar': x_bar,
            'y_bar': y_bar,
            'xp_bar': xp_bar,
            'yp_bar': yp_bar,
            'x_bar_sq': x_bar_sq,
            'y_bar_sq': y_bar_sq,
            'Xpi_sq': Xpi_sq,
            'Ypi_sq': Ypi_sq,
            'xxp': xxp,
            'yyp': yyp,
            'emit_x_sq': emit_x_sq,
            'emit_y_sq': emit_y_sq,
            'emit_x': np.sqrt(emit_x_sq),
            'emit_y': np.sqrt(emit_y_sq),
            'x_rms': x_rms,
            'y_rms': y_rms,
            'Xi_merge': Xi_merge,
            'Yi_merge': Yi_merge,
            'Pi_Xmerge': Pi_Xmerge,
            'Pi_Ymerge': Pi_Ymerge,
            'Xpi_merge': Xpi_merge,
            'Ypi_merge': Ypi_merge,
            'XO_merge': XO_merge,
            'YO_merge': YO_merge
        }

        self.emittance_results = results
        return results

    # PAT-style Monte Carlo method for particle generation
    def generate_particles(self, num_particles=10000):
        """Generate particles using Monte Carlo method as in PAT"""
        if not self.emittance_results:
            return []

        particles = []

        # Get necessary values from emittance results
        Xi_merge = self.emittance_results['Xi_merge']
        Yi_merge = self.emittance_results['Yi_merge']
        Xpi_merge = self.emittance_results['Xpi_merge']
        Ypi_merge = self.emittance_results['Ypi_merge']
        Pi_Xmerge = self.emittance_results['Pi_Xmerge']

        # Normalize weights
        weights = Pi_Xmerge / np.sum(Pi_Xmerge)

        # Generate particles using Monte Carlo sampling
        indices = np.random.choice(len(weights), size=num_particles, p=weights)

        for idx in indices:
            # Add some randomness to hole position (PAT method)
            x_hole = self.hole_space * np.floor(Xi_merge[idx] / self.hole_space + 0.5)
            y_hole = self.hole_space * np.floor(Yi_merge[idx] / self.hole_space + 0.5)

            # Random position inside hole
            x1 = (random.random() - 0.5) * self.hole_diameter
            y1 = (random.random() - 0.5) * self.hole_diameter

            # Particle position equals hole position plus random offset
            x = x_hole + x1
            y = y_hole + y1

            # Angle from position and divergence (PAT calculation)
            xp = Xpi_merge[idx]
            yp = Ypi_merge[idx]

            particles.append({
                'x': x,
                'y': y,
                'xp': xp,
                'yp': yp
            })

        return particles