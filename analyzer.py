import numpy as np
import copy
from PIL import Image
from scipy.ndimage import rotate
from scipy.signal import find_peaks

class PepperPotAnalyzer:
    """Class for analyzing pepper-pot images and calculating emittance"""

    def __init__(self):
        # Default parameters
        self.scaling = 57.87 / 1000  # Default scaling factor (mm/pixel)
        self.offset = 0  # Default offset
        self.distance = 41  # Default L value (distance from pepper-pot to screen in mm)
        self.threshold = 0.2  # Default intensity threshold
        self.rotation_angle = 0  # Default rotation angle
        self.alpha = 0.57  # Default peak detection sensitivity
        self.xhole_positions = []  # X hole positions
        self.yhole_positions = []  # Y hole positions
        self.peak_size = 10  # Default peak size
        self.intensity_scale = 1.0  # Default intensity scale factor (new parameter)

        # Image data
        self.raw_image = None
        self.background_image = None
        self.processed_image = None
        self.cropped_image = None
        self.x_profile = None
        self.y_profile = None

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

    def get_peaks(self, image, sigma, alpha=0.57, size=10):
        """Find peaks in image based on intensity thresholds"""
        if image is None:
            return [], []

        i_out = []
        j_out = []
        image_temp = copy.deepcopy(image)

        while True:
            k = np.argmax(image_temp)
            j, i = np.unravel_index(k, image_temp.shape)

            if image_temp[j, i] >= alpha * sigma:
                i_out.append(i)
                j_out.append(j)

                # Create a mask around the peak
                x = np.arange(i - size, i + size)
                y = np.arange(j - size, j + size)
                xv, yv = np.meshgrid(x, y)

                # Zero out the area around the peak to find the next one
                image_temp[yv.clip(0, image_temp.shape[0] - 1),
                xv.clip(0, image_temp.shape[1] - 1)] = 0
            else:
                break

        return i_out, j_out

    def radial_profile(self, data, center):
        """Calculate radial profile around a point"""
        y, x = np.indices((data.shape))
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        r = r.astype(np.int64)

        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())

        # Avoid division by zero
        nr = np.where(nr == 0, 1, nr)

        radialprofile = tbin / nr
        return radialprofile

    def analyze_holes(self, image, hole_coordinates, threshold=0.2, peak_size=10):
        """Analyze each hole in the image"""
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

            # Calculate radial profile to detect hole boundaries
            radial_prof = self.radial_profile(image, (x_hole, y_hole))

            try:
                # Find first minimum in radial profile (hole boundary)
                peaks, _ = find_peaks(-radial_prof, height=-1000, prominence=1)

                if len(peaks) > 0:
                    dx = peaks[0]
                    dy = peaks[0]

                    # Refine boundary box
                    x_min = max(0, x_hole - dx)
                    x_max = min(image.shape[1] - 1, x_hole + dx)
                    y_min = max(0, y_hole - dy)
                    y_max = min(image.shape[0] - 1, y_hole + dy)

                    # Extract refined region
                    img_region = image[y_min:y_max, x_min:x_max]

                    # Calculate X and Y profiles of region
                    x_profile = np.sum(img_region, axis=0)
                    y_profile = np.sum(img_region, axis=1)

                    # Find peaks in profiles
                    peaks_x, _ = find_peaks(x_profile, height=0, prominence=10)
                    peaks_y, _ = find_peaks(y_profile, height=0, prominence=10)

                    # Calculate peak widths based on threshold
                    if len(peaks_x) > 0 and len(peaks_y) > 0:
                        # Find width of peak where intensity falls below threshold * max
                        peak_width_x = self.find_peak_width(x_profile, peaks_x[0], threshold)
                        peak_width_y = self.find_peak_width(y_profile, peaks_y[0], threshold)

                        # Update hole boundaries based on width
                        left_x, right_x = peak_width_x
                        left_y, right_y = peak_width_y

                        # Final boundaries
                        final_x_min = max(0, x_hole - dx + left_x)
                        final_x_max = min(image.shape[1] - 1, x_hole - dx + right_x)
                        final_y_min = max(0, y_hole - dy + left_y)
                        final_y_max = min(image.shape[0] - 1, y_hole - dy + right_y)

                        # Final region
                        final_region = image[final_y_min:final_y_max, final_x_min:final_x_max]

                        # Store results
                        clean_hole_sizes.append([final_y_min, final_y_max, final_x_min, final_x_max])
                        clean_hole_data.append(final_region)
            except Exception as e:
                print(f"Error analyzing hole {hole_idx}: {e}")

        return clean_hole_data, clean_hole_sizes

    def find_peak_width(self, profile, peak_idx, threshold):
        """Find width of peak at threshold * max intensity"""
        peak_height = profile[peak_idx]
        threshold_value = peak_height * threshold

        # Find left boundary
        left_idx = peak_idx
        while left_idx > 0 and profile[left_idx] > threshold_value:
            left_idx -= 1

        # Find right boundary
        right_idx = peak_idx
        while right_idx < len(profile) - 1 and profile[right_idx] > threshold_value:
            right_idx += 1

        return left_idx, right_idx

    def calculate_emittance(self, hole_data, hole_sizes, x0_positions, y0_positions):
        """Calculate beam emittance using the pepper-pot method"""
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

            # Store coordinate ranges and profiles
            Xi_range_list.append(x_range)
            Pi_Xrange_list.append(x_profile)
            Yi_range_list.append(y_range)
            Pi_Yrange_list.append(y_profile)

        # Merge all ranges and profiles
        Xi_merge = np.concatenate(Xi_range_list)
        Pi_Xmerge = np.concatenate(Pi_Xrange_list)
        Yi_merge = np.concatenate(Yi_range_list)
        Pi_Ymerge = np.concatenate(Pi_Yrange_list)

        # Calculate <X> and <Y>
        x_bar = np.sum(Xi_merge * Pi_Xmerge) / np.sum(Pi_Xmerge)
        y_bar = np.sum(Yi_merge * Pi_Ymerge) / np.sum(Pi_Ymerge)

        # Calculate X' and Y' (angular divergence)
        Xpi_list = []
        Ypi_list = []
        XO_merge_list = []
        YO_merge_list = []

        # Calculate divergence for each hole
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
