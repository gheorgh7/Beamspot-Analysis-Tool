import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QLabel,
                             QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
                             QFileDialog, QSpinBox, QDoubleSpinBox, QGroupBox, QComboBox,
                             QCheckBox, QLineEdit, QSplitter, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import cv2
from scipy.ndimage import rotate
from scipy.optimize import curve_fit
from scipy import signal
from scipy.ndimage import maximum_filter, label, find_objects
import traceback


class BeamAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

        # Data storage
        self.main_image = None
        self.bg_image = None
        self.processed_image = None
        self.beam_data = None
        self.phase_space_data = None

    def init_ui(self):
        self.setWindowTitle("Beam Spot and Pepper-Pot Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create horizontal splitter for controls and display
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Control panel on the left
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        splitter.addWidget(controls_widget)

        # Image display and results on the right
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        splitter.addWidget(display_widget)

        # Set splitter sizes (30% controls, 70% display)
        splitter.setSizes([300, 700])

        # === Controls Panel ===
        # Image loading group
        image_group = QGroupBox("Image Loading")
        image_layout = QGridLayout(image_group)

        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.clicked.connect(self.load_image)
        image_layout.addWidget(self.load_image_btn, 0, 0)

        self.load_bg_btn = QPushButton("Load Background")
        self.load_bg_btn.clicked.connect(self.load_background)
        image_layout.addWidget(self.load_bg_btn, 0, 1)

        self.process_btn = QPushButton("Process Image")
        self.process_btn.clicked.connect(self.process_image)
        image_layout.addWidget(self.process_btn, 1, 0, 1, 2)

        # Auto crop checkbox
        self.auto_crop_cb = QCheckBox("Auto Crop")
        self.auto_crop_cb.setChecked(True)
        self.auto_crop_cb.stateChanged.connect(self.on_param_changed)
        image_layout.addWidget(self.auto_crop_cb, 2, 0, 1, 2)

        controls_layout.addWidget(image_group)

        # Image processing parameters
        proc_group = QGroupBox("Image Processing Parameters")
        proc_layout = QGridLayout(proc_group)

        proc_layout.addWidget(QLabel("Rotation Angle (°):"), 0, 0)
        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-180, 180)
        self.rotation_spin.setValue(0)
        self.rotation_spin.valueChanged.connect(self.on_param_changed)
        proc_layout.addWidget(self.rotation_spin, 0, 1)

        proc_layout.addWidget(QLabel("Crop X Min:"), 1, 0)
        self.crop_xmin = QSpinBox()
        self.crop_xmin.setRange(0, 5000)
        self.crop_xmin.valueChanged.connect(self.on_param_changed)
        proc_layout.addWidget(self.crop_xmin, 1, 1)

        proc_layout.addWidget(QLabel("Crop X Max:"), 2, 0)
        self.crop_xmax = QSpinBox()
        self.crop_xmax.setRange(0, 5000)
        self.crop_xmax.setValue(1000)
        self.crop_xmax.valueChanged.connect(self.on_param_changed)
        proc_layout.addWidget(self.crop_xmax, 2, 1)

        proc_layout.addWidget(QLabel("Crop Y Min:"), 3, 0)
        self.crop_ymin = QSpinBox()
        self.crop_ymin.setRange(0, 5000)
        self.crop_ymin.valueChanged.connect(self.on_param_changed)
        proc_layout.addWidget(self.crop_ymin, 3, 1)

        proc_layout.addWidget(QLabel("Crop Y Max:"), 4, 0)
        self.crop_ymax = QSpinBox()
        self.crop_ymax.setRange(0, 5000)
        self.crop_ymax.setValue(1000)
        self.crop_ymax.valueChanged.connect(self.on_param_changed)
        proc_layout.addWidget(self.crop_ymax, 4, 1)

        proc_layout.addWidget(QLabel("Threshold:"), 5, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(20)
        self.threshold_spin.valueChanged.connect(self.on_param_changed)
        proc_layout.addWidget(self.threshold_spin, 5, 1)

        proc_layout.addWidget(QLabel("Gaussian Blur:"), 6, 0)
        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(0, 21)
        self.blur_spin.setSingleStep(2)
        self.blur_spin.setValue(5)
        self.blur_spin.valueChanged.connect(self.on_param_changed)
        proc_layout.addWidget(self.blur_spin, 6, 1)

        controls_layout.addWidget(proc_group)

        # Analysis parameters
        analysis_group = QGroupBox("Analysis Parameters")
        analysis_layout = QGridLayout(analysis_group)

        analysis_layout.addWidget(QLabel("Analysis Type:"), 0, 0)
        self.analysis_type = QComboBox()
        self.analysis_type.addItems(["Single Beam", "Pepper-Pot (Grid)"])
        analysis_layout.addWidget(self.analysis_type, 0, 1)

        analysis_layout.addWidget(QLabel("Drift Distance (mm):"), 1, 0)
        self.drift_distance = QDoubleSpinBox()
        self.drift_distance.setRange(1, 1000)
        self.drift_distance.setValue(41)
        analysis_layout.addWidget(self.drift_distance, 1, 1)

        analysis_layout.addWidget(QLabel("Scaling (mm/pixel):"), 2, 0)
        self.scaling = QDoubleSpinBox()
        self.scaling.setRange(0.001, 1)
        self.scaling.setSingleStep(0.01)
        self.scaling.setValue(0.06)
        analysis_layout.addWidget(self.scaling, 2, 1)

        analysis_layout.addWidget(QLabel("Grid Pitch (mm):"), 3, 0)
        self.grid_pitch = QDoubleSpinBox()
        self.grid_pitch.setRange(0.01, 10)
        self.grid_pitch.setValue(0.085)  # Default from paper, TEM300 grid
        analysis_layout.addWidget(self.grid_pitch, 3, 1)

        analysis_layout.addWidget(QLabel("Grid Bar Width (mm):"), 4, 0)
        self.grid_bar_width = QDoubleSpinBox()
        self.grid_bar_width.setRange(0.001, 1)
        self.grid_bar_width.setValue(0.031)  # Default from paper, TEM300 grid
        analysis_layout.addWidget(self.grid_bar_width, 4, 1)

        controls_layout.addWidget(analysis_group)

        # Analyze button
        self.analyze_btn = QPushButton("Analyze Beam")
        self.analyze_btn.clicked.connect(self.analyze_beam)
        controls_layout.addWidget(self.analyze_btn)

        # Status label
        self.status_label = QLabel("Ready")
        controls_layout.addWidget(self.status_label)

        controls_layout.addStretch()

        # === Display Panel ===
        # Tabs for different visualizations
        tabs = QTabWidget()
        display_layout.addWidget(tabs)

        # Image Tab
        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)

        self.image_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        image_layout.addWidget(self.image_canvas)
        image_toolbar = NavigationToolbar(self.image_canvas, self)
        image_layout.addWidget(image_toolbar)
        self.image_ax = self.image_canvas.figure.subplots()

        tabs.addTab(image_tab, "Image")

        # Phase Space Tabs
        self.create_phase_space_tab(tabs, "X' vs X")
        self.create_phase_space_tab(tabs, "Y' vs Y")
        self.create_phase_space_tab(tabs, "Y vs X")
        self.create_phase_space_tab(tabs, "Y' vs X")
        self.create_phase_space_tab(tabs, "Y vs X'")
        self.create_phase_space_tab(tabs, "Y' vs X'")

        # Contour & Surface plot tabs
        contour_tab = QWidget()
        contour_layout = QVBoxLayout(contour_tab)
        self.contour_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        contour_layout.addWidget(self.contour_canvas)
        contour_toolbar = NavigationToolbar(self.contour_canvas, self)
        contour_layout.addWidget(contour_toolbar)
        self.contour_ax = self.contour_canvas.figure.subplots()
        tabs.addTab(contour_tab, "Contour Plot")

        surface_tab = QWidget()
        surface_layout = QVBoxLayout(surface_tab)
        self.surface_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        surface_layout.addWidget(self.surface_canvas)
        surface_toolbar = NavigationToolbar(self.surface_canvas, self)
        surface_layout.addWidget(surface_toolbar)
        self.surface_ax = self.surface_canvas.figure.add_subplot(111, projection='3d')
        tabs.addTab(surface_tab, "Surface Plot")

        # Results Tab
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        self.results_text = QLabel("No analysis results yet")
        results_layout.addWidget(self.results_text)
        tabs.addTab(results_tab, "Results")

        # Show the UI
        self.show()

    def on_param_changed(self):
        """Called when any image processing parameter changes"""
        if self.main_image is not None:
            self.process_image()

    def auto_detect_crop(self, image, threshold=10):
        """Automatically detect crop region based on image content"""
        if image is None:
            return 0, image.shape[1], 0, image.shape[0]

        # Find non-zero or above threshold pixels
        mask = image > threshold
        if not np.any(mask):
            return 0, image.shape[1], 0, image.shape[0]

        # Find bounds of non-zero regions
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        # Get min/max indices
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        # Add padding (10% of each dimension)
        pad_x = int((xmax - xmin) * 0.1)
        pad_y = int((ymax - ymin) * 0.1)

        xmin = max(0, xmin - pad_x)
        xmax = min(image.shape[1], xmax + pad_x)
        ymin = max(0, ymin - pad_y)
        ymax = min(image.shape[0], ymax + pad_y)

        return xmin, xmax, ymin, ymax

    def create_phase_space_tab(self, parent_tabs, title):
        """Create a tab for a phase space plot"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        canvas = FigureCanvas(Figure(figsize=(5, 4)))
        layout.addWidget(canvas)
        toolbar = NavigationToolbar(canvas, self)
        layout.addWidget(toolbar)
        ax = canvas.figure.subplots()

        parent_tabs.addTab(tab, title)

        # Store the canvas and axis as attributes for later access
        attr_name = title.replace("'", "p").replace(" vs ", "_vs_").lower()
        setattr(self, f"{attr_name}_canvas", canvas)
        setattr(self, f"{attr_name}_ax", ax)

    def load_image(self):
        """Load the main beam image"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg *.tif *.bmp)")
        if file_path:
            self.main_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.main_image is not None:
                self.update_image_display(self.main_image, "Main Image Loaded")
                # Set crop values based on image size
                self.crop_xmax.setValue(self.main_image.shape[1])
                self.crop_ymax.setValue(self.main_image.shape[0])
                # Process immediately
                self.process_image()
            else:
                self.status_label.setText("Failed to load image")

    def load_background(self):
        """Load the background image"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Background Image", "",
                                                   "Images (*.png *.jpg *.tif *.bmp)")
        if file_path:
            self.bg_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.bg_image is not None:
                self.update_image_display(self.bg_image, "Background Image Loaded")
                # Process immediately
                if self.main_image is not None:
                    self.process_image()
            else:
                self.status_label.setText("Failed to load background image")

    def process_image(self):
        """Process the loaded images: background subtraction, rotation, cropping"""
        if self.main_image is None:
            self.status_label.setText("No main image loaded")
            return

        try:
            # Start with the main image
            self.processed_image = self.main_image.copy().astype(float)

            # Subtract background if available
            if self.bg_image is not None:
                if self.bg_image.shape != self.main_image.shape:
                    self.bg_image = cv2.resize(self.bg_image, (self.main_image.shape[1], self.main_image.shape[0]))
                self.processed_image = self.processed_image - self.bg_image

            # Ensure no negative values
            self.processed_image = np.clip(self.processed_image, 0, 255)

            # Apply rotation if needed
            angle = self.rotation_spin.value()
            if angle != 0:
                self.processed_image = rotate(self.processed_image, angle, reshape=False)

            # Apply cropping
            if self.auto_crop_cb.isChecked():
                # Auto-detect crop region
                xmin, xmax, ymin, ymax = self.auto_detect_crop(self.processed_image,
                                                               threshold=self.threshold_spin.value())
                # Update crop spinboxes without triggering on_param_changed
                self.crop_xmin.blockSignals(True)
                self.crop_xmax.blockSignals(True)
                self.crop_ymin.blockSignals(True)
                self.crop_ymax.blockSignals(True)

                self.crop_xmin.setValue(xmin)
                self.crop_xmax.setValue(xmax)
                self.crop_ymin.setValue(ymin)
                self.crop_ymax.setValue(ymax)

                self.crop_xmin.blockSignals(False)
                self.crop_xmax.blockSignals(False)
                self.crop_ymin.blockSignals(False)
                self.crop_ymax.blockSignals(False)
            else:
                # Use manual crop settings
                xmin = max(0, self.crop_xmin.value())
                xmax = min(self.processed_image.shape[1], self.crop_xmax.value())
                ymin = max(0, self.crop_ymin.value())
                ymax = min(self.processed_image.shape[0], self.crop_ymax.value())

            if xmin < xmax and ymin < ymax:
                self.processed_image = self.processed_image[ymin:ymax, xmin:xmax]

            # Apply Gaussian blur if needed
            blur_size = self.blur_spin.value()
            if blur_size > 0:
                # Ensure blur size is odd
                if blur_size % 2 == 0:
                    blur_size += 1
                self.processed_image = cv2.GaussianBlur(self.processed_image, (blur_size, blur_size), 0)

            # Convert back to uint8 for display
            self.processed_image = self.processed_image.astype(np.uint8)

            # Update display
            self.update_image_display(self.processed_image, "Image Processed")
        except Exception as e:
            self.status_label.setText(f"Error processing image: {str(e)}")
            print(f"Error processing image: {str(e)}")
            traceback.print_exc()

    def update_image_display(self, image, status_text):
        """Update the image display with the given image"""
        self.image_ax.clear()
        self.image_ax.imshow(image, cmap='viridis')
        self.image_ax.set_title(status_text)
        self.image_canvas.draw()
        self.status_label.setText(status_text)

    def analyze_beam(self):
        """Analyze the beam based on the selected method"""
        if self.processed_image is None:
            self.status_label.setText("No processed image available")
            return

        try:
            analysis_type = self.analysis_type.currentText()
            self.status_label.setText(f"Analyzing beam using {analysis_type} method...")

            if analysis_type == "Single Beam":
                self.analyze_single_beam()
            else:  # Pepper-Pot
                self.analyze_pepper_pot()
        except Exception as e:
            self.status_label.setText(f"Error during analysis: {str(e)}")
            print(f"Error during analysis: {str(e)}")
            traceback.print_exc()

    def analyze_single_beam(self):
        """Analyze a single beam spot"""
        # Calculate beam centroid and size
        img = self.processed_image.astype(float)
        threshold = self.threshold_spin.value()
        mask = img > threshold

        if not np.any(mask):
            self.status_label.setText("No pixels above threshold")
            return

        # Background-subtracted and thresholded image
        img_thresh = np.where(mask, img, 0)

        # Calculate beam moments
        total_intensity = img_thresh.sum()
        if total_intensity <= 0:
            self.status_label.setText("Zero total intensity after thresholding")
            return

        y_indices, x_indices = np.indices(img_thresh.shape)
        x_centroid = np.sum(x_indices * img_thresh) / total_intensity
        y_centroid = np.sum(y_indices * img_thresh) / total_intensity

        # Calculate second moments (beam size)
        x_variance = np.sum(((x_indices - x_centroid) ** 2) * img_thresh) / total_intensity
        y_variance = np.sum(((y_indices - y_centroid) ** 2) * img_thresh) / total_intensity
        xy_variance = np.sum((x_indices - x_centroid) * (y_indices - y_centroid) * img_thresh) / total_intensity

        x_rms = np.sqrt(x_variance)
        y_rms = np.sqrt(y_variance)

        # Convert to physical units
        scaling = self.scaling.value()  # mm/pixel
        x_centroid_mm = x_centroid * scaling
        y_centroid_mm = y_centroid * scaling
        x_rms_mm = x_rms * scaling
        y_rms_mm = y_rms * scaling

        # Store beam data
        self.beam_data = {
            'x_centroid': x_centroid,
            'y_centroid': y_centroid,
            'x_rms': x_rms,
            'y_rms': y_rms,
            'xy_variance': xy_variance,
            'x_centroid_mm': x_centroid_mm,
            'y_centroid_mm': y_centroid_mm,
            'x_rms_mm': x_rms_mm,
            'y_rms_mm': y_rms_mm,
            'intensity': total_intensity,
            'image': img_thresh
        }

        # Update results display
        results_text = f"""
        Beam Analysis Results:
        ---------------------
        X Centroid: {x_centroid:.2f} pixels ({x_centroid_mm:.3f} mm)
        Y Centroid: {y_centroid:.2f} pixels ({y_centroid_mm:.3f} mm)
        X RMS Size: {x_rms:.2f} pixels ({x_rms_mm:.3f} mm)
        Y RMS Size: {y_rms:.2f} pixels ({y_rms_mm:.3f} mm)
        Total Intensity: {total_intensity:.0f}
        """

        self.results_text.setText(results_text)

        # Update visualizations
        self.update_visualizations()

        self.status_label.setText("Single beam analysis complete")

    def analyze_pepper_pot(self):
        """Analyze a pepper-pot or grid image for emittance measurement"""
        img = self.processed_image.astype(float)
        threshold = self.threshold_spin.value()
        scaling = self.scaling.value()  # mm/pixel
        drift_distance = self.drift_distance.value()  # mm

        # Find peaks (beamlets)
        # Use a local maximum filter
        data_max = maximum_filter(img, size=15)
        maxima = (img == data_max)

        # Filter by threshold
        maxima[img < threshold] = 0

        # Get coordinates of local maxima
        labeled, num_objects = label(maxima)
        slices = find_objects(labeled)

        if not slices:
            self.status_label.setText("No peaks found. Try adjusting the threshold.")
            return

        x_positions = []
        y_positions = []
        x_rms_values = []
        y_rms_values = []
        intensities = []

        # Analyze each beamlet
        for dy, dx in slices:
            x_center = (dx.start + dx.stop - 1) / 2
            y_center = (dy.start + dy.stop - 1) / 2

            # Define ROI around this peak
            roi_radius = 20
            x_min = max(0, int(x_center - roi_radius))
            x_max = min(img.shape[1], int(x_center + roi_radius))
            y_min = max(0, int(y_center - roi_radius))
            y_max = min(img.shape[0], int(y_center + roi_radius))

            roi = img[y_min:y_max, x_min:x_max]

            # Skip if ROI is too small
            if roi.shape[0] < 5 or roi.shape[1] < 5:
                continue

            # Calculate moments for this beamlet
            y_indices, x_indices = np.indices(roi.shape)
            total_intensity = roi.sum()

            if total_intensity <= 0:
                continue

            intensities.append(total_intensity)

            # Calculate centroid
            x_local_centroid = np.sum(x_indices * roi) / total_intensity
            y_local_centroid = np.sum(y_indices * roi) / total_intensity

            # Global position
            x_global = x_min + x_local_centroid
            y_global = y_min + y_local_centroid

            x_positions.append(x_global)
            y_positions.append(y_global)

            # Calculate second moments (beam size)
            x_variance = np.sum(((x_indices - x_local_centroid) ** 2) * roi) / total_intensity
            y_variance = np.sum(((y_indices - y_local_centroid) ** 2) * roi) / total_intensity

            x_rms = np.sqrt(x_variance)
            y_rms = np.sqrt(y_variance)

            x_rms_values.append(x_rms)
            y_rms_values.append(y_rms)

        if not x_positions:
            self.status_label.setText("No valid beamlets found for analysis")
            return

        # Convert arrays
        x_positions = np.array(x_positions)
        y_positions = np.array(y_positions)
        x_rms_values = np.array(x_rms_values)
        y_rms_values = np.array(y_rms_values)
        intensities = np.array(intensities)

        # Sort by intensity (optional)
        if len(intensities) > 0:
            sort_idx = np.argsort(intensities)[::-1]  # descending
            x_positions = x_positions[sort_idx]
            y_positions = y_positions[sort_idx]
            x_rms_values = x_rms_values[sort_idx]
            y_rms_values = y_rms_values[sort_idx]
            intensities = intensities[sort_idx]

        # Convert to physical units
        x_positions_mm = x_positions * scaling
        y_positions_mm = y_positions * scaling
        x_rms_mm = x_rms_values * scaling
        y_rms_mm = y_rms_values * scaling

        # Calculate divergence for each beamlet
        x_prime_values = x_rms_mm / drift_distance  # rad
        y_prime_values = y_rms_mm / drift_distance  # rad

        # Calculate emittance using the method from the paper
        # Equations (10)-(12) from the paper

        # Prevent division by zero
        if np.sum(intensities) == 0:
            self.status_label.setText("Zero total intensity - cannot calculate beam parameters")
            return

        # x-plane
        x2_avg = np.sum(intensities * x_positions_mm ** 2) / np.sum(intensities)
        xp2_avg = np.sum(intensities * (x_prime_values ** 2)) / np.sum(intensities)
        xxp_avg = np.sum(intensities * x_positions_mm * x_prime_values) / np.sum(intensities)

        # y-plane
        y2_avg = np.sum(intensities * y_positions_mm ** 2) / np.sum(intensities)
        yp2_avg = np.sum(intensities * (y_prime_values ** 2)) / np.sum(intensities)
        yyp_avg = np.sum(intensities * y_positions_mm * y_prime_values) / np.sum(intensities)

        # Cross terms for 4D emittance
        xy_avg = np.sum(intensities * x_positions_mm * y_positions_mm) / np.sum(intensities)
        xyp_avg = np.sum(intensities * x_positions_mm * y_prime_values) / np.sum(intensities)
        xpy_avg = np.sum(intensities * x_prime_values * y_positions_mm) / np.sum(intensities)
        xpyp_avg = np.sum(intensities * x_prime_values * y_prime_values) / np.sum(intensities)

        # Calculate emittances
        emittance_x = np.sqrt(max(0, x2_avg * xp2_avg - xxp_avg ** 2))  # π·mm·mrad
        emittance_y = np.sqrt(max(0, y2_avg * yp2_avg - yyp_avg ** 2))  # π·mm·mrad

        # Create 4D beam matrix
        sigma_4d = np.array([
            [x2_avg, xxp_avg, xy_avg, xyp_avg],
            [xxp_avg, xp2_avg, xpy_avg, xpyp_avg],
            [xy_avg, xpy_avg, y2_avg, yyp_avg],
            [xyp_avg, xpyp_avg, yyp_avg, yp2_avg]
        ])

        # Calculate 4D emittance - ensure matrix is positive-definite
        det_sigma = np.linalg.det(sigma_4d)
        if det_sigma <= 0:
            # Add small regularization to the diagonal if needed
            diag_avg = np.mean(np.diag(sigma_4d))
            reg_factor = 1e-5 * diag_avg
            sigma_4d = sigma_4d + np.eye(4) * reg_factor
            det_sigma = np.linalg.det(sigma_4d)

        emittance_4d = np.sqrt(max(0, det_sigma))

        # Store phase space data
        self.phase_space_data = {
            'x_positions': x_positions,
            'y_positions': y_positions,
            'x_rms_values': x_rms_values,
            'y_rms_values': y_rms_values,
            'intensities': intensities,
            'x_positions_mm': x_positions_mm,
            'y_positions_mm': y_positions_mm,
            'x_prime_values': x_prime_values,
            'y_prime_values': y_prime_values,
            'emittance_x': emittance_x,
            'emittance_y': emittance_y,
            'emittance_4d': emittance_4d,
            'sigma_4d': sigma_4d
        }

        # Update results display
        results_text = f"""
        Emittance Analysis Results:
        -------------------------
        Number of beamlets: {len(x_positions)}

        RMS Emittance X: {emittance_x:.6f} π·mm·mrad
        RMS Emittance Y: {emittance_y:.6f} π·mm·mrad
        4D RMS Emittance: {emittance_4d:.6f} π²·mm²·mrad²

        Beam Matrix Elements:
        <x²> = {x2_avg:.6f} mm²
        <xx'> = {xxp_avg:.6f} mm·mrad
        <x'²> = {xp2_avg:.6f} mrad²
        <y²> = {y2_avg:.6f} mm²
        <yy'> = {yyp_avg:.6f} mm·mrad
        <y'²> = {yp2_avg:.6f} mrad²
        <xy> = {xy_avg:.6f} mm²
        <xy'> = {xyp_avg:.6f} mm·mrad
        <x'y> = {xpy_avg:.6f} mm·mrad
        <x'y'> = {xpyp_avg:.6f} mrad²
        """

        self.results_text.setText(results_text)

        # Update visualizations
        self.update_visualizations()

        self.status_label.setText("Pepper-pot analysis complete")

    def update_visualizations(self):
        """Update all visualization plots with the latest analysis data"""
        try:
            if self.analysis_type.currentText() == "Single Beam":
                self.update_single_beam_plots()
            else:  # Pepper-Pot
                self.update_pepper_pot_plots()
        except Exception as e:
            self.status_label.setText(f"Error updating visualizations: {str(e)}")
            print(f"Error updating visualizations: {str(e)}")
            traceback.print_exc()

    def update_single_beam_plots(self):
        """Update plots for single beam analysis"""
        if self.beam_data is None:
            return

        # Mark centroid on the image
        self.image_ax.clear()
        self.image_ax.imshow(self.processed_image, cmap='viridis')
        self.image_ax.plot(self.beam_data['x_centroid'], self.beam_data['y_centroid'], 'r+', markersize=10)

        # Draw RMS ellipse
        theta = np.linspace(0, 2 * np.pi, 100)
        ellipse_x = self.beam_data['x_centroid'] + self.beam_data['x_rms'] * np.cos(theta)
        ellipse_y = self.beam_data['y_centroid'] + self.beam_data['y_rms'] * np.sin(theta)
        self.image_ax.plot(ellipse_x, ellipse_y, 'r-')

        self.image_ax.set_title("Beam Spot with Centroid and RMS Size")
        self.image_canvas.draw()

        # Surface plot
        self.surface_ax.clear()
        y_indices, x_indices = np.indices(self.beam_data['image'].shape)
        self.surface_ax.plot_surface(x_indices, y_indices, self.beam_data['image'], cmap='viridis')
        self.surface_ax.set_title("Beam Intensity Profile")
        self.surface_ax.set_xlabel('X (pixels)')
        self.surface_ax.set_ylabel('Y (pixels)')
        self.surface_ax.set_zlabel('Intensity')
        self.surface_canvas.draw()

        # Contour plot
        self.contour_ax.clear()
        self.contour_ax.contourf(self.beam_data['image'], cmap='viridis')
        self.contour_ax.set_title("Beam Intensity Contours")
        self.contour_ax.set_xlabel('X (pixels)')
        self.contour_ax.set_ylabel('Y (pixels)')
        self.contour_canvas.draw()

        # Simple placeholder for phase space plots
        # For a single beam without time/multiple shots, we cannot measure phase space directly
        for plot_type in ["xp_vs_x", "yp_vs_y", "y_vs_x", "yp_vs_x", "y_vs_xp", "yp_vs_xp"]:
            ax = getattr(self, f"{plot_type}_ax")
            ax.clear()
            ax.text(0.5, 0.5, "Phase space not available for single beam analysis",
                    ha='center', va='center', transform=ax.transAxes)
            canvas = getattr(self, f"{plot_type}_canvas")
            canvas.draw()

    def update_pepper_pot_plots(self):
        """Update plots for pepper-pot analysis"""
        if self.phase_space_data is None:
            return

        # Mark beamlet positions on the image
        self.image_ax.clear()
        self.image_ax.imshow(self.processed_image, cmap='viridis')
        x_pos = self.phase_space_data['x_positions']
        y_pos = self.phase_space_data['y_positions']
        intensities = self.phase_space_data['intensities']

        # Scale marker sizes based on intensities
        if intensities.size > 0:
            max_intensity = intensities.max()
            if max_intensity > 0:
                marker_sizes = 10 + 20 * intensities / max_intensity
            else:
                marker_sizes = 10 * np.ones_like(intensities)
        else:
            marker_sizes = []

        self.image_ax.scatter(x_pos, y_pos, s=marker_sizes, c='r', marker='+')

        # Draw RMS ellipses for each beamlet
        for i in range(len(x_pos)):
            theta = np.linspace(0, 2 * np.pi, 100)
            ellipse_x = x_pos[i] + self.phase_space_data['x_rms_values'][i] * np.cos(theta)
            ellipse_y = y_pos[i] + self.phase_space_data['y_rms_values'][i] * np.sin(theta)
            self.image_ax.plot(ellipse_x, ellipse_y, 'r-', alpha=0.3)

        self.image_ax.set_title("Pepper-Pot Beamlet Analysis")
        self.image_canvas.draw()

        # Phase space plots
        # X' vs X
        self.xp_vs_x_ax.clear()
        self.xp_vs_x_ax.scatter(
            self.phase_space_data['x_positions_mm'],
            self.phase_space_data['x_prime_values'],
            c=self.phase_space_data['intensities'],
            cmap='viridis',
            s=30
        )
        self.xp_vs_x_ax.set_title("X' vs X Phase Space")
        self.xp_vs_x_ax.set_xlabel('X Position (mm)')
        self.xp_vs_x_ax.set_ylabel('X Divergence (mrad)')
        self.xp_vs_x_canvas.draw()

        # Y' vs Y
        self.yp_vs_y_ax.clear()
        self.yp_vs_y_ax.scatter(
            self.phase_space_data['y_positions_mm'],
            self.phase_space_data['y_prime_values'],
            c=self.phase_space_data['intensities'],
            cmap='viridis',
            s=30
        )
        self.yp_vs_y_ax.set_title("Y' vs Y Phase Space")
        self.yp_vs_y_ax.set_xlabel('Y Position (mm)')
        self.yp_vs_y_ax.set_ylabel('Y Divergence (mrad)')
        self.yp_vs_y_canvas.draw()

        # Y vs X
        self.y_vs_x_ax.clear()
        self.y_vs_x_ax.scatter(
            self.phase_space_data['x_positions_mm'],
            self.phase_space_data['y_positions_mm'],
            c=self.phase_space_data['intensities'],
            cmap='viridis',
            s=30
        )
        self.y_vs_x_ax.set_title("Y vs X Position")
        self.y_vs_x_ax.set_xlabel('X Position (mm)')
        self.y_vs_x_ax.set_ylabel('Y Position (mm)')
        self.y_vs_x_canvas.draw()

        # Y' vs X
        self.yp_vs_x_ax.clear()
        self.yp_vs_x_ax.scatter(
            self.phase_space_data['x_positions_mm'],
            self.phase_space_data['y_prime_values'],
            c=self.phase_space_data['intensities'],
            cmap='viridis',
            s=30
        )
        self.yp_vs_x_ax.set_title("Y' vs X")
        self.yp_vs_x_ax.set_xlabel('X Position (mm)')
        self.yp_vs_x_ax.set_ylabel('Y Divergence (mrad)')
        self.yp_vs_x_canvas.draw()

        # Y vs X'
        self.y_vs_xp_ax.clear()
        self.y_vs_xp_ax.scatter(
            self.phase_space_data['x_prime_values'],
            self.phase_space_data['y_positions_mm'],
            c=self.phase_space_data['intensities'],
            cmap='viridis',
            s=30
        )
        self.y_vs_xp_ax.set_title("Y vs X'")
        self.y_vs_xp_ax.set_xlabel('X Divergence (mrad)')
        self.y_vs_xp_ax.set_ylabel('Y Position (mm)')
        self.y_vs_xp_canvas.draw()

        # Y' vs X'
        self.yp_vs_xp_ax.clear()
        self.yp_vs_xp_ax.scatter(
            self.phase_space_data['x_prime_values'],
            self.phase_space_data['y_prime_values'],
            c=self.phase_space_data['intensities'],
            cmap='viridis',
            s=30
        )
        self.yp_vs_xp_ax.set_title("Y' vs X'")
        self.yp_vs_xp_ax.set_xlabel('X Divergence (mrad)')
        self.yp_vs_xp_ax.set_ylabel('Y Divergence (mrad)')
        self.yp_vs_xp_canvas.draw()

        # Contour plot - 2D histogram of beamlet positions
        self.contour_ax.clear()

        # Skip if we don't have enough points
        if len(self.phase_space_data['x_positions']) > 3:
            hist, xedges, yedges = np.histogram2d(
                self.phase_space_data['x_positions'],
                self.phase_space_data['y_positions'],
                bins=min(20, len(self.phase_space_data['x_positions']) // 2),
                weights=self.phase_space_data['intensities']
            )
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            self.contour_ax.contourf(hist.T, extent=extent, cmap='viridis')
        else:
            self.contour_ax.text(0.5, 0.5, "Not enough data points for contour plot",
                                 ha='center', va='center', transform=self.contour_ax.transAxes)

        self.contour_ax.set_title("Beamlet Intensity Distribution")
        self.contour_ax.set_xlabel('X (pixels)')
        self.contour_ax.set_ylabel('Y (pixels)')
        self.contour_canvas.draw()

        # Surface plot - 3D view of beamlet intensities
        self.surface_ax.clear()
        x = self.phase_space_data['x_positions']
        y = self.phase_space_data['y_positions']
        z = self.phase_space_data['intensities']

        # Skip if we don't have enough points
        if len(x) > 3:
            try:
                # Create a grid for surface plotting
                xi = np.linspace(min(x), max(x), 20)
                yi = np.linspace(min(y), max(y), 20)
                xi, yi = np.meshgrid(xi, yi)

                # Interpolate z values on the grid
                from scipy.interpolate import griddata
                zi = griddata((x, y), z, (xi, yi), method='cubic', fill_value=0)

                self.surface_ax.plot_surface(xi, yi, zi, cmap='viridis')
            except Exception as e:
                print(f"Error creating surface plot: {str(e)}")
                self.surface_ax.text(0.5, 0.5, 0.5, "Error creating surface plot",
                                     ha='center', va='center')
        else:
            self.surface_ax.text(0.5, 0.5, 0.5, "Not enough data points for surface plot",
                                 ha='center', va='center')

        self.surface_ax.set_title("Beamlet Intensity Surface")
        self.surface_ax.set_xlabel('X (pixels)')
        self.surface_ax.set_ylabel('Y (pixels)')
        self.surface_ax.set_zlabel('Intensity')
        self.surface_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BeamAnalyzer()
    sys.exit(app.exec_())