import numpy as np
from PIL import Image
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QTabWidget, QGridLayout, QSpinBox, QDoubleSpinBox,
                               QGroupBox, QCheckBox, QSlider)
from PySide6.QtCore import Qt, QTimer, QSize
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots in Qt"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        if parent is not None:
            self.setParent(parent)
        self.fig.tight_layout()


class LeftPanel(QWidget):
    def __init__(self, parent, analyzer):
        super().__init__(parent)
        self.main_window = parent
        self.analyzer = analyzer
        self.setup_ui()

    def setup_ui(self):
        # [Setup UI code remains unchanged]
        self.layout = QVBoxLayout(self)

        # Add intensity scale control at the top
        self.intensity_control_group = QGroupBox("Intensity Scale Control")
        self.intensity_layout = QHBoxLayout(self.intensity_control_group)

        self.intensity_scale_label = QLabel("Scale Factor:")
        self.intensity_scale_spin = QDoubleSpinBox()
        self.intensity_scale_spin.setRange(0.01, 10.0)
        self.intensity_scale_spin.setValue(1.0)
        self.intensity_scale_spin.setSingleStep(0.1)
        self.intensity_scale_spin.valueChanged.connect(self.on_intensity_scale_changed)

        self.intensity_scale_slider = QSlider(Qt.Horizontal)
        self.intensity_scale_slider.setRange(1, 1000)
        self.intensity_scale_slider.setValue(100)  # 1.0 * 100
        self.intensity_scale_slider.setTickPosition(QSlider.TicksBelow)
        self.intensity_scale_slider.setTickInterval(100)
        self.intensity_scale_slider.valueChanged.connect(lambda v: self.intensity_scale_spin.setValue(v / 100))

        self.auto_scale_check = QCheckBox("Auto Scale")
        self.auto_scale_check.setChecked(True)
        self.auto_scale_check.stateChanged.connect(self.on_auto_scale_changed)

        self.intensity_layout.addWidget(self.intensity_scale_label)
        self.intensity_layout.addWidget(self.intensity_scale_spin)
        self.intensity_layout.addWidget(self.intensity_scale_slider)
        self.intensity_layout.addWidget(self.auto_scale_check)

        self.layout.addWidget(self.intensity_control_group)

        # Image display tabs
        self.image_tabs = QTabWidget()
        self.image_tabs.setTabPosition(QTabWidget.North)
        self.image_tabs.setDocumentMode(True)

        # Raw image tab
        self.raw_image_widget = QWidget()
        self.raw_image_layout = QVBoxLayout(self.raw_image_widget)
        self.raw_image_layout.setContentsMargins(2, 2, 2, 2)
        self.raw_image_layout.setSpacing(1)
        self.raw_image_canvas = MatplotlibCanvas(self.raw_image_widget, width=5, height=4)
        self.raw_image_toolbar = NavigationToolbar(self.raw_image_canvas, self.raw_image_widget)
        self.raw_image_toolbar.setIconSize(QSize(16, 16))
        self.raw_image_layout.addWidget(self.raw_image_toolbar)
        self.raw_image_layout.addWidget(self.raw_image_canvas)

        # Background image tab
        self.background_image_widget = QWidget()
        self.background_image_layout = QVBoxLayout(self.background_image_widget)
        self.background_image_layout.setContentsMargins(2, 2, 2, 2)
        self.background_image_layout.setSpacing(1)
        self.background_image_canvas = MatplotlibCanvas(self.background_image_widget, width=5, height=4)
        self.background_image_toolbar = NavigationToolbar(self.background_image_canvas, self.background_image_widget)
        self.background_image_toolbar.setIconSize(QSize(16, 16))
        self.background_image_layout.addWidget(self.background_image_toolbar)
        self.background_image_layout.addWidget(self.background_image_canvas)

        # Processed image tab
        self.processed_image_widget = QWidget()
        self.processed_image_layout = QVBoxLayout(self.processed_image_widget)
        self.processed_image_layout.setContentsMargins(2, 2, 2, 2)
        self.processed_image_layout.setSpacing(1)
        self.processed_image_canvas = MatplotlibCanvas(self.processed_image_widget, width=5, height=4)
        self.processed_image_toolbar = NavigationToolbar(self.processed_image_canvas, self.processed_image_widget)
        self.processed_image_toolbar.setIconSize(QSize(16, 16))
        self.processed_image_layout.addWidget(self.processed_image_toolbar)
        self.processed_image_layout.addWidget(self.processed_image_canvas)

        # Add tabs
        self.image_tabs.addTab(self.raw_image_widget, "Raw Image")
        self.image_tabs.addTab(self.background_image_widget, "Background")
        self.image_tabs.addTab(self.processed_image_widget, "Processed Image")

        self.layout.addWidget(self.image_tabs)

        # Image processing parameters
        self.param_group = QGroupBox("Image Processing Parameters")
        self.param_layout = QGridLayout(self.param_group)

        # Rotation
        self.param_layout.addWidget(QLabel("Rotation Angle:"), 0, 0)
        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-180, 180)
        self.rotation_spin.setValue(0)
        self.rotation_spin.setSingleStep(0.1)
        self.rotation_spin.valueChanged.connect(self.on_parameter_changed)
        self.param_layout.addWidget(self.rotation_spin, 0, 1)

        # Rotation slider
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(-180, 180)
        self.rotation_slider.setValue(0)
        self.rotation_slider.setTickPosition(QSlider.TicksBelow)
        self.rotation_slider.setTickInterval(30)
        self.rotation_slider.valueChanged.connect(lambda v: self.rotation_spin.setValue(v))
        self.param_layout.addWidget(self.rotation_slider, 0, 2, 1, 2)

        # Crop
        self.param_layout.addWidget(QLabel("Crop X Min:"), 1, 0)
        self.crop_x_min_spin = QSpinBox()
        self.crop_x_min_spin.setRange(0, 10000)
        self.crop_x_min_spin.setValue(360)
        self.crop_x_min_spin.valueChanged.connect(self.on_parameter_changed)
        self.param_layout.addWidget(self.crop_x_min_spin, 1, 1)

        # X Min slider
        self.crop_x_min_slider = QSlider(Qt.Horizontal)
        self.crop_x_min_slider.setRange(0, 1000)
        self.crop_x_min_slider.setValue(360)
        self.crop_x_min_slider.valueChanged.connect(lambda v: self.crop_x_min_spin.setValue(v))
        self.param_layout.addWidget(self.crop_x_min_slider, 1, 2, 1, 2)

        self.param_layout.addWidget(QLabel("Crop X Max:"), 2, 0)
        self.crop_x_max_spin = QSpinBox()
        self.crop_x_max_spin.setRange(0, 10000)
        self.crop_x_max_spin.setValue(800)
        self.crop_x_max_spin.valueChanged.connect(self.on_parameter_changed)
        self.param_layout.addWidget(self.crop_x_max_spin, 2, 1)

        # X Max slider
        self.crop_x_max_slider = QSlider(Qt.Horizontal)
        self.crop_x_max_slider.setRange(0, 1000)
        self.crop_x_max_slider.setValue(800)
        self.crop_x_max_slider.valueChanged.connect(lambda v: self.crop_x_max_spin.setValue(v))
        self.param_layout.addWidget(self.crop_x_max_slider, 2, 2, 1, 2)

        self.param_layout.addWidget(QLabel("Crop Y Min:"), 3, 0)
        self.crop_y_min_spin = QSpinBox()
        self.crop_y_min_spin.setRange(0, 10000)
        self.crop_y_min_spin.setValue(360)
        self.crop_y_min_spin.valueChanged.connect(self.on_parameter_changed)
        self.param_layout.addWidget(self.crop_y_min_spin, 3, 1)

        # Y Min slider
        self.crop_y_min_slider = QSlider(Qt.Horizontal)
        self.crop_y_min_slider.setRange(0, 1000)
        self.crop_y_min_slider.setValue(360)
        self.crop_y_min_slider.valueChanged.connect(lambda v: self.crop_y_min_spin.setValue(v))
        self.param_layout.addWidget(self.crop_y_min_slider, 3, 2, 1, 2)

        self.param_layout.addWidget(QLabel("Crop Y Max:"), 4, 0)
        self.crop_y_max_spin = QSpinBox()
        self.crop_y_max_spin.setRange(0, 10000)
        self.crop_y_max_spin.setValue(800)
        self.crop_y_max_spin.valueChanged.connect(self.on_parameter_changed)
        self.param_layout.addWidget(self.crop_y_max_spin, 4, 1)

        # Y Max slider
        self.crop_y_max_slider = QSlider(Qt.Horizontal)
        self.crop_y_max_slider.setRange(0, 1000)
        self.crop_y_max_slider.setValue(800)
        self.crop_y_max_slider.valueChanged.connect(lambda v: self.crop_y_max_spin.setValue(v))
        self.param_layout.addWidget(self.crop_y_max_slider, 4, 2, 1, 2)

        # Peak detection
        self.param_layout.addWidget(QLabel("Peak Detection Alpha:"), 5, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.01, 10)
        self.alpha_spin.setValue(0.57)
        self.alpha_spin.setSingleStep(0.01)
        self.param_layout.addWidget(self.alpha_spin, 5, 1)

        # Alpha slider
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(1, 1000)
        self.alpha_slider.setValue(57)
        self.alpha_slider.valueChanged.connect(lambda v: self.alpha_spin.setValue(v / 100))
        self.param_layout.addWidget(self.alpha_slider, 5, 2, 1, 2)

        self.param_layout.addWidget(QLabel("Peak Size:"), 6, 0)
        self.peak_size_spin = QSpinBox()
        self.peak_size_spin.setRange(1, 100)
        self.peak_size_spin.setValue(10)
        self.param_layout.addWidget(self.peak_size_spin, 6, 1)

        # Peak size slider
        self.peak_size_slider = QSlider(Qt.Horizontal)
        self.peak_size_slider.setRange(1, 100)
        self.peak_size_slider.setValue(10)
        self.peak_size_slider.valueChanged.connect(lambda v: self.peak_size_spin.setValue(v))
        self.param_layout.addWidget(self.peak_size_slider, 6, 2, 1, 2)

        # Threshold
        self.param_layout.addWidget(QLabel("Intensity Threshold:"), 7, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 1)
        self.threshold_spin.setValue(0.2)
        self.threshold_spin.setSingleStep(0.01)
        self.param_layout.addWidget(self.threshold_spin, 7, 1)

        # Threshold slider
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 100)
        self.threshold_slider.setValue(20)
        self.threshold_slider.valueChanged.connect(lambda v: self.threshold_spin.setValue(v / 100))
        self.param_layout.addWidget(self.threshold_slider, 7, 2, 1, 2)

        # Show profiles checkbox
        self.show_profiles_check = QCheckBox("Show Beam Profiles")
        self.show_profiles_check.setChecked(True)
        self.show_profiles_check.stateChanged.connect(self.on_show_profiles_changed)
        self.param_layout.addWidget(self.show_profiles_check, 8, 0, 1, 2)

        # Live update checkbox
        self.live_update_check = QCheckBox("Live Updates")
        self.live_update_check.setChecked(True)
        self.param_layout.addWidget(self.live_update_check, 8, 2, 1, 2)

        # Create a timer for debouncing rapid parameter changes
        self.param_timer = QTimer()
        self.param_timer.setSingleShot(True)
        self.param_timer.timeout.connect(self.process_images)

        self.layout.addWidget(self.param_group)

    # [Other methods remain unchanged]
    def on_intensity_scale_changed(self, value):
        """Update intensity scale value and redisplay images"""
        self.analyzer.intensity_scale = value
        if self.analyzer.cropped_image is not None:
            self.display_image(
                self.processed_image_canvas,
                self.analyzer.cropped_image,
                "Processed Image",
                self.show_profiles_check.isChecked(),
                self.main_window.colormap_combo.currentText()
            )

    def on_auto_scale_changed(self, state):
        """Enable/disable manual intensity controls"""
        self.intensity_scale_spin.setEnabled(not state)
        self.intensity_scale_slider.setEnabled(not state)

        # If auto scale is turned on, reset to default value
        if state:
            self.intensity_scale_spin.setValue(1.0)
            self.intensity_scale_slider.setValue(100)

    def on_parameter_changed(self):
        """Handler for parameter value changes"""
        if self.live_update_check.isChecked():
            # Restart the timer to debounce rapid changes
            self.param_timer.start(300)  # 300ms debounce time

    def on_show_profiles_changed(self):
        """Handler for show profiles checkbox state changes"""
        if self.analyzer.cropped_image is not None:
            # Redisplay processed image with or without profiles
            self.display_image(
                self.processed_image_canvas,
                self.analyzer.cropped_image,
                "Processed Image",
                self.show_profiles_check.isChecked(),
                self.main_window.colormap_combo.currentText()
            )

    def update_slider_ranges(self):
        """Update slider ranges based on the loaded image dimensions"""
        if self.analyzer.raw_image is None:
            return

        height, width = self.analyzer.raw_image.shape

        # Update X crop sliders
        self.crop_x_min_slider.setRange(0, width - 1)
        self.crop_x_max_slider.setRange(0, width)
        self.crop_x_min_slider.setValue(min(self.crop_x_min_spin.value(), width - 1))
        self.crop_x_max_slider.setValue(min(self.crop_x_max_spin.value(), width))

        # Update X crop spinboxes
        self.crop_x_min_spin.setRange(0, width - 1)
        self.crop_x_max_spin.setRange(0, width)

        # Update Y crop sliders
        self.crop_y_min_slider.setRange(0, height - 1)
        self.crop_y_max_slider.setRange(0, height)
        self.crop_y_min_slider.setValue(min(self.crop_y_min_spin.value(), height - 1))
        self.crop_y_max_slider.setValue(min(self.crop_y_max_spin.value(), height))

        # Update Y crop spinboxes
        self.crop_y_min_spin.setRange(0, height - 1)
        self.crop_y_max_spin.setRange(0, height)

    def load_image(self, filepath):
        """Load the main image"""
        try:
            self.analyzer.raw_image = self.analyzer.load_image(filepath)
            # Clear the figure to prevent duplicate colorbars
            self.raw_image_canvas.fig.clear()
            self.display_image(
                self.raw_image_canvas,
                self.analyzer.raw_image,
                "Raw Image",
                colormap=self.main_window.colormap_combo.currentText()
            )
            self.main_window.statusBar().showMessage(f"Loaded image: {os.path.basename(filepath)}")

            # Update slider ranges based on the image dimensions
            self.update_slider_ranges()

            # Process immediately if live update is enabled
            if self.live_update_check.isChecked() and self.analyzer.raw_image is not None:
                self.process_images()
            return True
        except Exception as e:
            self.main_window.statusBar().showMessage(f"Error loading image: {str(e)}")
            return False

    def load_background(self, filepath):
        """Load the background image"""
        try:
            self.analyzer.background_image = self.analyzer.load_image(filepath)
            # Clear the figure to prevent duplicate colorbars
            self.background_image_canvas.fig.clear()
            self.display_image(
                self.background_image_canvas,
                self.analyzer.background_image,
                "Background Image",
                colormap=self.main_window.colormap_combo.currentText()
            )
            self.main_window.statusBar().showMessage(f"Loaded background: {os.path.basename(filepath)}")

            # Process immediately if live update is enabled
            if self.live_update_check.isChecked() and self.analyzer.raw_image is not None:
                self.process_images()
            return True
        except Exception as e:
            self.main_window.statusBar().showMessage(f"Error loading background: {str(e)}")
            return False

    def process_images(self):
        """Process images with background subtraction and rotation"""
        if self.analyzer.raw_image is None:
            self.main_window.statusBar().showMessage("Please load an image first.")
            return False

        try:
            # Update analyzer parameters
            self.analyzer.rotation_angle = self.rotation_spin.value()

            # Subtract background if available
            if self.analyzer.background_image is not None:
                self.analyzer.processed_image = self.analyzer.subtract_background(
                    self.analyzer.raw_image, self.analyzer.background_image
                )
            else:
                self.analyzer.processed_image = self.analyzer.raw_image.copy()

            # Rotate image
            if self.analyzer.rotation_angle != 0:
                self.analyzer.processed_image = self.analyzer.rotate_image(
                    self.analyzer.processed_image, self.analyzer.rotation_angle
                )

            # Crop image
            x_min = self.crop_x_min_spin.value()
            x_max = self.crop_x_max_spin.value()
            y_min = self.crop_y_min_spin.value()
            y_max = self.crop_y_max_spin.value()

            self.analyzer.cropped_image = self.analyzer.crop_image(
                self.analyzer.processed_image, x_min, x_max, y_min, y_max
            )

            # Calculate profiles
            self.analyzer.x_profile, self.analyzer.y_profile = self.analyzer.calculate_profiles(
                self.analyzer.cropped_image
            )

            # Display processed image with or without profiles
            self.display_image(
                self.processed_image_canvas,
                self.analyzer.cropped_image,
                "Processed Image",
                self.show_profiles_check.isChecked(),
                self.main_window.colormap_combo.currentText()
            )

            # Store contour data for later use
            self.main_window.contour_data = {
                'image': self.analyzer.cropped_image,
                'x_profile': self.analyzer.x_profile,
                'y_profile': self.analyzer.y_profile
            }

            # Generate contour plot (will use the right panel method)
            self.main_window.right_panel.display_contour_plot()

            # Generate surface plot (will use the right panel method)
            self.main_window.right_panel.display_surface_plot()

            self.main_window.statusBar().showMessage("Images processed successfully")

            # Switch to processed image tab
            self.image_tabs.setCurrentIndex(2)

            return True

        except Exception as e:
            self.main_window.statusBar().showMessage(f"Error processing images: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def display_image(self, canvas, image, title="Image", show_profiles=False, colormap='jet'):
        """Display an image on the given canvas with enhanced profile integration"""
        if image is None:
            return

        # Clear the entire figure including colorbars
        canvas.fig.clear()

        # Create a single axes for the image
        canvas.axes = canvas.fig.add_subplot(111)

        # Get intensity scale factor (either from auto-scaling or manual setting)
        if self.auto_scale_check.isChecked():
            # Auto scale: set vmax to 90% of the max value
            vmin = np.min(image)
            vmax = np.max(image) * 0.9
        else:
            # Manual scale: apply the user-defined scale factor
            vmin = np.min(image)
            vmax = np.max(image) / self.analyzer.intensity_scale

        # Display image with enhanced visualization
        img = canvas.axes.imshow(image, cmap=colormap, vmin=vmin, vmax=vmax,
                                 aspect='equal', interpolation='nearest')

        canvas.axes.set_title(title)
        canvas.axes.set_xlabel('X Position (pixels)')
        canvas.axes.set_ylabel('Y Position (pixels)')

        # Add profiles if requested and available - integrated directly with the main plot
        if show_profiles and canvas == self.processed_image_canvas and self.analyzer.x_profile is not None and self.analyzer.y_profile is not None:
            height, width = image.shape

            # Add X profile at the bottom (x-axis)
            x_positions = np.arange(len(self.analyzer.x_profile))
            x_profile = self.analyzer.x_profile
            # Scale profile for better visualization
            max_height = height * 0.15
            x_profile_scaled = (x_profile / np.max(x_profile) * max_height) if np.max(x_profile) > 0 else x_profile
            # Plot at the bottom edge of the image using default matplotlib colors
            canvas.axes.plot(x_positions, height - x_profile_scaled, '-')
            canvas.axes.fill_between(x_positions, height, height - x_profile_scaled, alpha=0.3)

            # Add Y profile on the left (y-axis)
            y_positions = np.arange(len(self.analyzer.y_profile))
            y_profile = self.analyzer.y_profile
            # Scale profile for better visualization
            max_width = width * 0.15
            y_profile_scaled = (y_profile / np.max(y_profile) * max_width) if np.max(y_profile) > 0 else y_profile
            # Plot on the left edge of the image using default matplotlib colors
            canvas.axes.plot(y_profile_scaled, y_positions, '-')
            canvas.axes.fill_betweenx(y_positions, 0, y_profile_scaled, alpha=0.3)

        # Add colorbar
        cbar = canvas.fig.colorbar(img, ax=canvas.axes, pad=0.02)
        cbar.set_label('Intensity')

        try:
            # Refresh canvas with tight_layout - may generate warnings for complex layouts
            canvas.fig.tight_layout()
        except:
            # Fall back to simple draw without tight_layout if it fails
            pass

        canvas.draw()
