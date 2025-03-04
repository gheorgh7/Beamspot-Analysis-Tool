import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QTabWidget, QGridLayout, QDoubleSpinBox, QSpinBox,
                             QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
                             QLineEdit, QFileDialog, QMessageBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots in Qt"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        if parent is not None:
            self.setParent(parent)
        self.fig.tight_layout()


class RightPanel(QWidget):
    def __init__(self, parent, analyzer):
        super().__init__(parent)
        self.main_window = parent
        self.analyzer = analyzer
        self.setup_ui()
        
    def setup_ui(self):
        self.layout = QVBoxLayout(self)
        
        # Analysis parameters group
        self.analysis_group = QGroupBox("Analysis Parameters")
        self.analysis_layout = QGridLayout(self.analysis_group)
        self.analysis_layout.addWidget(QLabel("Bin Count:"), 5, 0)
        self.bin_count_spin = QSpinBox()
        self.bin_count_spin.setRange(10, 200)
        self.bin_count_spin.setValue(50)
        self.bin_count_spin.setSingleStep(10)
        self.analysis_layout.addWidget(self.bin_count_spin, 5, 1)
        
        # Scaling
        self.analysis_layout.addWidget(QLabel("Scaling (mm/pixel):"), 0, 0)
        self.scaling_spin = QDoubleSpinBox()
        self.scaling_spin.setRange(0.0001, 10)
        self.scaling_spin.setValue(57.87 / 1000)
        self.scaling_spin.setSingleStep(0.001)
        self.scaling_spin.setDecimals(6)
        self.analysis_layout.addWidget(self.scaling_spin, 0, 1)
        
        # Offset
        self.analysis_layout.addWidget(QLabel("Offset:"), 0, 2)
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-100, 100)
        self.offset_spin.setValue(0)
        self.offset_spin.setSingleStep(0.1)
        self.analysis_layout.addWidget(self.offset_spin, 0, 3)
        
        # Distance
        self.analysis_layout.addWidget(QLabel("Drift Distance L (mm):"), 1, 0)
        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(1, 1000)
        self.distance_spin.setValue(41)
        self.distance_spin.setSingleStep(1)
        self.analysis_layout.addWidget(self.distance_spin, 1, 1)
        
        # X0 and Y0 positions
        self.analysis_layout.addWidget(QLabel("X0 Positions:"), 2, 0)
        self.x0_edit = QLineEdit("-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6")
        self.analysis_layout.addWidget(self.x0_edit, 2, 1, 1, 3)
        
        self.analysis_layout.addWidget(QLabel("Y0 Positions:"), 3, 0)
        self.y0_edit = QLineEdit("-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12")
        self.analysis_layout.addWidget(self.y0_edit, 3, 1, 1, 3)

        # Analyze button
        self.analyze_btn = QPushButton("Analyze Beam Spots")
        self.analyze_btn.clicked.connect(self.analyze_spots)
        self.analysis_layout.addWidget(self.analyze_btn, 4, 0, 1, 4)

        # Results tabs
        self.results_tabs = QTabWidget()
        
        # Spot detection tab
        self.spots_widget = QWidget()
        self.spots_layout = QVBoxLayout(self.spots_widget)
        self.spots_canvas = MatplotlibCanvas(self.spots_widget, width=5, height=4)
        self.spots_toolbar = NavigationToolbar(self.spots_canvas, self.spots_widget)
        self.spots_layout.addWidget(self.spots_toolbar)
        self.spots_layout.addWidget(self.spots_canvas)
        
        # Emittance results tab
        self.emittance_widget = QWidget()
        self.emittance_layout = QVBoxLayout(self.emittance_widget)
        
        # Emittance table
        self.emittance_table = QTableWidget()
        self.emittance_table.setColumnCount(2)
        self.emittance_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.emittance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        self.emittance_layout.addWidget(self.emittance_table)
        
        # X-X' and Y-Y' plots
        self.phase_space_widget = QWidget()
        self.phase_space_layout = QVBoxLayout(self.phase_space_widget)
        
        # X-X' plot
        self.xx_canvas = MatplotlibCanvas(self.phase_space_widget, width=5, height=4)
        self.xx_toolbar = NavigationToolbar(self.xx_canvas, self.phase_space_widget)
        
        # Y-Y' plot
        self.yy_canvas = MatplotlibCanvas(self.phase_space_widget, width=5, height=4)
        self.yy_toolbar = NavigationToolbar(self.yy_canvas, self.phase_space_widget)
        
        # Contour plot tab
        self.contour_widget = QWidget()
        self.contour_layout = QVBoxLayout(self.contour_widget)
        self.contour_canvas = MatplotlibCanvas(self.contour_widget, width=5, height=4)
        self.contour_toolbar = NavigationToolbar(self.contour_canvas, self.contour_widget)
        self.contour_layout.addWidget(self.contour_toolbar)
        self.contour_layout.addWidget(self.contour_canvas)
        
        # Surface plot tab
        self.surface_widget = QWidget()
        self.surface_layout = QVBoxLayout(self.surface_widget)
        self.surface_canvas = MatplotlibCanvas(self.surface_widget, width=5, height=4)
        # Set up 3D axes
        self.surface_canvas.fig.clear()
        self.surface_canvas.axes = self.surface_canvas.fig.add_subplot(111, projection='3d')
        self.surface_toolbar = NavigationToolbar(self.surface_canvas, self.surface_widget)
        self.surface_layout.addWidget(self.surface_toolbar)
        self.surface_layout.addWidget(self.surface_canvas)
        
        # Add tabs
        self.results_tabs.addTab(self.spots_widget, "Spot Detection")
        self.results_tabs.addTab(self.emittance_widget, "Emittance Results")
        self.results_tabs.addTab(self.phase_space_widget, "Phase Space")
        self.results_tabs.addTab(self.contour_widget, "Contour Plot")
        self.results_tabs.addTab(self.surface_widget, "Surface Plot")
        
        # Layout for phase space plots
        phase_space_container = QWidget()
        phase_space_container_layout = QVBoxLayout(phase_space_container)
        
        phase_space_container_layout.addWidget(self.xx_toolbar)
        phase_space_container_layout.addWidget(self.xx_canvas)
        phase_space_container_layout.addWidget(self.yy_toolbar)
        phase_space_container_layout.addWidget(self.yy_canvas)
        
        self.phase_space_layout.addWidget(phase_space_container)
        
        # Add groups to layout
        self.layout.addWidget(self.analysis_group)
        self.layout.addWidget(self.results_tabs)
    
    def analyze_spots(self):
        """Analyze beam spots and calculate emittance"""
        if self.analyzer.cropped_image is None:
            QMessageBox.warning(self.main_window, "Warning", "Please process images first.")
            return False
            
        try:
            # Update analyzer parameters
            self.analyzer.alpha = self.main_window.left_panel.alpha_spin.value()
            self.analyzer.peak_size = self.main_window.left_panel.peak_size_spin.value()
            self.analyzer.threshold = self.main_window.left_panel.threshold_spin.value()
            self.analyzer.scaling = self.scaling_spin.value()
            self.analyzer.offset = self.offset_spin.value()
            self.analyzer.distance = self.distance_spin.value()
                
            # Parse X0 and Y0 positions
            try:
                x0_positions = [float(x.strip()) for x in self.x0_edit.text().split(',')]
                y0_positions = [float(y.strip()) for y in self.y0_edit.text().split(',')]
            except ValueError:
                QMessageBox.warning(self.main_window, "Warning", "Invalid X0 or Y0 positions. Please enter comma-separated values.")
                return False
                
            # Find spots
            sigma = self.analyzer.get_std(self.analyzer.cropped_image)
            x_hole, y_hole = self.analyzer.get_peaks(
                self.analyzer.cropped_image, sigma, self.analyzer.alpha, self.analyzer.peak_size
            )
                
            # Store hole positions
            self.analyzer.xhole_positions = x_hole
            self.analyzer.yhole_positions = y_hole
                
            # Create hole coordinates list
            hole_coordinates = []
            for i, j in zip(x_hole, y_hole):
                hole_coordinates.append([i, j])
                
            # Sort hole coordinates by y then x
            hole_coordinates = sorted(hole_coordinates, key=lambda k: [k[1], k[0]])
            self.analyzer.hole_coordinates = hole_coordinates
                
            # Analyze holes
            self.analyzer.clean_hole_data, self.analyzer.clean_hole_sizes = self.analyzer.analyze_holes(
                self.analyzer.cropped_image, hole_coordinates, self.analyzer.threshold, self.analyzer.peak_size
            )
                
            # Calculate emittance
            emittance_results = self.analyzer.calculate_emittance(
                self.analyzer.clean_hole_data, self.analyzer.clean_hole_sizes, x0_positions, y0_positions
            )
                
            # Display spot detection
            self.display_spots(self.main_window.colormap_combo.currentText())
                
            # Display emittance results
            self.display_emittance_results(emittance_results)
                
            # Display phase space plots
            self.display_phase_space(self.main_window.colormap_combo.currentText())
                
            # Update contour plot (without spot overlay)
            self.display_contour_plot(self.main_window.colormap_combo.currentText(), False)
                
            # Update surface plot (without spot overlay)
            self.display_surface_plot(self.main_window.colormap_combo.currentText(), False)
                
            # Switch to spot detection tab
            self.results_tabs.setCurrentIndex(0)
                
            self.main_window.statusBar().showMessage("Beam spots analyzed successfully")
            return True
                
        except Exception as e:
            QMessageBox.critical(self.main_window, "Error", f"Error analyzing spots: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def display_spots(self, colormap='jet'):
        """Display detected spots on the image with subtler indicators"""
        if self.analyzer.cropped_image is None or not self.analyzer.hole_coordinates:
            return
            
        # Clear the canvas
        self.spots_canvas.fig.clear()
            
        # Create main axes
        self.spots_canvas.axes = self.spots_canvas.fig.add_subplot(111)
            
        # Create a custom spectral colormap for better visualization
        if colormap == 'jet':
            # Use a slightly improved jet variant
            cmap = cm.jet
        else:
            cmap = plt.get_cmap(colormap)
            
        # Apply logarithmic normalization for better contrast
        norm = matplotlib.colors.LogNorm(vmin=np.max([1, np.min(self.analyzer.cropped_image)]), 
                                       vmax=np.max(self.analyzer.cropped_image))
            
        # Display image with enhanced colormap
        img = self.spots_canvas.axes.imshow(self.analyzer.cropped_image, 
                                          cmap=cmap, 
                                          norm=norm,
                                          interpolation='bicubic')
            
        # Plot hole coordinates with more subtle visualization
        x_coords = [coord[0] for coord in self.analyzer.hole_coordinates]
        y_coords = [coord[1] for coord in self.analyzer.hole_coordinates]
            
        # Add smaller, more subtle circles around detected spots
        for x, y in zip(x_coords, y_coords):
            circle = plt.Circle((x, y), radius=5, fill=False, edgecolor='white', 
                               linestyle='-', linewidth=0.8, alpha=0.5)
            self.spots_canvas.axes.add_patch(circle)
                
        # Add smaller scatter points at the center
        self.spots_canvas.axes.scatter(x_coords, y_coords, c='yellow', marker='+', 
                                     s=30, linewidths=0.8, alpha=0.7)
            
        # Add title with spot count and analysis parameters
        spot_count = len(self.analyzer.hole_coordinates)
        title = f'Detected Beam Spots: {spot_count}\n'
        title += f'(α={self.analyzer.alpha:.2f}, threshold={self.analyzer.threshold:.2f})'
        self.spots_canvas.axes.set_title(title)
            
        # Add grid for easier reading of coordinates
        self.spots_canvas.axes.grid(False)
            
        # Add axis labels
        self.spots_canvas.axes.set_xlabel('X Position (pixels)')
        self.spots_canvas.axes.set_ylabel('Y Position (pixels)')
            
        # Add colorbar
        cbar = self.spots_canvas.fig.colorbar(img, ax=self.spots_canvas.axes)
        cbar.set_label('Intensity (log scale)')
            
        try:
            # Refresh canvas with tight_layout
            self.spots_canvas.fig.tight_layout()
        except:
            # Fall back to simple draw if tight_layout fails
            pass
                
        self.spots_canvas.draw()

    def display_emittance_results(self, results):
        """Display emittance calculation results in a table without highlighting"""
        if not results:
            return

        # Clear table
        self.emittance_table.setRowCount(0)

        # Add results to table
        row_idx = 0

        # X emittance parameters with formatting and units
        for param, value, unit in [
            ("<X>", results['x_bar'], "mm"),
            ("<X'>", results['xp_bar'], "mrad"),
            ("<X²>", results['x_bar_sq'], "mm²"),
            ("<X'²>", results['Xpi_sq'], "mrad²"),
            ("<XX'>", results['xxp'], "mm·mrad"),
            ("ε_x²", results['emit_x_sq'], "mm²·mrad²"),
            ("ε_x", results['emit_x'], "mm·mrad"),
            ("X_rms", results['x_rms'], "mm")
        ]:
            self.emittance_table.insertRow(row_idx)
            param_item = QTableWidgetItem(param)
            value_item = QTableWidgetItem(f"{value:.6f} {unit}")

            # No highlighting for ε_x
            self.emittance_table.setItem(row_idx, 0, param_item)
            self.emittance_table.setItem(row_idx, 1, value_item)
            row_idx += 1

        # Spacer row
        self.emittance_table.insertRow(row_idx)
        self.emittance_table.setItem(row_idx, 0, QTableWidgetItem(""))
        self.emittance_table.setItem(row_idx, 1, QTableWidgetItem(""))
        row_idx += 1

        # Y emittance parameters with formatting and units
        for param, value, unit in [
            ("<Y>", results['y_bar'], "mm"),
            ("<Y'>", results['yp_bar'], "mrad"),
            ("<Y²>", results['y_bar_sq'], "mm²"),
            ("<Y'²>", results['Ypi_sq'], "mrad²"),
            ("<YY'>", results['yyp'], "mm·mrad"),
            ("ε_y²", results['emit_y_sq'], "mm²·mrad²"),
            ("ε_y", results['emit_y'], "mm·mrad"),
            ("Y_rms", results['y_rms'], "mm")
        ]:
            self.emittance_table.insertRow(row_idx)
            param_item = QTableWidgetItem(param)
            value_item = QTableWidgetItem(f"{value:.6f} {unit}")

            # No highlighting for ε_y
            self.emittance_table.setItem(row_idx, 0, param_item)
            self.emittance_table.setItem(row_idx, 1, value_item)
            row_idx += 1

    def display_phase_space(self, colormap='jet'):
        """Display all six phase space projections as in the paper"""
        if not self.analyzer.emittance_results:
            return

        # Get data from results
        results = self.analyzer.emittance_results

        # Optionally reconstruct 4D phase space (uncomment if implemented)
        # phase_space = self.analyzer.reconstruct_4d_phase_space(results)
        # if phase_space is None:
        #    return

        # Extract raw data
        Xi_merge = results['Xi_merge']
        Xpi_merge = results['Xpi_merge']
        Pi_Xmerge = results['Pi_Xmerge']
        Yi_merge = results['Yi_merge']
        Ypi_merge = results['Ypi_merge']
        Pi_Ymerge = results['Pi_Ymerge']

        # Create a widget to replace existing content
        phase_space_content = QWidget()
        grid_layout = QGridLayout(phase_space_content)

        # Create all canvases
        self.xx_canvas = MatplotlibCanvas(None, width=5, height=4)
        self.yy_canvas = MatplotlibCanvas(None, width=5, height=4)
        self.xy_canvas = MatplotlibCanvas(None, width=5, height=4)
        self.xy_prime_canvas = MatplotlibCanvas(None, width=5, height=4)
        self.x_prime_y_canvas = MatplotlibCanvas(None, width=5, height=4)
        self.x_prime_y_prime_canvas = MatplotlibCanvas(None, width=5, height=4)

        # Create all toolbars
        self.xx_toolbar = NavigationToolbar(self.xx_canvas, None)
        self.yy_toolbar = NavigationToolbar(self.yy_canvas, None)
        self.xy_toolbar = NavigationToolbar(self.xy_canvas, None)
        self.xy_prime_toolbar = NavigationToolbar(self.xy_prime_canvas, None)
        self.x_prime_y_toolbar = NavigationToolbar(self.x_prime_y_canvas, None)
        self.x_prime_y_prime_toolbar = NavigationToolbar(self.x_prime_y_prime_canvas, None)

        # Add all widgets to layout
        grid_layout.addWidget(self.xx_toolbar, 0, 0)
        grid_layout.addWidget(self.xx_canvas, 1, 0)
        grid_layout.addWidget(self.yy_toolbar, 0, 1)
        grid_layout.addWidget(self.yy_canvas, 1, 1)

        grid_layout.addWidget(self.xy_toolbar, 2, 0)
        grid_layout.addWidget(self.xy_canvas, 3, 0)
        grid_layout.addWidget(self.xy_prime_toolbar, 2, 1)
        grid_layout.addWidget(self.xy_prime_canvas, 3, 1)

        grid_layout.addWidget(self.x_prime_y_toolbar, 4, 0)
        grid_layout.addWidget(self.x_prime_y_canvas, 5, 0)
        grid_layout.addWidget(self.x_prime_y_prime_toolbar, 4, 1)
        grid_layout.addWidget(self.x_prime_y_prime_canvas, 5, 1)

        # Replace old layout
        old_layout = self.phase_space_widget.layout()
        if old_layout is not None:
            # Remove old layout widgets
            while old_layout.count():
                item = old_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

            # Delete old layout
            QWidget().setLayout(old_layout)

        # Set new layout with content
        new_layout = QVBoxLayout(self.phase_space_widget)
        new_layout.addWidget(phase_space_content)

        # Create all six plots with the new density plotting style
        # Standard x-x' plot
        self.create_phase_density_plot(self.xx_canvas, "X' vs X Phase Space",
                                       Xi_merge, Xpi_merge, Pi_Xmerge,
                                       "X (mm)", "X' (mrad)",
                                       results['x_bar'], results['xp_bar'],
                                       results['emit_x'], results['x_rms'],
                                       colormap)

        # Standard y-y' plot
        self.create_phase_density_plot(self.yy_canvas, "Y' vs Y Phase Space",
                                       Yi_merge, Ypi_merge, Pi_Ymerge,
                                       "Y (mm)", "Y' (mrad)",
                                       results['y_bar'], results['yp_bar'],
                                       results['emit_y'], results['y_rms'],
                                       colormap)

        # x-y plot
        self.create_phase_density_plot(self.xy_canvas, "Y vs X Phase Space",
                                       Xi_merge, Yi_merge, Pi_Xmerge,
                                       "X (mm)", "Y (mm)",
                                       results['x_bar'], results['y_bar'],
                                       np.sqrt(abs(results['x_bar_sq'] * results['y_bar_sq'])),
                                       np.sqrt(results['x_bar_sq'] + results['y_bar_sq']),
                                       colormap, draw_ellipses=False)

        # x-y' plot
        self.create_phase_density_plot(self.xy_prime_canvas, "Y' vs X Phase Space",
                                       Xi_merge, Ypi_merge, Pi_Xmerge,
                                       "X (mm)", "Y' (mrad)",
                                       results['x_bar'], results['yp_bar'],
                                       np.sqrt(abs(results['x_bar_sq'] * results['Ypi_sq'])),
                                       np.sqrt(results['x_bar_sq'] + results['Ypi_sq']),
                                       colormap, draw_ellipses=False)

        # x'-y plot
        self.create_phase_density_plot(self.x_prime_y_canvas, "Y vs X' Phase Space",
                                       Xpi_merge, Yi_merge, Pi_Xmerge,
                                       "X' (mrad)", "Y (mm)",
                                       results['xp_bar'], results['y_bar'],
                                       np.sqrt(abs(results['Xpi_sq'] * results['y_bar_sq'])),
                                       np.sqrt(results['Xpi_sq'] + results['y_bar_sq']),
                                       colormap, draw_ellipses=False)

        # x'-y' plot
        self.create_phase_density_plot(self.x_prime_y_prime_canvas, "Y' vs X' Phase Space",
                                       Xpi_merge, Ypi_merge, Pi_Xmerge,
                                       "X' (mrad)", "Y' (mrad)",
                                       results['xp_bar'], results['yp_bar'],
                                       np.sqrt(abs(results['Xpi_sq'] * results['Ypi_sq'])),
                                       np.sqrt(results['Xpi_sq'] + results['Ypi_sq']),
                                       colormap, draw_ellipses=False)

    def create_phase_density_plot(self, canvas, title, x_data, y_data, intensity_data,
                                  x_label, y_label, x_mean, y_mean, emittance, rms,
                                  colormap='jet', draw_ellipses=True):
        """Create a phase space density plot similar to those in the paper"""
        # Clear the canvas
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)

        # Ensure all arrays have the same length
        min_len = min(len(x_data), len(y_data), len(intensity_data))
        x_data = x_data[:min_len]
        y_data = y_data[:min_len]
        intensity_data = intensity_data[:min_len]

        # Skip if not enough data points
        if min_len < 10:
            ax.text(0.5, 0.5, "Insufficient data points for visualization",
                    ha='center', va='center', transform=ax.transAxes)
            canvas.draw()
            return

        # Create 2D histogram with weighted values (this represents phase space density)
        nbins = 50  # Higher resolution than before

        # Calculate appropriate range with margin
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        x_margin = 0.1 * (x_max - x_min)
        y_margin = 0.1 * (y_max - y_min)

        try:
            # Create 2D histogram with weighted values
            hist, x_edges, y_edges = np.histogram2d(
                x_data, y_data,
                bins=nbins,
                range=[[x_min - x_margin, x_max + x_margin], [y_min - y_margin, y_max + y_margin]],
                weights=intensity_data
            )

            # Normalize histogram by bin counts
            counts, _, _ = np.histogram2d(
                x_data, y_data,
                bins=nbins,
                range=[[x_min - x_margin, x_max + x_margin], [y_min - y_margin, y_max + y_margin]]
            )

            # Avoid division by zero
            hist_norm = np.divide(hist, counts, out=np.zeros_like(hist), where=counts > 0)

            # Get colormap
            cmap = plt.get_cmap(colormap)

            # Create a pcolormesh plot like in the paper with interpolation
            mesh = ax.pcolormesh(x_edges, y_edges, hist_norm.T,
                                 cmap=cmap, shading='auto')

            # Add contour lines for better visualization (similar to paper)
            if np.max(hist_norm) > 0:  # Only add contours if we have meaningful data
                contour_levels = np.linspace(0, np.max(hist_norm) * 0.95, 8)
                if len(contour_levels) > 2:  # Only add contours if we have enough levels
                    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
                    X, Y = np.meshgrid(x_centers, y_centers)
                    contours = ax.contour(X, Y, hist_norm.T,
                                          levels=contour_levels,
                                          colors='white', linewidths=0.5, alpha=0.7)

            # Draw RMS ellipses if requested
            if draw_ellipses:
                # 1-sigma ellipse (red)
                theta = np.linspace(0, 2 * np.pi, 100)
                x = x_mean + np.sqrt(emittance) * np.cos(theta)
                y = y_mean + np.sqrt(emittance) * np.sin(theta)
                ax.plot(x, y, 'r-', linewidth=1.5, alpha=0.8)

            # Add colorbar
            cbar = canvas.fig.colorbar(mesh, ax=ax)
            cbar.set_label('Phase Space Density')

        except Exception as e:
            # Fall back to scatter plot if histogram fails
            print(f"Error creating histogram plot: {e}")
            scatter = ax.scatter(x_data, y_data, c=intensity_data,
                                 cmap=colormap, alpha=0.7, s=5)

            # Add colorbar for scatter plot
            cbar = canvas.fig.colorbar(scatter, ax=ax)
            cbar.set_label('Intensity')

        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        # Store axes reference
        canvas.axes = ax

        # Adjust layout and draw
        try:
            canvas.fig.tight_layout()
        except:
            pass

        canvas.draw()
    def clear_layout(self, layout):
        """Clear all widgets from a layout"""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())

    def create_phase_plot(self, canvas, title, x_data, y_data, intensity_data,
                          x_label, y_label, x_mean, y_mean, emittance, rms,
                          colormap='jet', draw_ellipses=False):
        """Create a phase space plot with academic styling using rectangular bins"""
        # Clear the canvas
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)

        # Ensure all arrays have the same length by trimming to the minimum length
        min_len = min(len(x_data), len(y_data), len(intensity_data))
        x_data = x_data[:min_len]
        y_data = y_data[:min_len]
        intensity_data = intensity_data[:min_len]

        # Create a colormap for the data
        cmap = plt.get_cmap(colormap)

        # Use histogram2d for rectangular bins
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)

        # Add small margin
        x_margin = 0.05 * (x_max - x_min)
        y_margin = 0.05 * (y_max - y_min)

        # Set number of bins
        nbins = 200

        # Create 2D histogram with weighted values
        hist, x_edges, y_edges = np.histogram2d(
            x_data, y_data,
            bins=nbins,
            range=[[x_min - x_margin, x_max + x_margin], [y_min - y_margin, y_max + y_margin]],
            weights=intensity_data
        )

        # Normalize histogram by the bin counts
        counts, _, _ = np.histogram2d(
            x_data, y_data,
            bins=nbins,
            range=[[x_min - x_margin, x_max + x_margin], [y_min - y_margin, y_max + y_margin]]
        )

        # Avoid division by zero
        hist_norm = np.divide(hist, counts, out=np.zeros_like(hist), where=counts > 0)

        # Plot the histogram as an image
        im = ax.imshow(
            hist_norm.T,  # Transpose for correct orientation
            origin='lower',
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            aspect='auto',
            cmap=cmap,
            interpolation='nearest'
        )

        # Set limits with small margin
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.5, color='gray')

        # Add RMS ellipses for normal phase space plots
        if draw_ellipses and self.analyzer.emittance_results:
            try:
                # For x-x' or y-y' plots, calculate and draw ellipses
                if ('X\'' in title and 'X Phase' in title) or ('Y\'' in title and 'Y Phase' in title):
                    self.add_rms_ellipses_simple(ax, x_mean, y_mean, emittance)
            except Exception as e:
                print(f"Error drawing ellipses: {e}")

        # Add info text in a small box in the corner - only for x-x' and y-y' plots
        if draw_ellipses:
            try:
                if 'X\'' in title and 'X Phase' in title:
                    # Calculate Twiss parameters for x-x'
                    twiss_data = self.analyzer.emittance_results
                    alpha_x = -twiss_data['xxp'] / np.sqrt(abs(twiss_data['x_bar_sq'] * twiss_data['Xpi_sq']))
                    beta_x = twiss_data['x_bar_sq'] / np.sqrt(abs(twiss_data['emit_x_sq']))
                    gamma_x = twiss_data['Xpi_sq'] / np.sqrt(abs(twiss_data['emit_x_sq']))

                    info_text = f"Emittance (ε): {emittance:.4f} mm·mrad\n"
                    info_text += f"RMS Size: {rms:.4f} mm\n"
                    info_text += f"Twiss Parameters: α={alpha_x:.4f}, β={beta_x:.4f}, γ={gamma_x:.4f}"

                    # Place text box at the bottom of the plot
                    ax.text(0.5, 0.02, info_text,
                            transform=ax.transAxes, fontsize=8,
                            horizontalalignment='center', verticalalignment='bottom',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

                elif 'Y\'' in title and 'Y Phase' in title:
                    # Calculate Twiss parameters for y-y'
                    twiss_data = self.analyzer.emittance_results
                    alpha_y = -twiss_data['yyp'] / np.sqrt(abs(twiss_data['y_bar_sq'] * twiss_data['Ypi_sq']))
                    beta_y = twiss_data['y_bar_sq'] / np.sqrt(abs(twiss_data['emit_y_sq']))
                    gamma_y = twiss_data['Ypi_sq'] / np.sqrt(abs(twiss_data['emit_y_sq']))

                    info_text = f"Emittance (ε): {emittance:.4f} mm·mrad\n"
                    info_text += f"RMS Size: {rms:.4f} mm\n"
                    info_text += f"Twiss Parameters: α={alpha_y:.4f}, β={beta_y:.4f}, γ={gamma_y:.4f}"

                    # Place text box at the bottom of the plot
                    ax.text(0.5, 0.02, info_text,
                            transform=ax.transAxes, fontsize=8,
                            horizontalalignment='center', verticalalignment='bottom',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            except Exception as e:
                print(f"Error adding info text: {e}")

        # Add axis labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        # Add colorbar
        cbar = canvas.fig.colorbar(im, ax=ax)
        cbar.set_label('Intensity')

        # Store axes reference
        canvas.axes = ax

        # Adjust layout and draw
        try:
            canvas.fig.tight_layout()
        except:
            pass

        canvas.draw()

    def add_rms_ellipses_simple(self, ax, x_mean, y_mean, emittance):
        """Add simple RMS ellipses to phase space plots with proper coloring"""
        # Use Twiss parameters if available
        if self.analyzer.emittance_results:
            results = self.analyzer.emittance_results

            # Check if this is an x-x' or y-y' plot
            is_x_plot = 'xx' in ax.get_xlabel().lower() or 'x' in ax.get_ylabel().lower()

            try:
                if is_x_plot:
                    alpha = -results['xxp'] / np.sqrt(abs(results['x_bar_sq'] * results['Xpi_sq']))
                    beta = results['x_bar_sq'] / np.sqrt(abs(results['emit_x_sq']))
                    gamma = results['Xpi_sq'] / np.sqrt(abs(results['emit_x_sq']))
                else:
                    alpha = -results['yyp'] / np.sqrt(abs(results['y_bar_sq'] * results['Ypi_sq']))
                    beta = results['y_bar_sq'] / np.sqrt(abs(results['emit_y_sq']))
                    gamma = results['Ypi_sq'] / np.sqrt(abs(results['emit_y_sq']))

                # Draw ellipses for 1σ, 2σ, and 3σ
                for n_sigma, style in [(1, {'color': 'red', 'linestyle': '-', 'linewidth': 1.5, 'label': '1σ'}),
                                       (2, {'color': 'orange', 'linestyle': '--', 'linewidth': 1.2, 'label': '2σ'}),
                                       (3, {'color': 'gold', 'linestyle': ':', 'linewidth': 1.0, 'label': '3σ'})]:
                    theta = np.linspace(0, 2 * np.pi, 100)
                    area = n_sigma * np.sqrt(emittance)

                    # Calculate ellipse coordinates using Twiss parameters
                    x = x_mean + area * np.sqrt(beta) * np.cos(theta)
                    y = y_mean + area * (alpha * np.cos(theta) + np.sqrt(gamma) * np.sin(theta)) / np.sqrt(beta)

                    # Plot the ellipse
                    ax.plot(x, y, **style)

                # Add legend
                ax.legend(loc='upper right', fontsize=8)

            except Exception as e:
                print(f"Error drawing Twiss ellipses: {e}")
                # Fallback to simple circles if Twiss calculation fails
                for n_sigma, style in [(1, {'color': 'red', 'linestyle': '-', 'linewidth': 1.5, 'label': '1σ'}),
                                       (2, {'color': 'orange', 'linestyle': '--', 'linewidth': 1.2, 'label': '2σ'}),
                                       (3, {'color': 'gold', 'linestyle': ':', 'linewidth': 1.0, 'label': '3σ'})]:
                    theta = np.linspace(0, 2 * np.pi, 100)
                    radius = n_sigma * np.sqrt(emittance)
                    x = x_mean + radius * np.cos(theta)
                    y = y_mean + radius * np.sin(theta)
                    ax.plot(x, y, **style)
                ax.legend(loc='upper right', fontsize=8)
    def setup_phase_plot(self, canvas, title, x_data, y_data, intensity_data,
                         x_label, y_label, x_mean, y_mean, emittance, rms,
                         colormap='jet'):
        """Create a phase space plot in academic style without profiles"""
        # Clear the canvas
        canvas.fig.clear()

        # Create a single axes for the main plot (no integrated profiles)
        ax_main = canvas.fig.add_subplot(111)

        # Use scatter plot without log scale, similar to the academic paper style
        scatter = ax_main.scatter(x_data, y_data, c=intensity_data,
                                  cmap=colormap, alpha=0.7, s=5)

        # Set limits with margin
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        x_margin = 0.05 * (x_max - x_min)
        y_margin = 0.05 * (y_max - y_min)

        ax_main.set_xlim(x_min - x_margin, x_max + x_margin)
        ax_main.set_ylim(y_min - y_margin, y_max + y_margin)

        # Add grid lines (subtle)
        ax_main.grid(True, linestyle='--', alpha=0.6)

        # Add RMS ellipses to the phase space plot
        self.add_rms_ellipses(ax_main, x_mean, y_mean, emittance)

        # Calculate Twiss parameters
        if self.analyzer.emittance_results:
            twiss_data = self.analyzer.emittance_results

            try:
                twiss_alpha = -twiss_data['xxp'] / np.sqrt(abs(twiss_data['x_bar_sq'] * twiss_data['Xpi_sq']))
                twiss_beta = twiss_data['x_bar_sq'] / np.sqrt(abs(twiss_data['emit_x_sq']))
                twiss_gamma = twiss_data['Xpi_sq'] / np.sqrt(abs(twiss_data['emit_x_sq']))

                info_text = f"Emittance (ε): {emittance:.4f} mm·mrad\n"
                info_text += f"RMS Size: {rms:.4f} mm\n"
                info_text += f"Twiss Parameters: α={twiss_alpha:.4f}, β={twiss_beta:.4f}, γ={twiss_gamma:.4f}"
            except:
                # Fallback if calculation fails
                info_text = f"Emittance (ε): {emittance:.4f} mm·mrad\n"
                info_text += f"RMS Size: {rms:.4f} mm"
        else:
            # Fallback if no twiss parameters are available
            info_text = f"Emittance (ε): {emittance:.4f} mm·mrad\n"
            info_text += f"RMS Size: {rms:.4f} mm"

        # Add info text in a small box in the corner
        ax_main.text(0.98, 0.02, info_text,
                     transform=ax_main.transAxes, fontsize=9,
                     horizontalalignment='right', verticalalignment='bottom',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # Add title and labels with academic style
        ax_main.set_title(title, fontsize=12)
        ax_main.set_xlabel(x_label)
        ax_main.set_ylabel(y_label)

        # Add colorbar
        cbar = canvas.fig.colorbar(scatter, ax=ax_main)
        cbar.set_label('Intensity')

        # Store main axes reference
        canvas.axes = ax_main

        try:
            # Adjust layout
            canvas.fig.tight_layout()
        except:
            # Fall back to simple draw if adjustment fails
            pass

        canvas.draw()

    def setup_integrated_phase_plot(self, canvas, title, x_data, y_data, intensity_data,
                                  x_label, y_label, x_mean, y_mean, emittance, rms,
                                  colormap='jet'):
        """Create a phase space plot with profiles integrated on axes"""
        # Clear the canvas
        canvas.fig.clear()
        
        # Create a figure with two main plotting areas for the phase plot and info panel
        gs = canvas.fig.add_gridspec(3, 3, height_ratios=[1, 5, 1], width_ratios=[1, 5, 1])
        
        # Main phase space plot
        ax_main = canvas.fig.add_subplot(gs[1, 1])

        # Info panel at the bottom - without overlapping the main plot
        ax_info = canvas.fig.add_subplot(gs[2, 1])
        ax_info.axis('off')  # No axes for info panel
        
        # Use hexbin for better density visualization of scattered data
        hexbin = ax_main.hexbin(x_data, y_data, C=intensity_data, 
                             gridsize=40, cmap=colormap, 
                             mincnt=1, bins='log')
        
        # Calculate and set limits with margin
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        x_margin = 0.05 * (x_max - x_min)
        y_margin = 0.05 * (y_max - y_min)
        
        ax_main.set_xlim(x_min - x_margin, x_max + x_margin)
        ax_main.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Add grid lines
        ax_main.grid(True, linestyle='--', alpha=0.6)

        # Add multiple RMS ellipses to the phase space plot
        self.add_rms_ellipses(ax_main, x_mean, y_mean, emittance)
        
        # Calculate Twiss parameters from the provided data
        if self.analyzer.emittance_results:
            twiss_data = self.analyzer.emittance_results
            
            # Fix twiss parameter calculation to avoid division by zero
            try:
                twiss_alpha = -twiss_data['xxp'] / np.sqrt(abs(twiss_data['x_bar_sq'] * twiss_data['Xpi_sq']))
                twiss_beta = twiss_data['x_bar_sq'] / np.sqrt(abs(twiss_data['emit_x_sq']))
                twiss_gamma = twiss_data['Xpi_sq'] / np.sqrt(abs(twiss_data['emit_x_sq']))
                
                info_text = f"Emittance (ε): {emittance:.4f} mm·mrad\n"
                info_text += f"RMS Size: {rms:.4f} mm\n"
                info_text += f"Twiss Parameters: α={twiss_alpha:.4f}, β={twiss_beta:.4f}, γ={twiss_gamma:.4f}"
            except:
                # Fallback if calculation fails
                info_text = f"Emittance (ε): {emittance:.4f} mm·mrad\n"
                info_text += f"RMS Size: {rms:.4f} mm"
        else:
            # Fallback if no twiss parameters are available
            info_text = f"Emittance (ε): {emittance:.4f} mm·mrad\n"
            info_text += f"RMS Size: {rms:.4f} mm"
        
        # Add text info in a box at the bottom, outside the plot area
        ax_info.text(0.5, 0.5, info_text, 
                   horizontalalignment='center', verticalalignment='center', 
                   transform=ax_info.transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Add a title at the top
        canvas.fig.suptitle(title, fontsize=12, fontweight='bold')
        
        # Add colorbar on the right side
        cbar_ax = canvas.fig.add_subplot(gs[1, 0])
        cbar = canvas.fig.colorbar(hexbin, cax=cbar_ax)
        cbar.set_label('Intensity (log scale)')
        
        # Set main plot labels
        ax_main.set_xlabel(x_label)
        ax_main.set_ylabel(y_label)
        
        # Store main axes reference
        canvas.axes = ax_main
        
        try:
            # Adjust layout - using subplots_adjust for more control
            canvas.fig.subplots_adjust(wspace=0.25, hspace=0.25)
        except:
            # Fall back to simple draw if adjustment fails
            pass
            
        canvas.draw()

    def add_rms_ellipses(self, ax, x_mean, y_mean, emittance):
        """Add multiple RMS ellipses to the phase space plot - academic style"""
        # Calculate Twiss parameters
        if self.analyzer.emittance_results:
            results = self.analyzer.emittance_results

            try:
                alpha = -results['xxp'] / np.sqrt(abs(results['x_bar_sq'] * results['Xpi_sq']))
                beta = results['x_bar_sq'] / np.sqrt(abs(results['emit_x_sq']))
                gamma = results['Xpi_sq'] / np.sqrt(abs(results['emit_x_sq']))

                # Draw three RMS ellipses for 1-sigma, 2-sigma, and 3-sigma
                for n, (color, style, width) in enumerate([
                    ('red', '-', 1.5),  # 1-sigma
                    ('orange', '--', 1.2),  # 2-sigma
                    ('gold', ':', 1.0)  # 3-sigma
                ]):
                    n_sigma = n + 1
                    self.add_twiss_ellipse(ax, x_mean, y_mean, emittance,
                                           alpha, beta, gamma,
                                           n_sigma=n_sigma,
                                           color=color, linestyle=style, linewidth=width,
                                           label=f'{n_sigma}σ')

                # Add legend in upper right
                ax.legend(loc='upper right', fontsize=8)
            except:
                # If calculation fails, just draw a simple circle
                theta = np.linspace(0, 2 * np.pi, 100)
                radius = np.sqrt(emittance)
                x = x_mean + radius * np.cos(theta)
                y = y_mean + radius * np.sin(theta)
                ax.plot(x, y, 'r-', linewidth=1.5, alpha=0.8, label='1σ')
                ax.legend(loc='upper right', fontsize=8)
    def add_twiss_ellipse(self, ax, x_mean, y_mean, emittance, alpha, beta, gamma, 
                         n_sigma=1, color='red', linestyle='-', linewidth=1.5, label=None):
        """Add a properly shaped phase space ellipse using Twiss parameters"""
        # Create ellipse points using parametric equation
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Scale by n_sigma
        area = n_sigma * np.sqrt(emittance)
        
        # Ellipse using Twiss parameters
        x = x_mean + area * np.sqrt(beta) * np.cos(theta)
        y = y_mean + area * (alpha * np.cos(theta) + np.sqrt(gamma) * np.sin(theta)) / np.sqrt(beta)
        
        # Plot the ellipse
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.8, label=label)

    def display_contour_plot(self, colormap='jet', show_spots=False):
        """Display contour plot with properly positioned profiles on the axes"""
        if not hasattr(self.main_window, 'contour_data') or self.main_window.contour_data is None:
            if self.analyzer.cropped_image is None:
                return

            # Store data for later use
            self.main_window.contour_data = {
                'image': self.analyzer.cropped_image,
                'x_profile': self.analyzer.x_profile,
                'y_profile': self.analyzer.y_profile
            }

        # Get the data
        image_data = self.main_window.contour_data['image']

        # Clear the canvas
        self.contour_canvas.fig.clear()

        # Create a single axes for the main plot
        ax_main = self.contour_canvas.fig.add_subplot(111)

        # Apply mild Gaussian smoothing for better contours
        smoothed_data = gaussian_filter(image_data, sigma=1.0)

        # Create coordinate grids
        height, width = image_data.shape
        y_grid, x_grid = np.mgrid[0:height, 0:width]

        # Get colormap
        cmap = plt.get_cmap(colormap)

        # Calculate contour levels with linear spacing
        levels = 20

        # Create filled contour plot with contour lines
        contour_filled = ax_main.contourf(x_grid, y_grid, smoothed_data,
                                          levels=levels, cmap=cmap)
        contour_lines = ax_main.contour(x_grid, y_grid, smoothed_data,
                                        levels=levels, colors='black',
                                        linewidths=0.5, alpha=0.3)

        # Add X profile along bottom axis (x-axis)
        if self.main_window.contour_data['x_profile'] is not None:
            x_data = np.arange(len(self.main_window.contour_data['x_profile']))
            x_profile = self.main_window.contour_data['x_profile']

            # Scale profile for better visualization
            max_height = height * 0.15
            x_profile_scaled = x_profile / np.max(x_profile) * max_height if np.max(x_profile) > 0 else x_profile

            # Plot at the bottom of the image using default matplotlib colors
            ax_main.plot(x_data, height - x_profile_scaled)
            ax_main.fill_between(x_data, height, height - x_profile_scaled, alpha=0.3)

        # Add Y profile along left side (y-axis)
        if self.main_window.contour_data['y_profile'] is not None:
            y_data = np.arange(len(self.main_window.contour_data['y_profile']))
            y_profile = self.main_window.contour_data['y_profile']

            # Scale profile for better visualization
            max_width = width * 0.15
            y_profile_scaled = y_profile / np.max(y_profile) * max_width if np.max(y_profile) > 0 else y_profile

            # Plot on the left side of the image using default matplotlib colors
            ax_main.plot(y_profile_scaled, y_data)
            ax_main.fill_betweenx(y_data, 0, y_profile_scaled, alpha=0.3)

        # Add labels and title
        ax_main.set_xlabel('X Position (pixels)')
        ax_main.set_ylabel('Y Position (pixels)')
        ax_main.set_title('Beam Intensity Contour Map', fontsize=12)

        # Add colorbar
        cbar = self.contour_canvas.fig.colorbar(contour_filled, ax=ax_main)
        cbar.set_label('Intensity')

        # Set reference to main axes
        self.contour_canvas.axes = ax_main

        try:
            # Adjust layout
            self.contour_canvas.fig.tight_layout()
        except:
            # Fall back to simple draw without layout adjustment
            pass

        self.contour_canvas.draw()
    def display_surface_plot(self, colormap='jet', show_spots=False):
        """Display 3D surface plot of beam intensity"""
        if not hasattr(self.main_window, 'contour_data') or self.main_window.contour_data is None:
            if self.analyzer.cropped_image is None:
                return
                
            # Store data for later use
            self.main_window.contour_data = {
                'image': self.analyzer.cropped_image,
                'x_profile': self.analyzer.x_profile,
                'y_profile': self.analyzer.y_profile
            }
            
        # Get the data
        image_data = self.main_window.contour_data['image']
        
        # Clear the canvas and create 3D axes
        self.surface_canvas.fig.clear()
        ax = self.surface_canvas.fig.add_subplot(111, projection='3d')
        
        # Apply mild Gaussian smoothing for better surface
        smoothed_data = gaussian_filter(image_data, sigma=1.0)
        
        # Create coordinate grids
        height, width = smoothed_data.shape
        y_grid, x_grid = np.mgrid[0:height, 0:width]
        
        # Downsample the grid for better performance
        # Use every 4th point to reduce computation load
        stride = 4
        
        # Get colormap
        if colormap == 'jet':
            cmap = plt.get_cmap('jet')
        else:
            cmap = plt.get_cmap(colormap)
        
        # Create surface plot with shading
        surf = ax.plot_surface(
            x_grid[::stride, ::stride],
            y_grid[::stride, ::stride],
            smoothed_data[::stride, ::stride],
            cmap=cmap,
            linewidth=0,
            antialiased=True,
            rstride=1,
            cstride=1,
            alpha=0.8
        )
        
        # Add contour lines projected on bottom plane for reference
        offset_z = np.min(smoothed_data) - 0.1 * (np.max(smoothed_data) - np.min(smoothed_data))
        contour = ax.contourf(
            x_grid[::stride, ::stride],
            y_grid[::stride, ::stride],
            smoothed_data[::stride, ::stride],
            zdir='z',
            offset=offset_z,
            cmap=cmap,
            alpha=0.5,
            levels=10
        )
        
        # Set axis limits
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_zlim(offset_z, np.max(smoothed_data) * 1.1)
        
        # Set labels and title
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_zlabel('Intensity')
        ax.set_title('Beam Intensity Surface Plot', fontsize=12, fontweight='bold')
        
        # Set optimal viewing angle
        ax.view_init(elev=30, azim=45)
        
        # Add colorbar
        cbar = self.surface_canvas.fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1)
        cbar.set_label('Intensity')
        
        # Set reference to surface axes
        self.surface_canvas.axes = ax
        
        try:
            # Adjust layout without using tight_layout
            self.surface_canvas.fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
        except:
            # Fall back to simple draw if adjustments fail
            pass
            
        self.surface_canvas.draw()
    
    def save_numerical_results(self, filepath):
        """Save numerical results to CSV file"""
        if not self.analyzer.emittance_results:
            return
            
        try:
            import csv
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Parameter', 'Value', 'Unit'])
                
                # X emittance parameters
                writer.writerow(['', 'X Emittance Parameters', ''])
                for param, value, unit in [
                    ("<X>", self.analyzer.emittance_results['x_bar'], "mm"),
                    ("<X'>", self.analyzer.emittance_results['xp_bar'], "mrad"),
                    ("<X²>", self.analyzer.emittance_results['x_bar_sq'], "mm²"),
                    ("<X'²>", self.analyzer.emittance_results['Xpi_sq'], "mrad²"),
                    ("<XX'>", self.analyzer.emittance_results['xxp'], "mm·mrad"),
                    ("ε_x²", self.analyzer.emittance_results['emit_x_sq'], "mm²·mrad²"),
                    ("ε_x", self.analyzer.emittance_results['emit_x'], "mm·mrad"),
                    ("X_rms", self.analyzer.emittance_results['x_rms'], "mm")
                ]:
                    writer.writerow([param, f"{value:.6f}", unit])
                
                # Y emittance parameters
                writer.writerow(['', '', ''])
                writer.writerow(['', 'Y Emittance Parameters', ''])
                for param, value, unit in [
                    ("<Y>", self.analyzer.emittance_results['y_bar'], "mm"),
                    ("<Y'>", self.analyzer.emittance_results['yp_bar'], "mrad"),
                    ("<Y²>", self.analyzer.emittance_results['y_bar_sq'], "mm²"),
                    ("<Y'²>", self.analyzer.emittance_results['Ypi_sq'], "mrad²"),
                    ("<YY'>", self.analyzer.emittance_results['yyp'], "mm·mrad"),
                    ("ε_y²", self.analyzer.emittance_results['emit_y_sq'], "mm²·mrad²"),
                    ("ε_y", self.analyzer.emittance_results['emit_y'], "mm·mrad"),
                    ("Y_rms", self.analyzer.emittance_results['y_rms'], "mm")
                ]:
                    writer.writerow([param, f"{value:.6f}", unit])
                    
                # Twiss parameters
                results = self.analyzer.emittance_results
                writer.writerow(['', '', ''])
                writer.writerow(['', 'Twiss Parameters', ''])
                
                # X Twiss parameters
                alpha_x = -results['xxp'] / np.sqrt(results['x_bar_sq'] * results['Xpi_sq'])
                beta_x = results['x_bar_sq'] / np.sqrt(results['emit_x_sq'])
                gamma_x = results['Xpi_sq'] / np.sqrt(results['emit_x_sq'])
                
                writer.writerow(['α_x', f"{alpha_x:.6f}", ''])
                writer.writerow(['β_x', f"{beta_x:.6f}", 'mm/mrad'])
                writer.writerow(['γ_x', f"{gamma_x:.6f}", 'mrad/mm'])
                
                # Y Twiss parameters
                alpha_y = -results['yyp'] / np.sqrt(results['y_bar_sq'] * results['Ypi_sq'])
                beta_y = results['y_bar_sq'] / np.sqrt(results['emit_y_sq'])
                gamma_y = results['Ypi_sq'] / np.sqrt(results['emit_y_sq'])
                
                writer.writerow(['α_y', f"{alpha_y:.6f}", ''])
                writer.writerow(['β_y', f"{beta_y:.6f}", 'mm/mrad'])
                writer.writerow(['γ_y', f"{gamma_y:.6f}", 'mrad/mm'])
                
            return True
                
        except Exception as e:
            QMessageBox.critical(self.main_window, "Error", f"Error saving numerical results: {str(e)}")
            return False