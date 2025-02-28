from PySide6.QtGui import QAction
from PySide6.QtWidgets import QToolBar, QLabel, QComboBox, QMessageBox, QFileDialog, QInputDialog
from PySide6.QtCore import Qt
import os


def setup_menus_toolbar(main_window, left_panel, right_panel):
    """Setup application menus and toolbars"""

    # Define all handler functions first
    def load_image_dialog():
        filepath, _ = QFileDialog.getOpenFileName(
            main_window, "Open Image", "", "Image Files (*.bmp *.jpg *.png *.tif *.tiff)"
        )
        if filepath:
            left_panel.load_image(filepath)

    def load_background_dialog():
        filepath, _ = QFileDialog.getOpenFileName(
            main_window, "Open Background Image", "", "Image Files (*.bmp *.jpg *.png *.tif *.tiff)"
        )
        if filepath:
            left_panel.load_background(filepath)

    def save_results():
        if not main_window.analyzer.emittance_results:
            QMessageBox.warning(main_window, "Warning", "No results to save.")
            return

        # Get save filename
        filepath, selected_filter = QFileDialog.getSaveFileName(
            main_window, "Save Results", "", "Text Files (*.txt);;CSV Files (*.csv)"
        )

        if not filepath:
            return

        try:
            # Check if CSV format was selected
            if selected_filter == "CSV Files (*.csv)":
                right_panel.save_numerical_results(filepath)
                main_window.statusBar().showMessage(f"Results saved to {filepath}")
                return

            # Otherwise, save as text file
            with open(filepath, 'w') as f:
                # Write header with timestamp
                from datetime import datetime
                now = datetime.now()
                f.write(f"Pepper-pot Emittance Analysis Results - {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("====================================================================\n\n")

                # Add analysis parameters
                f.write("Analysis Parameters:\n")
                f.write("-------------------\n")
                f.write(f"Scaling (mm/pixel): {main_window.analyzer.scaling:.6f}\n")
                f.write(f"Offset: {main_window.analyzer.offset:.2f}\n")
                f.write(f"Distance L (mm): {main_window.analyzer.distance:.2f}\n")
                f.write(f"Intensity Threshold: {main_window.analyzer.threshold:.2f}\n")
                f.write(f"Peak Detection Alpha: {main_window.analyzer.alpha:.2f}\n")
                f.write(f"Number of detected spots: {len(main_window.analyzer.hole_coordinates)}\n\n")

                # X emittance parameters
                f.write("X Emittance Parameters:\n")
                f.write("----------------------\n")
                results = main_window.analyzer.emittance_results
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
                    f.write(f"{param}: {value:.6f} {unit}\n")

                f.write("\n")

                # Y emittance parameters
                f.write("Y Emittance Parameters:\n")
                f.write("----------------------\n")
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
                    f.write(f"{param}: {value:.6f} {unit}\n")

                # Add Twiss parameters
                f.write("\nTwiss Parameters:\n")
                f.write("---------------\n")

                # Calculate Twiss parameters safely
                try:
                    # X Twiss parameters
                    alpha_x = -results['xxp'] / (results['x_bar_sq'] * results['Xpi_sq']) ** 0.5
                    beta_x = results['x_bar_sq'] / (results['emit_x_sq']) ** 0.5
                    gamma_x = results['Xpi_sq'] / (results['emit_x_sq']) ** 0.5

                    f.write(f"α_x: {alpha_x:.6f}\n")
                    f.write(f"β_x: {beta_x:.6f} mm/mrad\n")
                    f.write(f"γ_x: {gamma_x:.6f} mrad/mm\n\n")

                    # Y Twiss parameters
                    alpha_y = -results['yyp'] / (results['y_bar_sq'] * results['Ypi_sq']) ** 0.5
                    beta_y = results['y_bar_sq'] / (results['emit_y_sq']) ** 0.5
                    gamma_y = results['Ypi_sq'] / (results['emit_y_sq']) ** 0.5

                    f.write(f"α_y: {alpha_y:.6f}\n")
                    f.write(f"β_y: {beta_y:.6f} mm/mrad\n")
                    f.write(f"γ_y: {gamma_y:.6f} mrad/mm\n")
                except:
                    f.write("Could not calculate Twiss parameters due to numerical issues.\n")

            main_window.statusBar().showMessage(f"Results saved to {filepath}")

        except Exception as e:
            QMessageBox.critical(main_window, "Error", f"Error saving results: {str(e)}")

    def export_plots():
        if not main_window.analyzer.cropped_image:
            QMessageBox.warning(main_window, "Warning", "No data to export. Please process images first.")
            return

        # Get directory to save files
        export_dir = QFileDialog.getExistingDirectory(main_window, "Select Export Directory")
        if not export_dir:
            return

        try:
            # Save processed image
            if main_window.analyzer.cropped_image is not None:
                left_panel.processed_image_canvas.fig.savefig(f"{export_dir}/processed_image.png", dpi=300,
                                                              bbox_inches='tight')

            # Save spots detection if available
            if main_window.analyzer.hole_coordinates:
                right_panel.spots_canvas.fig.savefig(f"{export_dir}/detected_spots.png", dpi=300, bbox_inches='tight')

            # Save phase space plots if emittance results available
            if main_window.analyzer.emittance_results:
                right_panel.xx_canvas.fig.savefig(f"{export_dir}/x_phase_space.png", dpi=300, bbox_inches='tight')
                right_panel.yy_canvas.fig.savefig(f"{export_dir}/y_phase_space.png", dpi=300, bbox_inches='tight')

                # Also save numerical results as CSV
                right_panel.save_numerical_results(f"{export_dir}/emittance_results.csv")

            QMessageBox.information(main_window, "Export Complete", f"Plots have been saved to:\n{export_dir}")

        except Exception as e:
            QMessageBox.critical(main_window, "Error", f"Error exporting plots: {str(e)}")

    def change_colormap(main_window, cmap_name):
        main_window.colormap_combo.setCurrentText(cmap_name)
        on_colormap_changed(main_window, left_panel, right_panel, cmap_name)

    def show_about(main_window):
        QMessageBox.about(
            main_window,
            "About Pepper-pot Emittance Analyzer",
            "Pepper-pot Emittance Analyzer\n\n"
            "A tool for analyzing particle accelerator beam spots and calculating emittance "
            "using the pepper-pot method.\n\n"
            "Based on 'Emittance Formula for Slits and Pepper-pot Measurement' by Min Zhang."
        )

    # Now that all functions are defined, create the menus and toolbar
    # Main menu bar
    menu_bar = main_window.menuBar()

    # File menu
    file_menu = menu_bar.addMenu("&File")

    load_image_action = QAction("Load Image", main_window)
    load_image_action.setShortcut("Ctrl+O")
    load_image_action.triggered.connect(load_image_dialog)
    file_menu.addAction(load_image_action)

    load_background_action = QAction("Load Background", main_window)
    load_background_action.setShortcut("Ctrl+B")
    load_background_action.triggered.connect(load_background_dialog)
    file_menu.addAction(load_background_action)

    file_menu.addSeparator()

    save_results_action = QAction("Save Results", main_window)
    save_results_action.setShortcut("Ctrl+S")
    save_results_action.triggered.connect(save_results)
    file_menu.addAction(save_results_action)

    export_plots_action = QAction("Export Plots", main_window)
    export_plots_action.setShortcut("Ctrl+E")
    export_plots_action.triggered.connect(export_plots)
    file_menu.addAction(export_plots_action)

    file_menu.addSeparator()

    exit_action = QAction("Exit", main_window)
    exit_action.setShortcut("Ctrl+Q")
    exit_action.triggered.connect(main_window.close)
    file_menu.addAction(exit_action)

    # Analysis menu
    analysis_menu = menu_bar.addMenu("&Analysis")

    process_action = QAction("Process Images", main_window)
    process_action.setShortcut("Ctrl+P")
    process_action.triggered.connect(lambda: left_panel.process_images())
    analysis_menu.addAction(process_action)

    analyze_action = QAction("Analyze Beam Spots", main_window)
    analyze_action.setShortcut("Ctrl+A")
    analyze_action.triggered.connect(lambda: right_panel.analyze_spots())
    analysis_menu.addAction(analyze_action)

    # View menu
    view_menu = menu_bar.addMenu("&View")

    colormap_menu = view_menu.addMenu("Colormap")

    # Add colormap options
    for cmap_name in ['jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo']:
        cmap_action = QAction(cmap_name.capitalize(), main_window)
        cmap_action.setData(cmap_name)
        cmap_action.triggered.connect(lambda checked, name=cmap_name: change_colormap(main_window, name))
        colormap_menu.addAction(cmap_action)

    # Help menu
    help_menu = menu_bar.addMenu("&Help")

    about_action = QAction("About", main_window)
    about_action.triggered.connect(lambda: show_about(main_window))
    help_menu.addAction(about_action)

    # Create toolbar
    toolbar = QToolBar("Main Toolbar")
    main_window.addToolBar(toolbar)

    # Add actions to toolbar (without duplicate buttons)
    toolbar.addAction(load_image_action)
    toolbar.addAction(load_background_action)
    toolbar.addAction(process_action)
    toolbar.addAction(analyze_action)

    # Add a separator
    toolbar.addSeparator()

    # Add colormap selection to toolbar
    main_window.colormap_combo = QComboBox()
    main_window.colormap_combo.addItems(['jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo'])
    main_window.colormap_combo.setCurrentText('jet')
    main_window.colormap_combo.currentTextChanged.connect(
        lambda name: on_colormap_changed(main_window, left_panel, right_panel, name)
    )

    toolbar.addWidget(QLabel("Colormap: "))
    toolbar.addWidget(main_window.colormap_combo)

    # Attach the functions to the main window
    main_window.load_image_dialog = load_image_dialog
    main_window.load_background_dialog = load_background_dialog
    main_window.save_results = save_results
    main_window.export_plots = export_plots
    main_window.change_colormap = change_colormap
    main_window.show_about = show_about


def on_colormap_changed(main_window, left_panel, right_panel, cmap_name):
    """Update all plots when colormap is changed"""
    # Redisplay processed image with current colormap
    if main_window.analyzer.cropped_image is not None:
        left_panel.display_image(
            left_panel.processed_image_canvas,
            main_window.analyzer.cropped_image,
            "Processed Image",
            left_panel.show_profiles_check.isChecked(),
            cmap_name
        )

    # Redisplay raw image with current colormap
    if main_window.analyzer.raw_image is not None:
        left_panel.display_image(
            left_panel.raw_image_canvas,
            main_window.analyzer.raw_image,
            "Raw Image",
            show_profiles=False,
            colormap=cmap_name
        )

    # Redisplay background image with current colormap
    if main_window.analyzer.background_image is not None:
        left_panel.display_image(
            left_panel.background_image_canvas,
            main_window.analyzer.background_image,
            "Background Image",
            show_profiles=False,
            colormap=cmap_name
        )

    # Redisplay spots with current colormap
    if main_window.analyzer.cropped_image is not None and main_window.analyzer.hole_coordinates:
        right_panel.display_spots(cmap_name)


    # Update phase space plots if emittance results exist
    if main_window.analyzer.emittance_results:
        right_panel.display_phase_space(cmap_name)
def setup_pat_menu(main_window, left_panel, right_panel):
    """Setup PAT-specific menu items"""
    pat_menu = main_window.menuBar().addMenu("&PAT Methods")

    # Process with PAT methods
    pat_process_action = QAction("Process Images (PAT Style)", main_window)
    pat_process_action.triggered.connect(lambda: left_panel.process_images_pat_style())
    pat_menu.addAction(pat_process_action)

    # Analyze with PAT methods
    pat_analyze_action = QAction("Analyze Beam (PAT Style)", main_window)
    pat_analyze_action.triggered.connect(lambda: right_panel.analyze_spots_pat())
    pat_menu.addAction(pat_analyze_action)

    # Generate particles
    generate_particles_action = QAction("Generate Particles (Monte Carlo)", main_window)
    generate_particles_action.triggered.connect(lambda: generate_particles(main_window))
    pat_menu.addAction(generate_particles_action)

    pat_menu.addSeparator()

    # Save particles
    save_particles_action = QAction("Save Generated Particles", main_window)
    save_particles_action.triggered.connect(lambda: save_particles(main_window))
    pat_menu.addAction(save_particles_action)

    # Helper functions
    def generate_particles(main_window):
        if not main_window.analyzer.emittance_results:
            QMessageBox.warning(main_window, "Warning", "No emittance results available.")
            return

        num_particles, ok = QInputDialog.getInt(
            main_window, "Number of Particles",
            "Enter number of particles to generate:", 10000, 100, 1000000, 1000
        )

        if ok:
            try:
                main_window.analyzer.particles = main_window.analyzer.generate_particles(num_particles)
                QMessageBox.information(
                    main_window, "Success",
                    f"Generated {len(main_window.analyzer.particles)} particles using PAT Monte Carlo method."
                )
            except Exception as e:
                QMessageBox.critical(main_window, "Error", f"Error generating particles: {str(e)}")

    def save_particles(main_window):
        if not hasattr(main_window.analyzer, 'particles') or not main_window.analyzer.particles:
            QMessageBox.warning(main_window, "Warning", "No particles to save. Generate particles first.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            main_window, "Save Particles", "", "Text Files (*.txt);;CSV Files (*.csv)"
        )

        if not filepath:
            return

        try:
            with open(filepath, 'w') as f:
                # PAT-style particle file format
                f.write(f"{len(main_window.analyzer.particles)}\n")
                f.write(" 0 0 0 0 0 2.4\n")  # Header line as in PAT format

                for p in main_window.analyzer.particles:
                    # Format matches PAT output
                    f.write(
                        f"{p['x'] / 10:.6f}\t{p['xp'] / 1000:.6f}\t{p['y'] / 10:.6f}\t{p['yp'] / 1000:.6f}  0  \t  2.4  \t\n")

            QMessageBox.information(main_window, "Success",
                                    f"Saved {len(main_window.analyzer.particles)} particles to {filepath}")
        except Exception as e:
            QMessageBox.critical(main_window, "Error", f"Error saving particles: {str(e)}")