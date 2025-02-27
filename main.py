import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PySide6.QtWidgets import QSplitter
from PySide6.QtCore import Qt

# Import our custom modules
from analyzer import PepperPotAnalyzer
from left_panel import LeftPanel
from right_panel import RightPanel
from toolbar_menu import setup_menus_toolbar


class EmittanceAnalyzerGUI(QMainWindow):
    """Main window for the Pepper-pot Emittance Analyzer application"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Pepper-pot Emittance Analyzer")
        self.setMinimumSize(1200, 800)

        # Initialize analyzer
        self.analyzer = PepperPotAnalyzer()
        
        # Initialize contour data
        self.contour_data = None

        # Setup UI
        self.setup_ui()

        # Status message
        self.statusBar().showMessage("Ready")

    def setup_ui(self):
        """Setup the user interface"""
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QHBoxLayout(self.central_widget)

        # Create splitter for left and right panels
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.main_splitter)

        # Left panel - Images and controls
        self.left_panel = LeftPanel(self, self.analyzer)

        # Right panel - Analysis and results
        self.right_panel = RightPanel(self, self.analyzer)

        # Add panels to splitter
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.right_panel)
        self.main_splitter.setSizes([600, 600])  # Initial sizes

        # Setup menus and toolbars
        setup_menus_toolbar(self, self.left_panel, self.right_panel)


def main():
    app = QApplication(sys.argv)
    
    # Set application style for a more modern look
    app.setStyle('Fusion')
    
    window = EmittanceAnalyzerGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
