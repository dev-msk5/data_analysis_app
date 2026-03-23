# src/datalab/gui/main_window.py

from PyQt6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QTableView,
    QVBoxLayout,
    QWidget,
    QPushButton
)

from datalab.core.dataset_controller import DatasetController
from datalab.gui.table_model import PandasTableModel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DataLab")

        self.controller = DatasetController()

        # UI elements
        self.table = QTableView()
        self.load_button = QPushButton("Load Dataset")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.table)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Connect button
        self.load_button.clicked.connect(self.load_dataset)

    def load_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Dataset",
            "",
            "Data Files (*.csv *.xlsx *.json)"
        )

        if file_path:
            self.controller.load(file_path)
            df = self.controller.get_data()

            model = PandasTableModel(df)
            self.table.setModel(model)
