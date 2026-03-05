import sys
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QLabel,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QInputDialog,
    QMessageBox,
)
from PyQt5.QtCore import Qt

cfg = __import__(str(__name__).split(".")[0] + ".core.config", fromlist=[""])

METHODS = {
    "MAD": {"threshold": 3.5},
    # "PCA + Mahalanobis": {"n_components": 3, "threshold": 3.0},
    # "Mahalanobis": {"threshold": 3.0},
    "IsolationForest": {"contamination": 0.01},
    "LOF": {"n_neighbors": 20, "contamination": 0.01},
}


class PipelineDialog(QDialog):

    def __init__(self, removeOutliersCallback):
        super().__init__(None)
        self.removeOutliersCallback = removeOutliersCallback
        self.setWindowTitle("Pipeline Builder")
        self.resize(420, 450)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Pipeline"))

        self.pipeline_list = QListWidget()
        layout.addWidget(self.pipeline_list)

        btn_layout = QHBoxLayout()

        btn_add = QPushButton("Add")
        btn_add.clicked.connect(self.add_method)

        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(self.remove_method)

        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_remove)

        layout.addLayout(btn_layout)

        layout.addWidget(QLabel("Parameters"))

        self.param_form = QFormLayout()
        layout.addLayout(self.param_form)

        self.pipeline_list.itemClicked.connect(self.show_parameters)

        self.current_item = None
        self.param_widgets = {}

        self.run_btn = QPushButton("Run pipeline")
        self.run_btn.clicked.connect(self.execute_pipeline)
        layout.addWidget(self.run_btn)

    def add_method(self):

        method, ok = QInputDialog.getItem(
            self, "Add method", "Method:", list(METHODS.keys()), 0, False
        )

        if not ok:
            return

        item = QListWidgetItem(method)
        item.setData(Qt.UserRole, METHODS[method].copy())
        self.pipeline_list.addItem(item)

    def remove_method(self):
        for item in self.pipeline_list.selectedItems():
            self.pipeline_list.takeItem(self.pipeline_list.row(item))
        self.clear_params()

    def clear_params(self):
        while self.param_form.rowCount():
            self.param_form.removeRow(0)

    def show_parameters(self, item):

        self.current_item = item
        self.clear_params()

        params = item.data(Qt.UserRole)
        self.param_widgets = {}

        for key, val in params.items():

            if isinstance(val, int):
                widget = QSpinBox()
                widget.setMaximum(999999)
                widget.setValue(val)
            else:
                widget = QDoubleSpinBox()
                widget.setDecimals(6)
                widget.setSingleStep(0.1)
                widget.setValue(val)

            widget.valueChanged.connect(self.update_params)

            self.param_widgets[key] = widget
            self.param_form.addRow(QLabel(key), widget)

    def update_params(self):

        if self.current_item is None:
            return

        params = {}
        for key, widget in self.param_widgets.items():
            params[key] = widget.value()

        self.current_item.setData(Qt.UserRole, params)

    def get_pipeline(self):

        steps = []

        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            steps.append((item.text(), item.data(Qt.UserRole)))

        return steps

    def execute_pipeline(self):
        pipeline = self.get_pipeline()

        if len(pipeline) == 0:
            QMessageBox.warning(self, "Error", "Pipeline is empty")
            return

        self.removeOutliersCallback(pipeline)

        QMessageBox.information(self, "OK", "Pipeline finished")
        self.accept()
