from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QComboBox, QLineEdit, QWidget, \
    QDesktopWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal
import main
import sys
import os


class Worker(QThread):
    finished = pyqtSignal(float, str)

    def __init__(self, dataset, training_rate, preprocess, n_bands, model, n_runs):
        super().__init__()
        self.dataset = dataset
        self.training_rate = training_rate
        self.preprocess = preprocess
        self.n_bands = n_bands
        self.model = model
        self.n_runs = n_runs

    def run(self):
        # Implement your func function here.
        # accuracy, image_path = func(self.n_runs, self.dataset, self.preprocess)
        tmp_folder = 'temp'
        hyperparams = {
                'dataset'      : self.dataset,
                'training_rate': self.training_rate,
                'preprocess'   : self.preprocess,
                'n_bands'      : self.n_bands,
                'model'        : self.model,
                'n_runs'       : self.n_runs,
                'res_folder'   : tmp_folder,
                'load_model'   : False,
                'patch_size'   : 0,
                'batch_size'   : 0,
        }
        if (self.model in ['nn', 'cnn1d', 'cnn2d']):
            hyperparams['n_runs']       = 200
            hyperparams['patch_size']   = 10
            hyperparams['batch_size']   = 1000
        try:
            run_results = main.main(
                    show_results_switch = False,
                    hyperparams = hyperparams)

            accuracy = run_results['accuracy']

            name = "pred-{dataset}_{training_rate}-{preprocess}_{n_bands_in}-{model}_runs{n_runs}_bsz{batch_size}_psz{patch_size}".format(**hyperparams)

            abs_res_folder = os.path.join(os.getcwd(), "..", tmp_folder, "{preprocess}_{model}".format(**hyperparams))
            if not os.path.exists(abs_res_folder):
                os.makedirs(abs_res_folder)
            self.finished.emit(accuracy, os.path.join(abs_res_folder, name + "_acy%.2f.png" % accuracy))
        except Exception as e:
            print(e)
            self.finished.emit(0, "error")


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create Widgets
        # dataset and training_rate
        self.dataset_selector = QComboBox(self)
        self.training_rate_input = QLineEdit(self)
        # preprocess and n_bands
        self.preprocess_selector = QComboBox(self)
        self.n_bands_input = QLineEdit(self)
        # model and n_runs
        self.model_selector = QComboBox(self)
        self.n_runs_input = QLineEdit(self)

        self.accuracy_output = QLabel(self)
        self.image_output = QLabel(self)
        self.submit_button = QPushButton('Submit', self)
        # self.save_path = QLineEdit(self)

        self.width, self.height = 400, 700
        self.resize(self.width, self.height)

        # Populate ComboBoxes
        self.dataset_selector.addItems(['IndianPines'])
        self.preprocess_selector.addItems(['pca', 'ica', 'lda', 'tsne'])
        self.model_selector.addItems(['svm', 'knn', 'nn', 'cnn1d', 'cnn2d'])

        # Set up Layout
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select dataset:"))
        layout.addWidget(self.dataset_selector)

        layout.addWidget(QLabel("Training_rate:"))
        layout.addWidget(self.training_rate_input)

        layout.addWidget(QLabel("Select preprocess:"))
        layout.addWidget(self.preprocess_selector)

        layout.addWidget(QLabel("Enter n_bands:"))
        layout.addWidget(self.n_bands_input)

        layout.addWidget(QLabel("Select model:"))
        layout.addWidget(self.model_selector)

        layout.addWidget(QLabel("Enter n_runs:"))
        layout.addWidget(self.n_runs_input)

        # submit button
        layout.addWidget(self.submit_button)
        layout.addWidget(QLabel("Accuracy:"))

        # output
        layout.addWidget(self.accuracy_output)

        layout.addWidget(QLabel("Result image:"))
        layout.addWidget(self.image_output)

        # Set up Central Widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Set up Signal-Slot
        self.submit_button.clicked.connect(self.run_func)

    def run_func(self):
        dataset = self.dataset_selector.currentText()
        training_rate = float(self.training_rate_input.text())

        preprocess = self.preprocess_selector.currentText()
        n_bands = int(self.n_bands_input.text())

        model = self.model_selector.currentText()
        n_runs = int(self.n_runs_input.text())

        self.worker = Worker(dataset, training_rate, preprocess, n_bands, model, n_runs)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

        # self.accuracy_output.setText(str(accuracy))
        # pixmap = QPixmap(image_path)
        # self.image_output.setPixmap(pixmap)

    def on_worker_finished(self, accuracy, image_path):
        self.accuracy_output.setText(str(accuracy))
        pixmap = QPixmap(image_path)
        self.image_output.setPixmap(pixmap)


app = QApplication(sys.argv)
win = Window()
win.show()
sys.exit(app.exec_())
