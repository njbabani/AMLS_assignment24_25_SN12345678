# ELEC0134: Applied Machine Learning Systems Assignment

## File Roles

- **A/ANN_A.py**: Implements an Artificial Neural Network (ANN) for BreastMNIST (Task A).
- **A/CNN_A.py**: Implements a Convolutional Neural Network (CNN) for BreastMNIST (Task A).
- **A/SVM_A.py**: Implements and evaluates a Support Vector Machine (SVM) for BreastMNIST (Task A).
- **A/utils_A.py**: Contains utility functions (data augmentation, shuffling, plotting, etc.) for Task A.
- **B/ANN_B.py**: Implements an ANN for BloodMNIST (Task B).
- **B/CNN_B.py**: Implements a CNN for BloodMNIST (Task B).
- **B/SVM_B.py**: Implements and evaluates an SVM for BloodMNIST (Task B).
- **B/utils_B.py**: Contains utility functions for data preparation and plotting for Task B.
- **Datasets/**: Directory reserved for raw dataset files if required.
- **env/**: Contains environment configuration files (`environments.yml` for Conda and `requirements.txt` for pip) listing all necessary dependencies.
- **Results/**: Stores pre-trained models (saved as `.pth` for ANN/CNN and `.joblib` for SVM).
- **main.py**: Provides an interactive command-line interface to choose between Task A and Task B, select the model type (SVM, ANN, or CNN), and either train or load/evaluate the model.
- **LICENSE**: Contains the project license information.
- **.gitignore**: Lists files and directories to be ignored by Git (e.g., `__pycache__/`, virtual environment folders, temporary files, etc.).

## Dependencies

To run this project, you will need the following Python packages (among others):

- Python (>=3.6)
- **numpy**
- **scipy**
- **scikit-learn**
- **torch** (PyTorch)
- **torchvision**
- **matplotlib**
- **optuna**

You can install these packages via pip (using `requirements.txt`) or set up the Conda environment using `env/environments.yml`.

## Repo Structure

AMLS_assignment24_25/
├── A/
│   ├── ANN_A.py
│   ├── CNN_A.py
│   ├── SVM_A.py
│   ├── utils_A.py
│   └── __init__.py
├── B/
│   ├── ANN_B.py
│   ├── CNN_B.py
│   ├── SVM_B.py
│   ├── utils_B.py
│   └── __init__.py
├── Datasets/
├── env/
│   ├── environments.yml
│   └── requirements.txt
├── Results/
│   ├── best_ann_model_A.pth
│   ├── best_ann_model_B.pth
│   ├── best_cnn_model_A.pth
│   ├── best_cnn_model_B.pth
│   ├── best_svm_model_A.joblib
│   └── best_svm_model_B.joblib
├── LICENSE
├── main.py
├── README.md
└── .gitignore