from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV


def preprocess_for_svm_blood(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Prepares BloodMNIST dataset for SVM by flattening images and normalizing pixel values.

    Args:
        x_train, y_train, x_val, y_val, x_test, y_test (np.ndarray): Dataset images and labels.

    Returns:
        tuple: Preprocessed datasets for SVM.
    """
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_val_flat = x_val.reshape(x_val.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    scaler = StandardScaler()
    x_train_svm = scaler.fit_transform(x_train_flat)
    x_val_svm = scaler.transform(x_val_flat)
    x_test_svm = scaler.transform(x_test_flat)

    return x_train_svm, y_train.ravel(), x_val_svm, y_val.ravel(), x_test_svm, y_test.ravel()


def train_svm_B(x_train, y_train, use_gridsearch=False):
    """
    Trains an SVM model with optional GridSearchCV.

    Args:
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        use_gridsearch (bool, optional): Use GridSearchCV. Defaults to True.

    Returns:
        Trained SVM model.
    """
    if use_gridsearch:
        param_grid = {'C': [0.1, 1, 5, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf'], 'class_weight': [None, 'balanced']}
        svm = GridSearchCV(SVC(), param_grid, scoring='accuracy', cv=10, verbose=2, n_jobs=-1)
    else:
        svm = SVC(kernel='rbf', C=1, gamma='scale', class_weight=None)

    svm.fit(x_train, y_train)
    return svm


def evaluate_svm_B(svm, x_test, y_test):
    """
    Evaluates an SVM model on BloodMNIST dataset.

    Args:
        svm (SVC or GridSearchCV): Trained SVM model.
        x_test, y_test (np.ndarray): Test datasets.

    Returns:
        dict: Evaluation metrics.
    """
    y_test_pred = svm.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred, digits=4)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Classification Report:\n", report)

    return {"test_accuracy": test_accuracy, "classification_report": report, "y_test_pred": y_test_pred, "y_test": y_test}
