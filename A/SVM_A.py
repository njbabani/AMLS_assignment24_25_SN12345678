from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


def preprocess_for_svm(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Preprocesses data for SVM by flattening images and scaling features.

    Args:
        x_train (np.ndarray): Training images.
        y_train (np.ndarray): Training labels.
        x_val (np.ndarray): Validation images.
        y_val (np.ndarray): Validation labels.
        x_test (np.ndarray): Test images.
        y_test (np.ndarray): Test labels.

    Returns:
        tuple: Preprocessed datasets:
            - x_train_svm (np.ndarray): Flattened and scaled training images.
            - y_train_svm (np.ndarray): Flattened training labels.
            - x_val_svm (np.ndarray): Flattened and scaled validation images.
            - y_val_svm (np.ndarray): Flattened validation labels.
            - x_test_svm (np.ndarray): Flattened and scaled test images.
            - y_test_svm (np.ndarray): Flattened test labels.
    """
    # Flatten images for SVM (reshape to 2D)
    x_train_flattened = x_train.reshape(x_train.shape[0], -1)
    x_val_flattened = x_val.reshape(x_val.shape[0], -1)
    x_test_flattened = x_test.reshape(x_test.shape[0], -1)

    # Convert 2D array to 1D array for labels
    y_train_svm = y_train.ravel()
    y_val_svm = y_val.ravel()
    y_test_svm = y_test.ravel()

    # Scale the features by normalizing pixel values
    scaler = StandardScaler()
    x_train_svm = scaler.fit_transform(x_train_flattened)
    x_val_svm = scaler.transform(x_val_flattened)
    x_test_svm = scaler.transform(x_test_flattened)

    return (
        x_train_svm, y_train_svm,
        x_val_svm, y_val_svm,
        x_test_svm, y_test_svm
    )


def train_svm_A(x_train, y_train, use_gridsearch=False):
    """
    Trains an SVM model with optional GridSearchCV.

    Args:
        x_train (np.ndarray): Training feature set.
        y_train (np.ndarray): Training labels.
        use_gridsearch (bool, optional): Whether to use GridSearchCV.

    Returns:
        svm: Trained SVM model.
    """
    if use_gridsearch:
        # Define hyperparameter grid
        param_grid = {
            'C': [0.1, 1, 5, 10, 20],  # Regularization parameter
            'gamma': ['scale', 'auto'],  # Kernel coefficient
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],  # Kernels
            'class_weight': [None, 'balanced']  # For class balancing
        }

        # GridSearchCV setup
        svm = GridSearchCV(
            estimator=SVC(),
            param_grid=param_grid,
            scoring='accuracy',
            cv=10,  # 10-fold cross-validation
            verbose=2,
            n_jobs=-1  # Use all available cores
        )
    else:
        # Train a basic SVM model
        svm = SVC(kernel='rbf', C=1, gamma='scale', class_weight=None)

    # Train model
    print("Training SVM Model...")
    svm.fit(x_train, y_train)

    return svm


def evaluate_svm_A(svm, x_val, y_val, x_test, y_test):
    """
    Evaluates a trained SVM model on validation and test sets.

    Args:
        svm (sklearn.svm.SVC or GridSearchCV): The trained SVM model.
        x_val (np.ndarray): Validation feature set.
        y_val (np.ndarray): Validation labels.
        x_test (np.ndarray): Test feature set.
        y_test (np.ndarray): Test labels.

    Returns:
        dict: A dictionary containing accuracy
              scores and classification report.
    """
    # Predict on validation set
    y_val_pred = svm.predict(x_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    # Predict on test set
    y_test_pred = svm.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Generate classification report
    report = classification_report(y_test, y_test_pred, digits=4)

    # Print results
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Classification Report:\n", report)

    return {
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "classification_report": report,
        "y_test_pred": y_test_pred,
        "y_test": y_test
    }
