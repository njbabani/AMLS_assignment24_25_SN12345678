from sklearn.preprocessing import StandardScaler


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