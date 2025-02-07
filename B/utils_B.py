import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def prepare_data_bloodmnist(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Prepares BloodMNIST dataset by flattening images and normalizing pixel values.

    Args:
        x_train, y_train, x_val, y_val, x_test, y_test (np.ndarray): Input dataset images and labels.

    Returns:
        tuple: Processed PyTorch tensors.
    """
    # Normalize images
    x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_val_flat = x_val.reshape(x_val.shape[0], -1) / 255.0
    x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0

    # Convert to PyTorch tensors
    X_train = torch.tensor(x_train_flat, dtype=torch.float32)
    X_val = torch.tensor(x_val_flat, dtype=torch.float32)
    X_test = torch.tensor(x_test_flat, dtype=torch.float32)
    Y_train = torch.tensor(y_train, dtype=torch.long)
    Y_val = torch.tensor(y_val, dtype=torch.long)
    Y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def plot_confusion_matrix_B(y_test, y_test_pred, class_labels):
    """
    Generates and displays a normalized confusion matrix for BloodMNIST.

    Args:
        y_test (np.ndarray): True test labels.
        y_test_pred (np.ndarray): Predicted test labels.
        class_labels (list): List of class names.

    Returns:
        None
    """
    cm = confusion_matrix(y_test, y_test_pred)
    cm = cm / cm.sum(axis=1, keepdims=True) * 100  # Normalize per row

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues, values_format=".2f", ax=ax, colorbar=True)

    ax.grid(False)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("Normalized Confusion Matrix (BloodMNIST)", fontsize=12)
    plt.tight_layout()
    plt.show()
