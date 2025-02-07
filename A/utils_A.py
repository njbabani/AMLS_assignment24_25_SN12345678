import numpy as np
import random
from torch.utils.data import Dataset
from torchvision.datasets import BreastMNIST
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class AugmentedDataset(Dataset):
    """
    A custom dataset class for storing augmented images
    and their corresponding labels.

    Args:
        images (list of torch.Tensor): A list of image tensors.
        labels (list of int): A list of corresponding labels for the images.

    Attributes:
        images (list of torch.Tensor): The stored image tensors.
        labels (list of int): The stored labels.
    """

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieves a sample (image, label) pair from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image tensor and its label.
        """
        return self.images[idx], self.labels[idx]


def augmented_transform():
    """
    Defines a series of data augmentation transformations.

    The augmentations include random vertical and horizontal flips,
    as well as random brightness adjustments.

    Returns:
        torchvision.transforms.Compose: A composed set of transformations.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Lambda(
            lambda img: TF.adjust_brightness(
                img, brightness_factor=random.uniform(0.8, 1.2)
            )
        ),
    ])


def shuffle_data(x_data, y_data):
    """
    Shuffles the data and labels in unison.

    Args:
        x_data (np.ndarray): Array of data (e.g., images).
        y_data (np.ndarray): Array of labels.

    Returns:
        tuple: A tuple containing shuffled data and labels as NumPy arrays.
    """
    combined = list(zip(x_data, y_data))
    random.shuffle(combined)
    x_data, y_data = zip(*combined)
    return np.array(x_data), np.array(y_data)


def prepare_breastmnist_data():
    """
    Loads, augments, shuffles, and processes the BreastMNIST dataset.

    Returns:
        tuple: Processed datasets:
            - x_train (np.ndarray): Training images (after augmentation).
            - y_train (np.ndarray): Training labels.
            - x_val (np.ndarray): Validation images.
            - y_val (np.ndarray): Validation labels.
            - x_test (np.ndarray): Test images.
            - y_test (np.ndarray): Test labels.
    """
    # Step 1: Load the datasets
    train_dataset = BreastMNIST(split='train', download=True)
    val_dataset = BreastMNIST(split='val', download=True)
    test_dataset = BreastMNIST(split='test', download=True)

    # Extract original images and labels
    x_train, y_train = train_dataset.imgs, train_dataset.labels
    x_val, y_val = val_dataset.imgs, val_dataset.labels
    x_test, y_test = test_dataset.imgs, test_dataset.labels

    # Step 2: Apply transformations and augmentations for the training set
    transform = augmented_transform()
    augmented_images = []
    augmented_labels = []

    # Loop through training dataset and augment the data
    for train_pic in train_dataset:
        train_img, train_label = train_pic

        # Transform and add the augmented image to the list
        augmented_image = transform(train_img)
        augmented_images.append(augmented_image)
        augmented_labels.append(train_label)

    # Convert augmented data to numpy arrays
    augmented_images_np = np.array(
        [img.numpy().squeeze(0) for img in augmented_images]
        )
    augmented_labels_np = np.array(augmented_labels)

    # Concatenate original and augmented data
    x_train = np.concatenate((x_train, augmented_images_np), axis=0)
    y_train = np.concatenate((y_train, augmented_labels_np), axis=0)

    # Step 3: Shuffle the training data
    x_train, y_train = shuffle_data(x_train, y_train)

    return x_train, y_train, x_val, y_val, x_test, y_test


def plot_confusion_matrix_A(y_test, y_test_pred, class_labels):
    """
    Generates and displays a normalized confusion matrix.

    Args:
        y_test (np.ndarray): True test labels.
        y_test_pred (np.ndarray): Predicted test labels.
        class_labels (list): List of class names.

    Returns:
        None
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    # Normalize the confusion matrix (row-wise percentages)
    cm = cm / cm.sum(axis=1, keepdims=True) * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 3))

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_labels
    )
    disp.plot(cmap=plt.cm.Blues, values_format=".2f", ax=ax, colorbar=True)
    ax.grid(False)

    # Title and formatting
    plt.title(r"Normalized Confusion Matrix", fontsize=10)
    plt.tight_layout()

    # Display the figure
    plt.show()
