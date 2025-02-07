import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


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
