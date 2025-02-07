import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) model for image classification.

    Args:
        input_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
        num_conv_layers (int, optional): Number of convolutional layers.
        base_num_filters (int, optional): Number of filters in the first layer.
        kernel_size (int, optional): Kernel size for convolutional layers.
        is_binary (bool, optional): Whether it's a binary classification task.

    Returns:
        torch.nn.Module: CNN model.
    """

    def __init__(self, input_channels, num_conv_layers=3, base_num_filters=32, kernel_size=3, is_binary=True):
        super(CNN, self).__init__()
        layers = []
        in_channels = input_channels

        # Add convolutional layers
        for i in range(num_conv_layers):
            out_channels = base_num_filters * (2 ** i)  # Doubles the number of filters per layer
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))  # Reduces spatial dimensions
            in_channels = out_channels

        # Global Average Pooling (GAP) before FC layer
        layers.append(nn.AdaptiveAvgPool2d(1))  # (C, H, W) → (C, 1, 1)

        self.conv_layers = nn.Sequential(*layers)

        # Fully connected output layer
        self.fc = nn.Linear(in_channels, 1 if is_binary else 8)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)  # Flatten before FC layer
        return self.fc(x)


def prepare_data_cnn(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Prepares image data for CNN by normalizing pixel values and adding channel dimension.

    Args:
        x_train, y_train, x_val, y_val, x_test, y_test (np.ndarray): Datasets.

    Returns:
        tuple: Processed PyTorch tensors.
    """
    # Normalize images
    X_train_tensor = torch.tensor(x_train / 255.0, dtype=torch.float32)
    X_val_tensor = torch.tensor(x_val / 255.0, dtype=torch.float32)
    X_test_tensor = torch.tensor(x_test / 255.0, dtype=torch.float32)

    # Add channel dimension for grayscale images
    if X_train_tensor.ndim == 3:
        X_train_tensor = X_train_tensor.unsqueeze(1)
        X_val_tensor = X_val_tensor.unsqueeze(1)
        X_test_tensor = X_test_tensor.unsqueeze(1)

    # Convert labels to PyTorch tensors
    Y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    Y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    Y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return X_train_tensor, Y_train_tensor, X_val_tensor, Y_val_tensor, X_test_tensor, Y_test_tensor


def train_cnn_A(X_train, Y_train, X_val, Y_val, use_optuna=False):
    """
    Trains a CNN model, optionally using Optuna for hyperparameter tuning.

    Args:
        X_train (torch.Tensor): Training features.
        Y_train (torch.Tensor): Training labels.
        X_val (torch.Tensor): Validation features.
        Y_val (torch.Tensor): Validation labels.
        use_optuna (bool): Whether to use Optuna for hyperparameter tuning.

    Returns:
        tuple: Trained CNN model and best hyperparameters.
    """
    if use_optuna:
        def objective(trial):
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
            optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop", "adamw"])

            train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)

            cnn = CNN(X_train.shape[1]).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = getattr(optim, optimizer_name.capitalize())(cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)

            cnn.train()
            for epoch in range(50):
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = cnn(batch_X).squeeze()
                    loss = criterion(outputs, batch_y.squeeze())
                    loss.backward()
                    optimizer.step()

            val_outputs = cnn(X_val.to(device)).squeeze()
            val_predictions = (torch.sigmoid(val_outputs) > 0.5).float()
            val_accuracy = accuracy_score(Y_val.cpu().numpy(), val_predictions.cpu().numpy())

            return val_accuracy

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        best_params = study.best_params
        print("Best Hyperparameters:", best_params)
    else:
        best_params = {"batch_size": 32, "learning_rate": 0.001, "weight_decay": 0.0001, "optimizer": "adam"}

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=best_params["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=best_params["batch_size"], shuffle=False)

    cnn = CNN(X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = getattr(optim, best_params["optimizer"].capitalize())(cnn.parameters(), lr=best_params["learning_rate"], weight_decay=best_params["weight_decay"])

    # Training loop
    num_epochs = 50
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        cnn.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = cnn(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(correct_train / total_train)

        # Validation phase
        cnn.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = cnn(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(correct_val / total_val)

    plot_cnn_training_curves(train_losses, val_losses, train_accs, val_accs)
    return cnn, best_params


def evaluate_cnn_A(cnn, X_test, Y_test):
    """
    Evaluates a trained CNN model on the test dataset.

    Args:
        cnn (torch.nn.Module): Trained CNN model.
        X_test (torch.Tensor): Test features.
        Y_test (torch.Tensor): Test labels.

    Returns:
        dict: Evaluation metrics, including test accuracy and predictions.
    """
    cnn.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    predictions = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        outputs = cnn(X_test.to(device)).squeeze()
        loss = criterion(outputs, Y_test.to(device))
        test_loss += loss.item()

        # Convert logits to binary predictions (for binary classification tasks)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct_test += (predicted.cpu() == Y_test.cpu()).sum().item()
        total_test += Y_test.size(0)

        predictions = predicted.cpu().numpy()

    test_accuracy = correct_test / total_test

    print(f"[INFO] Test Loss: {test_loss:.4f}")
    print(f"[INFO] Test Accuracy: {test_accuracy:.4f}")

    return {"test_loss": test_loss, "test_accuracy": test_accuracy, "predictions": predictions}


def plot_cnn_training_curves(train_losses, val_losses, train_accs, val_accs):
    """
    Plots training and validation loss/accuracy curves.

    Args:
        train_losses (list): Training loss history.
        val_losses (list): Validation loss history.
        train_accs (list): Training accuracy history.
        val_accs (list): Validation accuracy history.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss", color='blue')
    plt.plot(val_losses, label="Validation Loss", color='darkblue')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Training Accuracy", color='green')
    plt.plot(val_accs, label="Validation Accuracy", color='darkgreen')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
