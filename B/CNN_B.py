import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report


# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) model for BloodMNIST.

    Args:
        input_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: CNN model.
    """
    def __init__(self, input_channels, num_conv_layers=3, base_num_filters=32, kernel_size=3, num_classes=8):
        super(CNN, self).__init__()
        layers = []
        in_channels = input_channels  # Input image channels

        # Add Convolutional Layers with Increasing Channels
        for i in range(num_conv_layers):
            out_channels = base_num_filters * (2 ** i)  # Doubles the channels at each layer
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))  # Pooling halves spatial dimensions
            in_channels = out_channels  # Update input channels for the next layer

        # Global Average Pooling (GAP) before FC layer
        layers.append(nn.AdaptiveAvgPool2d(1))  # Reduces (C, H, W) → (C, 1, 1)

        self.conv_layers = nn.Sequential(*layers)

        # Fully connected layer
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)  # Flatten after GAP
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

    # If grayscale, add channel dimension (N, 1, H, W)
    if X_train_tensor.ndim == 3:
        X_train_tensor = X_train_tensor.unsqueeze(1)
        X_val_tensor = X_val_tensor.unsqueeze(1)
        X_test_tensor = X_test_tensor.unsqueeze(1)
    # If RGB, permute to (N, C, H, W)
    elif X_train_tensor.ndim == 4:
        X_train_tensor = X_train_tensor.permute(0, 3, 1, 2)
        X_val_tensor = X_val_tensor.permute(0, 3, 1, 2)
        X_test_tensor = X_test_tensor.permute(0, 3, 1, 2)

    # Convert labels to PyTorch tensors
    Y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Multi-class labels
    Y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    Y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, Y_train_tensor, X_val_tensor, Y_val_tensor, X_test_tensor, Y_test_tensor


def train_cnn_B(X_train, Y_train, X_val, Y_val, use_optuna=False):
    """
    Trains a CNN model on BloodMNIST, optionally using Optuna for hyperparameter tuning.

    Args:
        X_train, Y_train (torch.Tensor): Training dataset.
        X_val, Y_val (torch.Tensor): Validation dataset.
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
            criterion = nn.CrossEntropyLoss()
            optimizer = getattr(optim, optimizer_name.capitalize())(cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)

            cnn.train()
            for epoch in range(50):
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    loss = criterion(cnn(batch_X), batch_y)
                    loss.backward()
                    optimizer.step()

            val_outputs = cnn(X_val.to(device))
            val_predictions = torch.argmax(val_outputs, dim=1)
            val_accuracy = accuracy_score(Y_val.cpu().numpy(), val_predictions.cpu().numpy())

            return val_accuracy

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        best_params = study.best_params
        print("Best Hyperparameters:", best_params)
    else:
        best_params = {'batch_size': 64, 'learning_rate': 0.0010847201523660587, 'weight_decay': 0.0025689992565340975, 'optimizer': 'AdamW'}

    batch_size = best_params['batch_size']
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

    cnn = CNN(X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, best_params['optimizer'])(cnn.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])

    # Training loop
    num_epochs = 100
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        cnn.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = cnn(batch_X).squeeze()
            batch_y = batch_y.squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct_train += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(running_loss / len(train_loader))
        train_accs.append(correct_train / total_train)

        # Validation phase
        cnn.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = cnn(batch_X).squeeze()
                batch_y = batch_y.squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1)
                correct_val += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

        # Print epoch stats
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}")

    plot_cnn_training_curves_blood(train_losses, val_losses, train_accs, val_accs)
    return cnn, best_params


def evaluate_cnn_B(cnn_, X_test, Y_test, use_best=True):
    """
    Evaluates a CNN model on the test set for Task A.

    Args:
        cnn_ (torch.nn.Module or dict): Trained CNN model instance or a state dictionary.
                                        If use_best is True, cnn_ is assumed to be a state dict.
        X_test (torch.Tensor): Test feature set.
        Y_test (torch.Tensor): Test labels.
        use_best (bool): If True, a new CNN instance is created and cnn_'s state dict is loaded.
                         Otherwise, cnn_ is used directly.

    Returns:
        dict: A dictionary containing the test accuracy and a detailed classification report.
    """
    if use_best:
        # For BloodMNIST, images are RGB (1 channel)
        input_channels = 3
        # Instantiate a new CNN model with the correct architecture
        cnn = CNN(input_channels).to(device)
        # Load the state dictionary into the model
        cnn.load_state_dict(cnn_)
    else:
        # Use the provided model instance directly
        cnn = cnn_

    cnn.eval()
    with torch.no_grad():
        outputs = cnn(X_test.to(device)).squeeze()
        predictions = torch.argmax(outputs, dim=1)
        test_accuracy = accuracy_score(Y_test.cpu().numpy(), predictions.cpu().numpy())
        report = classification_report(Y_test.cpu().numpy(), predictions.cpu().numpy(), digits=4)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Classification Report:\n", report)

    return {"test_accuracy": test_accuracy, "classification_report": report}


def plot_cnn_training_curves_blood(train_losses, val_losses, train_accs, val_accs):
    """
    Plots training and validation loss/accuracy curves for BloodMNIST CNN.

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
    plt.plot(train_losses, label="Training Loss", color='lime')
    plt.plot(val_losses, label="Validation Loss", color='darkgreen')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Training Accuracy", color='lime')
    plt.plot(val_accs, label="Validation Accuracy", color='darkgreen')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
