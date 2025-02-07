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

    def __init__(self, input_channels, num_classes=8):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.fc(torch.flatten(self.conv_layers(x), start_dim=1))


def train_cnn_blood(X_train, Y_train, X_val, Y_val, use_optuna=False):
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
        best_params = {'batch_size': 32, 'learning_rate': 0.001, 'weight_decay': 0.0001, 'optimizer': 'adam'}

    batch_size = best_params['batch_size']
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

    cnn = CNN(X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, best_params['optimizer'].capitalize())(cnn.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])

    # Training loop
    num_epochs = 50
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        cnn.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = cnn(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
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
                outputs = cnn(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1)
                correct_val += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(correct_val / total_val)

    plot_cnn_training_curves_blood(train_losses, val_losses, train_accs, val_accs)
    return cnn, best_params


def evaluate_cnn_blood(cnn, X_test, Y_test):
    """
    Evaluates a CNN model on the BloodMNIST test set.

    Args:
        cnn (torch.nn.Module): Trained CNN model.
        X_test (torch.Tensor): Test feature set.
        Y_test (torch.Tensor): Test labels.

    Returns:
        dict: Test accuracy and classification report.
    """
    cnn.eval()
    with torch.no_grad():
        outputs = cnn(X_test.to(device))
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
