import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report


# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ANN(nn.Module):
    """
    Artificial Neural Network (ANN) Model for BloodMNIST.

    Args:
        input_size (int): Number of input features.
        hidden_sizes (list): List of hidden layer sizes.
        dropout_rate (float): Dropout rate for regularization.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: ANN model.
    """

    def __init__(self, input_size, hidden_sizes, dropout_rate, num_classes=8):
        super(ANN, self).__init__()
        layers = []
        prev_size = input_size

        # Add hidden layers with ReLU activation and Dropout
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = size

        # Output layer for multi-class classification
        layers.append(nn.Linear(prev_size, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_ann_B(X_train, Y_train, X_val, Y_val, use_optuna=False):
    """
    Trains an ANN model on BloodMNIST, optionally using Optuna for hyperparameter tuning.

    Args:
        X_train, Y_train (torch.Tensor): Training dataset.
        X_val, Y_val (torch.Tensor): Validation dataset.
        use_optuna (bool): Whether to use Optuna for hyperparameter tuning.

    Returns:
        tuple: Trained ANN model and best hyperparameters.
    """
    if use_optuna:
        def objective(trial):
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop', 'adamw'])

            train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)

            ann = ANN(X_train.shape[1], [512, 256, 128, 64], dropout_rate, num_classes=8).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = getattr(optim, optimizer_name.capitalize())(ann.parameters(), lr=learning_rate, weight_decay=weight_decay)

            ann.train()
            for epoch in range(50):
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    loss = criterion(ann(batch_X), batch_y)
                    loss.backward()
                    optimizer.step()

            val_outputs = ann(X_val.to(device))
            val_predictions = torch.argmax(val_outputs, dim=1)
            val_accuracy = accuracy_score(Y_val.cpu().numpy(), val_predictions.cpu().numpy())

            return val_accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        best_params = study.best_params
        print("Best Hyperparameters:", best_params)
    else:
        best_params = {'learning_rate': 0.00022749387063450467, 'dropout_rate': 0.26128241747910286, 'weight_decay': 3.004922888051895e-05, 'batch_size': 32, 'optimizer': 'adam'}

    batch_size = best_params['batch_size']
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

    ann = ANN(X_train.shape[1], [512, 256, 128, 64], best_params['dropout_rate'], num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, best_params['optimizer'].capitalize())(ann.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])

    # Training loop
    num_epochs = 100
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        ann.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_y = batch_y.squeeze()
            optimizer.zero_grad()
            outputs = ann(batch_X)
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
        ann.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                batch_y = batch_y.squeeze()
                outputs = ann(batch_X)
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

    plot_ann_training_curves_blood(train_losses, val_losses, train_accs, val_accs)
    return ann, best_params


def evaluate_ann_B(ann_, X_test, Y_test, use_best=True):
    """
    Evaluates an ANN model on the test set.

    Args:
        ann_ (torch.nn.Module): Trained ANN model.
        X_test (torch.Tensor): Test feature set.
        Y_test (torch.Tensor): Test labels.
        use_best (bool): Controls whether to use loaded model or not

    Returns:
        dict: Test accuracy and classification report.
    """
    if use_best:
        # Use values from Optuna
        best_params = {'learning_rate': 0.00022749387063450467, 'dropout_rate': 0.26128241747910286, 'weight_decay': 3.004922888051895e-05, 'batch_size': 32, 'optimizer': 'adam'}

        # Matching 28x28x3
        input_size = 2352

        # Fixed architecture
        hidden_layer_sizes = [512, 256, 128, 64]

        # Creates the ANN module
        ann = ANN(input_size, hidden_layer_sizes, best_params['dropout_rate']).to(device)

        # Load the best model
        ann.load_state_dict(ann_)
    else:
        # Simply passes the ann_ directly as a model
        ann = ann_

    ann.eval()
    with torch.no_grad():
        outputs = ann(X_test.to(device))
        predictions = torch.argmax(outputs, dim=1)
        test_accuracy = accuracy_score(Y_test.cpu().numpy(), predictions.cpu().numpy())

    report = classification_report(Y_test.cpu().numpy(), predictions.cpu().numpy(), digits=4)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Classification Report:\n", report)

    return {"test_accuracy": test_accuracy, "classification_report": report}


def plot_ann_training_curves_blood(train_losses, val_losses, train_accs, val_accs):
    """
    Plots training and validation loss/accuracy curves for BloodMNIST ANN.

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
    plt.plot(train_losses, label="Training Loss", color='red')
    plt.plot(val_losses, label="Validation Loss", color='darkred')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Training Accuracy", color='red')
    plt.plot(val_accs, label="Validation Accuracy", color='darkred')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
