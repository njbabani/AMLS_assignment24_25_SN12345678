import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

# Set device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ANN(nn.Module):
    """
    Artificial Neural Network (ANN) Model.

    Args:
        input_size (int): Number of input features.
        hidden_sizes (list): List of hidden layer sizes.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        torch.nn.Module: ANN Model.
    """

    def __init__(self, input_size, hidden_sizes, dropout_rate):
        super(ANN, self).__init__()
        layers = []
        prev_size = input_size

        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = size

        # Output layer (Binary Classification)
        layers.append(nn.Linear(prev_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def prepare_data_ann(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Prepares BreastMNIST dataset for ANN by flattening images and normalizing.

    Args:
        x_train, y_train, x_val, y_val, x_test, y_test (np.ndarray): Datasets.

    Returns:
        tuple: Processed PyTorch tensors.
    """
    # Flatten images
    x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_val_flat = x_val.reshape(x_val.shape[0], -1) / 255.0
    x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0

    # Convert to PyTorch tensors
    X_train = torch.tensor(x_train_flat, dtype=torch.float32)
    X_val = torch.tensor(x_val_flat, dtype=torch.float32)
    X_test = torch.tensor(x_test_flat, dtype=torch.float32)
    Y_train = torch.tensor(y_train, dtype=torch.float32)
    Y_val = torch.tensor(y_val, dtype=torch.float32)
    Y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def train_ann_model(X_train, Y_train, X_val, Y_val, use_optuna=False):
    """
    Trains an ANN model, optionally using Optuna hyperparameter tuning.

    Args:
        X_train (torch.Tensor): Training features.
        Y_train (torch.Tensor): Training labels.
        X_val (torch.Tensor): Validation features.
        Y_val (torch.Tensor): Validation labels.
        use_optuna (bool): Whether to use Optuna for hyperparameter tuning.

    Returns:
        tuple: Trained ANN model, best hyperparameters.
    """
    if use_optuna:
        def objective(trial):
            # Optuna hyperparameters to optimize
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop', 'adamw'])

            # Get train dataset
            train_dataset = TensorDataset(X_train, Y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            ann = ANN(X_train.shape[1], [128, 64, 32], dropout_rate).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = getattr(optim, optimizer_name.capitalize())(ann.parameters(), lr=learning_rate, weight_decay=weight_decay)

            ann.train()
            for epoch in range(50):
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = ann(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            return accuracy_score(
                Y_train.cpu().numpy(),
                (torch.sigmoid(ann(X_train.to(device))).cpu().numpy() > 0.5).astype(int))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        best_params = study.best_params
        print("Best Hyperparameters:", best_params)
    else:
        best_params = {
            'learning_rate': 0.0005,
            'dropout_rate': 0.3,
            'weight_decay': 0.002,
            'batch_size': 32,
            'optimizer': 'adam'
        }

    batch_size = best_params['batch_size']
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

    ann = ANN(X_train.shape[1], [128, 64, 32], best_params['dropout_rate']).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = getattr(optim, best_params['optimizer'].capitalize())(ann.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])

    # Tracking loss and accuracy
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    num_epochs = 50

    # Training
    for epoch in range(num_epochs):
        ann.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = ann(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)

        # Validation phase
        ann.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = ann(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

        # Print epoch stats
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}")

    # Plot training curves
    plot_ann_training_curves(train_losses, val_losses, train_accs, val_accs)

    return ann, best_params


def evaluate_ann_model(ann, X_test, Y_test):
    """
    Evaluates an ANN model on the test set.

    Args:
        ann (torch.nn.Module): Trained ANN model.
        X_test (torch.Tensor): Test feature set.
        Y_test (torch.Tensor): Test labels.

    Returns:
        dict: Test accuracy and classification report.
    """
    ann.eval()
    with torch.no_grad():
        outputs = ann(X_test.to(device)).squeeze()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        test_accuracy = accuracy_score(Y_test.cpu().numpy(), predictions.cpu().numpy())

    report = classification_report(Y_test.cpu().numpy(), predictions.cpu().numpy(), digits=4)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Classification Report:\n", report)

    return {"test_accuracy": test_accuracy, "classification_report": report}


def plot_ann_training_curves(train_losses, val_losses, train_accs, val_accs):
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
    plt.plot(train_accs, label="Training Accuracy", color='red')
    plt.plot(val_accs, label="Validation Accuracy", color='darkred')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
