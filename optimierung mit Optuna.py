# Refaktoriertes CNN mit Optuna Hyperparameter-Tuning (PyTorch + MNIST)
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import optuna

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Affine_train=True
print("Device:", device)
print("CUDA verfügbar:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =====================================================
# CONFIG
# =====================================================

CONFIG = {
    "epochs": 5,
    "n_trials": 20
}

# =====================================================
# TRANSFORMS
# =====================================================


test_transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if Affine_train==True:
        train_transform = transforms.Compose([    transforms.RandomAffine(
        degrees=10,            # Rotation
        translate=(0.1,0.1),   # Verschiebung
        scale=(0.9,1.1)        # Skalierung
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])

else:
    train_transform=test_transform

# =====================================================
# DATASETS
# =====================================================

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

# =====================================================
# CNN MODELL
# =====================================================

class CNN(nn.Module):

    def __init__(self, hidden_dim=128, dropout=0.2):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(32 * 7 * 7, hidden_dim),
            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):

        x = self.conv(x)
        x = self.fc(x)

        return x

# =====================================================
# TRAINING FUNKTION
# =====================================================

def train_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss

# =====================================================
# EVALUATION FUNKTION
# =====================================================

def evaluate(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)

            correct += (predictions == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy

# =====================================================
# OPTUNA OBJECTIVE
# =====================================================

def objective(trial):

    # ---------------------------------------------
    # HYPERPARAMETER
    # ---------------------------------------------

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-3,
        4e-3,
        log=True
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [128]
    )

    hidden_dim = trial.suggest_int(
        "hidden_dim",
        80,
        140
    )

    dropout = trial.suggest_float(
        "dropout",
        0.15,
        0.3
    )

    optimizer_name = trial.suggest_categorical(
        "optimizer",
        ["Adam", "SGD"]
    )

    # ---------------------------------------------
    # DATALOADER
    # ---------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    # ---------------------------------------------
    # MODELL
    # ---------------------------------------------

    model = CNN(
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)

    # ---------------------------------------------
    # LOSS
    # ---------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------
    # OPTIMIZER
    # ---------------------------------------------

    if optimizer_name == "Adam":

        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

    else:

        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate
        )

    # ---------------------------------------------
    # TRAINING
    # ---------------------------------------------

    for epoch in range(CONFIG["epochs"]):

        loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        accuracy = evaluate(
            model,
            test_loader,
            device
        )

        print(
            f"Trial {trial.number} | "
            f"Epoch {epoch+1} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

        # ---------------------------------------------
        # EARLY STOPPING / PRUNING
        # ---------------------------------------------

        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return accuracy

# =====================================================
# OPTUNA STUDY
# =====================================================

study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=CONFIG["n_trials"]
)

# =====================================================
# BESTE PARAMETER
# =====================================================

print("\nBeste Hyperparameter:")
print(study.best_params)

print("\nBeste Accuracy:")
print(study.best_value)
