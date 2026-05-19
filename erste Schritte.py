import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as iio

# -----------------------------
# DEVICE (GPU oder CPU wählen)
# -----------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)  # z.B. "cuda:0" oder "cpu"
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# -----------------------------
# DATEN VORBEREITEN (MNIST)
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),              # Bild → Tensor, Werte [0,255] → [0,1]
    transforms.Normalize((0.5,), (0.5,))  # → [-1,1] (stabileres Training)
])

# MNIST laden (wird automatisch heruntergeladen)
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# DataLoader erstellt Batches und shuffled die Daten
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64)


# =====================================================
# CNN MODELL
# =====================================================

class CNN(nn.Module):

    def __init__(self, hidden_dim=128, dropout=0.0):

        super().__init__()

        # Convolution-Teil (Feature Extraction)
        self.conv = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, padding=1), # (1,28,28) → (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                            # → (16,14,14)

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # → (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2)                             # → (32,7,7)
        )
        # Fully Connected Teil (Klassifikation)
        self.fc = nn.Sequential(

            nn.Flatten(),
            nn.Linear(32 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 10)   # 10 Klassen (Ziffern 0–9)
        )

    def forward(self, x):

        x = self.conv(x) # Feature Extraction
        x = self.fc(x) # Klassifikation

        return x

# Modell auf Device schieben (GPU/CPU)
model = CNN().to(device)

# -----------------------------
# LOSS + OPTIMIZER
# -----------------------------
criterion = nn.CrossEntropyLoss()              # Klassifikations-Loss
optimizer = optim.Adam(model.parameters(), lr=0.002)


# -----------------------------
# TRANSFORM FÜR EIGENE BILDER
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),                 # NumPy → PIL
    transforms.Grayscale(),                  # RGB → 1 Kanal
    transforms.functional.invert,            # weißer Hintergrund → schwarz
    transforms.ToTensor(),                   # → Tensor (1,28,28)
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# EIGENE BILDER LADEN
# -----------------------------
test_images = []
result=[2,7,3,8,9,1]

for i in range(1, 7):
    img = iio.imread("bild" + str(i) + ".png")  # NumPy Array (H,W,3)
    test_images.append(transform(img))


# -----------------------------
# TRAINING
# -----------------------------
epochs = 8

for epoch in range(epochs): #run process of training -> evaluation -> test on my own images for epochs times
    model.train()  # Trainingsmodus aktivieren
    total_loss = 0
    predictions=[]

    for images, labels in train_loader:
        # Daten auf GPU/CPU verschieben
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)        # Forward Pass
        loss = criterion(outputs, labels)

        optimizer.zero_grad()          # Gradienten zurücksetzen
        loss.backward()                # Backpropagation
        optimizer.step()               # Gewichte updaten

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# model.eval()

# fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# axes = axes.flatten()

# softmax = nn.Softmax(dim=1)

# for i, im in enumerate(test_images):

#     with torch.no_grad():

#         # ---------------------------------
#         # Bild vorbereiten
#         # ---------------------------------

#         input_img = im.unsqueeze(0).to(device)

#         # ---------------------------------
#         # Vorhersage
#         # ---------------------------------

#         outputs = model(input_img)

#         probabilities = softmax(outputs)

#         pred = torch.argmax(probabilities, dim=1)

#         predicted_class = pred.item()

#         # ---------------------------------
#         # Softmax-Vektor
#         # ---------------------------------

#         probs = probabilities.cpu().numpy()[0]

#         # ---------------------------------
#         # Bild denormalisieren
#         # ---------------------------------

#         img_show = im * 0.5 + 0.5

#         # ---------------------------------
#         # Plot
#         # ---------------------------------

#         axes[i].imshow(
#             img_show.squeeze(),
#             cmap="gray"
#         )

#         axes[i].axis("off")

#         axes[i].set_title(
#             f"Pred: {predicted_class}\n"
#             f"True: {result[i]}\n"
#             f"Conf: {np.max(probs):.2f}"
#         )

#         # ---------------------------------
#         # Softmax-Werte ausgeben
#         # ---------------------------------

#         print(f"\nBild {i+1}")

#         print("Vorhersage:", predicted_class)

#         print("Wahrscheinlichkeiten:")

#         for digit, prob in enumerate(probs):

#             print(f"{digit}: {prob:.4f}")

#         plt.tight_layout()

#         plt.show()
# -----------------------------
# EVALUATION
# -----------------------------
model.eval()  # Evaluationsmodus
correct = 0
total = 0

with torch.no_grad():  # keine Gradienten berechnen (schneller)
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # höchste Wahrscheinlichkeit wählen

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
print("\n")

# -----------------------------
# PREDICTION OWN IMAGES
# -----------------------------
for i, im in enumerate(test_images):

    with torch.no_grad():
        im = im.unsqueeze(0)            # → (1,1,28,28) (Batch hinzufügen)
        im = im.to(device)              # auf GPU/CPU
        outputs = model(im)
        _, pred = torch.max(outputs, 1)
        predictions.append(pred.item())

        print("Prediction:", outputs)
        print("Reference:",result[i])


print("\n")




# Bild transformieren

# anzeigen (vorher denormalisieren), nur für einzelne images möglich
# img_show = img * 0.5 + 0.5
# plt.imshow(img_show.squeeze(), cmap="gray")
# plt.show()
