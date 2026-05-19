import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -----------------------------
# DEVICE (GPU oder CPU wählen)
# -----------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"CUDA verfügbar: {torch.cuda.is_available()}")
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

MODEL_PATH = "mnist_cnn.pth"

# -----------------------------
# HYPERPARAMETER
# -----------------------------
HP = {
    "learning_rate": 0.001352,
    "hidden_dim":    135,
    "dropout":       0.1736,
    "batch_size":    128,
    "epochs":        20,
}

# =====================================================
# CNN MODELL
# =====================================================
class CNN(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (1,28,28) → (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → (16,14,14)
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # → (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2)                               # → (32,7,7)
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
# TRAINING
# =====================================================
def train_model(hp: dict):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=hp["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=hp["batch_size"])

    model = CNN(hidden_dim=hp["hidden_dim"], dropout=hp["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hp["learning_rate"])

    for epoch in range(hp["epochs"]):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{hp['epochs']}  -  Loss: {total_loss:.4f}")

    # Evaluation auf Testset
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")

    # Modell + alle Hyperparameter speichern
    torch.save({"hyperparameters": hp, "state_dict": model.state_dict()}, MODEL_PATH)
    print(f"Modell gespeichert -> {MODEL_PATH}")
    return model, hp


# =====================================================
# MODELL LADEN ODER TRAINIEREN
# =====================================================
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "hyperparameters" in checkpoint:
        # Aktuelles Format: Hyperparameter + state_dict
        loaded_hp = checkpoint["hyperparameters"]
        model = CNN(hidden_dim=loaded_hp["hidden_dim"], dropout=loaded_hp["dropout"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(f"Gespeichertes Modell geladen aus '{MODEL_PATH}'")
        print(f"  Hyperparameter: {loaded_hp}")
        print("Zum Neu-Trainieren die Datei loeschen oder MODEL_PATH aendern.")
    else:
        # Altes / inkompatibles Format -> neu trainieren
        print("Inkompatibles Checkpoint-Format -> Modell wird neu trainiert.\n")
        os.remove(MODEL_PATH)
        model, _ = train_model(HP)
        model.eval()
else:
    print("Kein gespeichertes Modell gefunden – Training startet ...\n")
    model, _ = train_model(HP)
    model.eval()


# =====================================================
# PREPROCESSING FÜR GEZEICHNETE BILDER
# =====================================================
def preprocess_canvas(pil_image: Image.Image) -> torch.Tensor:
    """
    MNIST-kompatibles Preprocessing:
      1. Bounding Box der gezeichneten Pixel finden
      2. Ziffer auf 20x20 skalieren  (wie MNIST intern)
      3. In 28x28 zentrieren
      4. Normalisieren mit (0.5, 0.5)  wie im Training
    """
    arr = np.array(pil_image, dtype=np.float32)   # (280,280), Werte 0–255

    # --- Bounding Box der nicht-leeren Pixel finden ---
    mask_rows = np.any(arr > 10, axis=1)
    mask_cols = np.any(arr > 10, axis=0)

    if not mask_rows.any():
        # Leeres Bild – Nulltensor zurückgeben
        tensor = torch.zeros(1, 1, 28, 28)
        return tensor.to(device)

    rmin, rmax = np.where(mask_rows)[0][[0, -1]]
    cmin, cmax = np.where(mask_cols)[0][[0, -1]]

    # --- Ziffer ausschneiden ---
    cropped = arr[rmin:rmax+1, cmin:cmax+1]

    # --- Auf max. 20x20 skalieren (Seitenverhältnis erhalten) ---
    h, w   = cropped.shape
    scale  = 20.0 / max(h, w)
    new_h  = max(1, int(round(h * scale)))
    new_w  = max(1, int(round(w * scale)))

    cropped_pil = Image.fromarray(cropped.astype(np.uint8))
    cropped_pil = cropped_pil.resize((new_w, new_h), Image.LANCZOS)

    # --- In 28x28 zentrieren ---
    final = np.zeros((28, 28), dtype=np.float32)
    top   = (28 - new_h) // 2
    left  = (28 - new_w) // 2
    final[top:top + new_h, left:left + new_w] = np.array(cropped_pil, dtype=np.float32)

    # --- Normalisieren wie Trainings-Transform ---
    final = final / 255.0          # [0, 1]
    final = (final - 0.5) / 0.5   # [-1, 1]

    tensor = torch.tensor(final).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
    return tensor.to(device)


# =====================================================
# KLASSIFIKATION
# =====================================================
def classify(tensor: torch.Tensor):
    """Gibt (predicted_class, probs_array) zurück."""
    with torch.no_grad():
        outputs = model(tensor)
        softmax = nn.Softmax(dim=1)
        probs   = softmax(outputs).cpu().numpy()[0]
        pred    = int(np.argmax(probs))
    return pred, probs


# =====================================================
# ZEICHENFENSTER (tkinter)
# =====================================================
CANVAS_SIZE   = 280   # angezeigte Fenstergröße
BRUSH_RADIUS  = 12    # Pinselgröße in Pixeln

class DrawApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Ziffer zeichnen – MNIST Klassifikator")
        self.root.resizable(False, False)

        # --- internes PIL-Bild (schwarzer Hintergrund) ---
        self._pil_img  = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self._pil_draw = ImageDraw.Draw(self._pil_img)
        self._tk_img   = None   # wird bei jedem Pinselstrich aktualisiert

        self._build_ui()
        self._bind_events()

    # --------------------------------------------------
    def _build_ui(self):
        # Hauptframe
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")

        # Anleitung
        ttk.Label(
            frame,
            text="Zeichne eine Ziffer (0–9) und klicke auf 'Klassifizieren'.",
            font=("Helvetica", 11)
        ).grid(row=0, column=0, columnspan=3, pady=(0, 6))

        # Zeichenfläche
        self.canvas = tk.Canvas(
            frame,
            width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="black", cursor="crosshair",
            highlightthickness=1, highlightbackground="#555"
        )
        self.canvas.grid(row=1, column=0, columnspan=3, padx=4, pady=4)

        # Canvas-Item einmalig anlegen – wird bei jedem Strich nur aktualisiert,
        # nicht neu gestapelt (das war der ursprüngliche Bug)
        blank = ImageTk.PhotoImage(self._pil_img)
        self._tk_img = blank          # Referenz halten damit GC nicht aufräumt
        self._canvas_img_id = self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=0, columnspan=3, pady=6)

        ttk.Button(btn_frame, text="Klassifizieren", command=self._on_classify,
                   width=16).grid(row=0, column=0, padx=6)
        ttk.Button(btn_frame, text="Löschen",        command=self._on_clear,
                   width=12).grid(row=0, column=1, padx=6)
        ttk.Button(btn_frame, text="Schließen",      command=self.root.destroy,
                   width=12).grid(row=0, column=2, padx=6)

        # Ergebnisbereich
        self.result_label = ttk.Label(
            frame, text="", font=("Helvetica", 18, "bold"), foreground="#1a73e8"
        )
        self.result_label.grid(row=3, column=0, columnspan=3, pady=(4, 0))

        self.conf_label = ttk.Label(frame, text="", font=("Helvetica", 10))
        self.conf_label.grid(row=4, column=0, columnspan=3)

        # Matplotlib-Balkendiagramm (Softmax-Wahrscheinlichkeiten)
        self.fig, self.ax = plt.subplots(figsize=(4.2, 2.2))
        self.fig.patch.set_facecolor("#f5f5f5")
        self._init_bar_chart()
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.mpl_canvas.get_tk_widget().grid(row=5, column=0, columnspan=3, pady=6)

        # 28×28 Vorschau – zeigt exakt was das Modell sieht
        preview_frame = ttk.LabelFrame(frame, text="Modell-Eingabe (28×28)", padding=6)
        preview_frame.grid(row=6, column=0, columnspan=3, pady=(0, 6))
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack()

    # --------------------------------------------------
    def _init_bar_chart(self):
        self.ax.clear()
        self.bars = self.ax.bar(range(10), [0]*10, color="#90caf9", edgecolor="#1565c0")
        self.ax.set_xticks(range(10))
        self.ax.set_xlabel("Ziffer", fontsize=9)
        self.ax.set_ylabel("Wahrscheinlichkeit", fontsize=9)
        self.ax.set_ylim(0, 1)
        self.ax.set_title("Softmax-Ausgabe", fontsize=10)
        self.fig.tight_layout()

    # --------------------------------------------------
    def _bind_events(self):
        self.canvas.bind("<B1-Motion>",      self._on_draw)
        self.canvas.bind("<ButtonPress-1>",  self._on_draw)

    # --------------------------------------------------
    def _on_draw(self, event):
        x, y = event.x, event.y
        r = BRUSH_RADIUS
        # Auf PIL-Bild malen (akkumuliert alle Striche korrekt)
        self._pil_draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
        # Vorhandenes Canvas-Item aktualisieren – KEIN create_image (kein Stapeln)
        self._tk_img = ImageTk.PhotoImage(self._pil_img)
        self.canvas.itemconfig(self._canvas_img_id, image=self._tk_img)

    # --------------------------------------------------
    def _on_clear(self):
        self._pil_draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        # Canvas-Item aktualisieren statt alles löschen
        self._tk_img = ImageTk.PhotoImage(self._pil_img)
        self.canvas.itemconfig(self._canvas_img_id, image=self._tk_img)
        self.result_label.config(text="")
        self.conf_label.config(text="")
        self.preview_label.config(image="")
        self._init_bar_chart()
        self.mpl_canvas.draw()

    # --------------------------------------------------
    def _on_classify(self):
        tensor = preprocess_canvas(self._pil_img)
        pred, probs = classify(tensor)

        # 28×28 Vorschau rendern (zeigt exakt den Modell-Input)
        preview_arr = (tensor.cpu().numpy()[0, 0] * 0.5 + 0.5) * 255  # [-1,1] -> [0,255]
        preview_pil = Image.fromarray(preview_arr.astype(np.uint8), mode="L")
        # Auf 140x140 hochskaliert (nearest-neighbor damit Pixel sichtbar bleiben)
        preview_pil = preview_pil.resize((140, 140), Image.NEAREST)
        self._preview_tk = ImageTk.PhotoImage(preview_pil)
        self.preview_label.config(image=self._preview_tk)

        # Beschriftungen aktualisieren
        self.result_label.config(
            text=f"Vorhergesagte Ziffer: {pred}"
        )
        self.conf_label.config(
            text=f"Konfidenz: {probs[pred]*100:.1f}%"
        )

        # Balkendiagramm aktualisieren
        for bar, p in zip(self.bars, probs):
            bar.set_height(p)
            bar.set_color("#ef5350" if bar == self.bars[pred] else "#90caf9")
        self.bars[pred].set_color("#43a047")   # grün für Vorhersage
        self.ax.set_title(f"Softmax-Ausgabe  →  Vorhersage: {pred}", fontsize=10)
        self.mpl_canvas.draw()

        # Kurze Konsolenausgabe
        print(f"\nVorhersage: {pred}  (Konfidenz: {probs[pred]*100:.1f}%)")
        for digit, p in enumerate(probs):
            print(f"  {digit}: {p:.4f}")


# =====================================================
# PROGRAMM STARTEN
# =====================================================
if __name__ == "__main__":
    root = tk.Tk()
    app  = DrawApp(root)
    root.mainloop()
    exit()