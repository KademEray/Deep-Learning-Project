import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
from experiment_7_model_layer import CustomLSTM, MultiInputLSTMWithGates, DualAttention
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

# Prüfen, ob CUDA verfügbar ist, und das Gerät auswählen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DynamicLSTMModel(nn.Module):
    """
    A PyTorch model that dynamically creates LSTM layers for each input dimension, 
    applies multi-input LSTM with gates, and uses dual attention mechanism.
    Args:
        seq_length (int): The length of the input sequences.
        hidden_dim (int): The number of features in the hidden state of the LSTM.
        input_sizes (list of int): A list of input dimensions for each LSTM layer.
        num_layers (int, optional): Number of recurrent layers. Default is 3.
        dropout (float, optional): Dropout probability. Default is 0.3.
    Raises:
        ValueError: If input_sizes is not a list of integers.
    Methods:
        forward(*inputs):
            Forward pass of the model. Takes multiple inputs corresponding to each LSTM layer.
            Args:
                *inputs: Variable length input list where each element is a tensor of shape 
                         (batch_size, seq_length, input_size).
            Returns:
                torch.Tensor: The output of the model after applying LSTM layers, multi-input LSTM, 
                              dual attention, and fully connected layers.
    """
    def __init__(self, seq_length, hidden_dim, input_sizes, num_layers=3, dropout=0.3):
        super(DynamicLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        # Überprüfen und sicherstellen, dass input_sizes eine Liste aus int-Werten ist
        if not all(isinstance(input_size, int) for input_size in input_sizes):
            raise ValueError("input_sizes muss eine Liste aus ganzzahligen Werten sein.")

        # Dynamische LSTM-Schichten für jede Eingabedimension
        self.lstm_layers = nn.ModuleList([
            CustomLSTM(input_size, hidden_dim, num_layers=num_layers, dropout=dropout)
            for input_size in input_sizes
        ])

        # Multi-Input LSTM und Attention
        self.multi_input_lstm = MultiInputLSTMWithGates(hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.dual_attention_layer = DualAttention(hidden_dim, hidden_dim)

        # Fully Connected Layer für jeden Zeitschritt
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)  # Endausgabe für 10 Features
        )


    def forward(self, *inputs):
        outputs = [lstm_layer(input_data) for lstm_layer, input_data in zip(self.lstm_layers, inputs)]
        processed_signals = [output[0] for output in outputs]

        self_gate = torch.sigmoid(self.dual_attention_layer(processed_signals[0]))
        Index = processed_signals[1] if len(processed_signals) > 1 else processed_signals[0]

        # Multi-Input LSTM zur Fusion
        if len(processed_signals) == 1:
            combined_input, _ = self.multi_input_lstm(processed_signals[0], processed_signals[0], processed_signals[0],
                                                      Index, self_gate)
        else:
            combined_input, _ = self.multi_input_lstm(processed_signals[0], processed_signals[1], processed_signals[2],
                                                      Index, self_gate)

        attended = self.dual_attention_layer(combined_input)
        output = self.fc(attended)
        return output


class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for sequence data.
    Args:
        input_dim (int): Number of input channels/features.
        seq_length (int): Length of the input sequences.
        output_dim (int): Number of output classes.
    Attributes:
        conv1 (nn.Conv1d): First 1D convolutional layer.
        conv2 (nn.Conv1d): Second 1D convolutional layer.
        relu (nn.ReLU): ReLU activation function.
        flattened_dim (int): Dimension of the flattened output from the convolutional layers.
        fc (nn.Linear): Fully connected layer for classification.
    Methods:
        forward(x):
            Defines the forward pass of the network.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, output_dim).
    """
    def __init__(self, input_dim, seq_length, output_dim):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.output_dim = output_dim

        # Dynamische Berechnung der Flattened-Dimension
        self.flattened_dim = None
        with torch.no_grad():
            temp_input = torch.zeros(1, input_dim, seq_length)
            temp_output = self.relu(self.conv2(self.relu(self.conv1(temp_input))))
            self.flattened_dim = temp_output.numel()

        # Fully Connected Layer
        self.fc = nn.Linear(self.flattened_dim, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)  # Transponiere für Conv1d (Batch, Features, Seq)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Dynamisches Flattening
        x = self.fc(x)
        return x


class FusionModelWithCNN(nn.Module):
    """
    A PyTorch model that fuses multiple LSTM models and a CNN model for time series prediction.
    Args:
        hidden_dim (int): The number of hidden units in the LSTM and CNN models. Default is 64.
        seq_length (int): The length of the input sequences. Default is 30.
        output_features (int): The number of output features. Default is 10.
        group_features (dict): A dictionary containing the input dimensions for each group of features.
    Attributes:
        hidden_dim (int): The number of hidden units in the LSTM and CNN models.
        seq_length (int): The length of the input sequences.
        output_features (int): The number of output features.
        input_sizes (dict): A dictionary containing the input dimensions for each group of features.
        standard_model (DynamicLSTMModel): The LSTM model for the "Standard" group.
        indicators_1_model (DynamicLSTMModel): The LSTM model for the "Indicators_Group_1" group.
        indicators_2_model (DynamicLSTMModel): The LSTM model for the "Indicators_Group_2" group.
        cnn_model (SimpleCNN): The CNN model for fusing the inputs.
        fusion_fc (nn.Sequential): The fully connected layer for final fusion of the outputs.
    Methods:
        forward(Y, X1, X2):
            Performs a forward pass through the model.
            Args:
                Y (torch.Tensor): The input tensor for the "Standard" group.
                X1 (torch.Tensor): The input tensor for the "Indicators_Group_1" group.
                X2 (torch.Tensor): The input tensor for the "Indicators_Group_2" group.
            Returns:
                torch.Tensor: The output tensor after fusion.
    """
    def __init__(self, hidden_dim=64, seq_length=30, output_features=10, group_features=None):
        super(FusionModelWithCNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.output_features = output_features

        # Extrahiere die Eingabedimensionen aus den Gruppen
        self.input_sizes = {
            key: len(value) if isinstance(value, list) else value
            for key, value in group_features.items()
        }

        # Initialisiere die LSTM-Modelle mit den Dimensionen
        self.standard_model = DynamicLSTMModel(
            seq_length, hidden_dim, [self.input_sizes["Standard"]]
        ).to(device)

        self.indicators_1_model = DynamicLSTMModel(
            seq_length, hidden_dim, [self.input_sizes["Indicators_Group_1"]]
        ).to(device)

        self.indicators_2_model = DynamicLSTMModel(
            seq_length, hidden_dim, [self.input_sizes["Indicators_Group_2"]]
        ).to(device)

        # Initialisiere das CNN-Modell für die Fusion
        total_input_dim = sum(self.input_sizes.values())
        self.cnn_model = SimpleCNN(
            input_dim=total_input_dim, seq_length=seq_length, output_dim=hidden_dim
        )

        # Fully Connected Layer zur finalen Fusion
        self.fusion_fc = nn.Sequential(
            nn.Linear(output_features * 3 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_features)
        )

    def forward(self, Y, X1, X2):
        # Vorhersagen mit den LSTM-Modellen
        standard_pred = self.standard_model(Y)
        indicators_1_pred = self.indicators_1_model(X1)
        indicators_2_pred = self.indicators_2_model(X2)

        # CNN-Verarbeitung
        combined_input = torch.cat((Y, X1, X2), dim=2)
        cnn_output = self.cnn_model(combined_input)

        # Kombiniere alle Ergebnisse
        cnn_output = cnn_output.unsqueeze(1).repeat(1, combined_input.size(1), 1)
        combined = torch.cat((standard_pred, indicators_1_pred, indicators_2_pred, cnn_output), dim=2)

        # Finale Ausgabe durch Fusion Layer
        output = self.fusion_fc(combined)
        return output


def train_fusion_model_with_cnn(X_standard, X_group1, X_group2, Y_samples, hidden_dim, batch_size, learning_rate, epochs, model_dir):
    """
    Trains a fusion model with CNN using the provided datasets and parameters.
    Args:
        X_standard (torch.Tensor): Standard input features tensor.
        X_group1 (torch.Tensor): Group 1 input features tensor.
        X_group2 (torch.Tensor): Group 2 input features tensor.
        Y_samples (torch.Tensor): Target output tensor.
        hidden_dim (int): Dimension of the hidden layers.
        batch_size (int): Size of each batch for training.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train the model.
        model_dir (str): Directory to save the trained model and plots.
    Returns:
        str: Path to the saved model file.
    Prints:
        Shapes of the input tensors.
        Training and validation loss for each epoch.
        Validation MSE and RMSE for each epoch.
        Paths to the saved model and plots.
    """
    print(f"X_standard Shape: {X_standard.shape}")
    print(f"X_group1 Shape: {X_group1.shape}")
    print(f"X_group2 Shape: {X_group2.shape}")
    print(f"Y_samples Shape: {Y_samples.shape}")

    group_features = {
        "Standard": X_standard.shape[-1],  # Einzelne Eingabedimension
        "Indicators_Group_1": X_group1.shape[-1],
        "Indicators_Group_2": X_group2.shape[-1],
    }

    input_sizes = [group_features["Standard"], group_features["Indicators_Group_1"],
                   group_features["Indicators_Group_2"]]

    model = FusionModelWithCNN(
        hidden_dim=hidden_dim,
        seq_length=X_standard.shape[1],  # Die Sequenzlänge aus den Eingabedaten extrahieren
        output_features=Y_samples.size(-1),
        group_features=group_features,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataset = torch.utils.data.TensorDataset(X_standard, X_group1, X_group2, Y_samples)

    # Aufteilen in Trainings- und Validierungsdaten
    val_split = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [val_split, len(dataset) - val_split])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    scaler = torch.amp.GradScaler('cuda')

    model.train()
    train_loss_values = []
    val_loss_values = []
    val_mse_values = []
    val_rmse_values = []

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (X_standard, X_group1, X_group2, y) in enumerate(train_loader):
            X_standard, X_group1, X_group2, y = (
                X_standard.float().to(device),
                X_group1.float().to(device),
                X_group2.float().to(device),
                y.float().to(device),
            )

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                output = model(X_standard, X_group1, X_group2)
                output = output[:, -30:, :]  # Kürzt die Ausgabe auf die letzten 30 Zeitschritte

                loss = criterion(output, y)

            scaler.scale(loss).backward()  # Gradient scaling
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        # Durchschnittlicher Trainingsverlust
        epoch_loss = running_loss / len(train_loader)
        train_loss_values.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss:.6f}")

        # Validierungsmodus
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        with torch.no_grad():
            for X_standard, X_group1, X_group2, y in val_loader:
                X_standard, X_group1, X_group2, y = (
                    X_standard.float().to(device),
                    X_group1.float().to(device),
                    X_group2.float().to(device),
                    y.float().to(device),
                )

                output = model(X_standard, X_group1, X_group2)
                output = output[:, -30:, :]  # Kürzt die Ausgabe auf die letzten 30 Zeitschritte

                loss = criterion(output, y)
                val_loss += loss.item()

                # MSE berechnen
                mse = nn.MSELoss()(output, y).item()
                val_mse += mse

        # Durchschnittlicher Validierungsverlust und Metriken
        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        val_rmse = val_mse ** 0.5

        val_loss_values.append(val_loss)
        val_mse_values.append(val_mse)
        val_rmse_values.append(val_rmse)

        print(f"Validation Loss: {val_loss:.6f}, MSE: {val_mse:.6f}, RMSE: {val_rmse:.6f}")

        model.train()  # Zurück in den Trainingsmodus

    # Modell speichern
    model_path = os.path.join(model_dir, "fusion_model_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Fusion Model gespeichert unter {model_path}")

    # Plots erstellen
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_loss_values, label="Training Loss", color="blue")
    plt.plot(range(1, epochs + 1), val_loss_values, label="Validation Loss", color="orange")
    plt.title("Training und Validation Loss über die Epochen")
    plt.xlabel("Epoche")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(model_dir, "loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss-Plot gespeichert unter {loss_plot_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), val_mse_values, label="Validation MSE", color="green")
    plt.plot(range(1, epochs + 1), val_rmse_values, label="Validation RMSE", color="red")
    plt.title("Validation MSE und RMSE über die Epochen")
    plt.xlabel("Epoche")
    plt.ylabel("Fehler")
    plt.legend()
    plt.grid(True)
    metrics_plot_path = os.path.join(model_dir, "metrics_plot.png")
    plt.savefig(metrics_plot_path)
    plt.close()
    print(f"Metriken-Plot gespeichert unter {metrics_plot_path}")

    return model_path


# Hauptfunktion
if __name__ == '__main__':

    # Setzen der Zufallszahlenseeds für Python, NumPy und PyTorch
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    # Falls Sie CUDA verwenden, setzen Sie auch den Seed für CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)  # Für Multi-GPU, falls verwendet

    # Zusätzliche Einstellungen, um deterministisches Verhalten sicherzustellen
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Definiert die Verzeichnisse für gespeicherte und trainierte Daten
    samples_dir = "./Data/Samples"  # Verzeichnis mit den vorbereiteten Trainingssequenzen
    model_dir = "./Data/Models"  # Verzeichnis zum Speichern des trainierten Modells

    # Wichtige Modell- und Trainingsparameter
    seq_length = 50  # Länge der Eingabesequenzen in Tagen
    hidden_dim = 64  # Dimension der versteckten Schicht im Modell
    batch_size = 256  # Anzahl der Samples pro Batch
    learning_rate = 0.001  # Lernrate für den Optimizer
    epochs = 50  # Anzahl der Trainingsdurchläufe (Epochen)

    with open(os.path.join(samples_dir, 'Standard_X.pkl'), 'rb') as f:
        X_standard = pickle.load(f)  # Standard-Feature-Gruppe
    with open(os.path.join(samples_dir, 'Indicators_Group_1_X.pkl'), 'rb') as f:
        X_group1 = pickle.load(f)  # Momentum-Indikatorgruppe
    with open(os.path.join(samples_dir, 'Indicators_Group_2_X.pkl'), 'rb') as f:
        X_group2 = pickle.load(f)  # Volatilität und Trend-Indikatorgruppe
    with open(os.path.join(samples_dir, 'Standard_Y.pkl'), 'rb') as f:
        Y_samples = pickle.load(f)  # Zielwerte für die Vorhersage

    print(
        f"Geladene Y_samples Form: {Y_samples.shape}")  # Ausgabe der Form, um zu bestätigen, dass die Daten korrekt geladen wurden (sollte [Anzahl Samples, 1] sein)

    # Starte das Training des Fusion Models mit den geladenen Daten und Parametern
    train_fusion_model_with_cnn(
        X_standard,
        X_group1,
        X_group2,
        Y_samples,
        hidden_dim,
        batch_size,
        learning_rate,
        epochs,
        model_dir
    )