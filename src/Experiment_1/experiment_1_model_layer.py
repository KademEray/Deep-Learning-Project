import random
import numpy as np
import torch
import torch.nn as nn

# Setzen der Zufallszahlenseeds für Python, NumPy und PyTorch
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# Falls CUDA verwendet wird, den Seed auch für CUDA setzen
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)  # Für Multi-GPU, falls verwendet

# Zusätzliche Einstellungen, um deterministisches Verhalten sicherzustellen
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CustomLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout, batch_first=True
        )

    def forward(self, x):
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        output, (hidden, _) = self.lstm(x)
        return output, hidden  # Return the full sequence output


class MultiInputLSTMWithGates(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(MultiInputLSTMWithGates, self).__init__()
        self.lstm = nn.LSTM(hidden_dim * 4, output_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, Y, X_p, X_n, Index, self_gate):
        # Erweitern Sie self_gate auf [Batch, Seq_length, Hidden_dim]
        if self_gate.dim() == 2:  # Wenn self_gate die Dimension [Batch, Hidden_dim] hat
            self_gate = self_gate.unsqueeze(1)  # Hinzufügen der Zeitdimension

        # Modulate the signals with self_gate
        X_p = X_p * self_gate
        X_n = X_n * self_gate
        Index = Index * self_gate

        # Concatenate inputs along the feature dimension
        concat_input = torch.cat((Y, X_p, X_n, Index), dim=2)  # [batch_size, seq_length, hidden_dim * 4]

        # Apply LSTM
        output, (hidden, _) = self.lstm(concat_input)

        return output, hidden[0]


class DualAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DualAttention, self).__init__()
        self.time_attention = nn.Linear(input_dim, output_dim)
        self.input_attention = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if x.dim() == 3:
            time_weights = torch.softmax(self.time_attention(x), dim=1)
            input_weights = torch.softmax(self.input_attention(x), dim=2)
        elif x.dim() == 2:
            time_weights = torch.softmax(self.time_attention(x), dim=0)
            input_weights = torch.softmax(self.input_attention(x), dim=1)
            time_weights = time_weights.unsqueeze(1)
            input_weights = input_weights.unsqueeze(0)
        else:
            raise ValueError("Eingabe hat eine unerwartete Dimension")

        attended = x * time_weights * input_weights
        # Entfernen Sie die Summierung über die Zeitdimension
        # attended = torch.sum(attended, dim=1)

        return attended  # Behalten Sie die Form [Batch, Seq_length, Features] bei


class ComplexFusionModel(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=64, output_dim=1, num_layers=3, dropout=0.3):
        """
        Initialisiert das ComplexFusionModel, das mehrere LSTM-Schichten und Attention-Mechanismen kombiniert, um
        verschiedene Eingabesignale und Features zu integrieren und die finale Ausgabe zu berechnen.

        Parameter:
        - input_dim (int): Eingabedimension der Y-Signale.
        - hidden_dim (int): Dimension der verborgenen Zustände der LSTM-Schichten.
        - output_dim (int): Ausgabe-Dimension, meist 1 für Regressionsaufgaben.
        - num_layers (int): Anzahl der LSTM-Schichten.
        - dropout (float): Dropout-Rate zur Vermeidung von Überanpassung.
        """
        super(ComplexFusionModel, self).__init__()

        # Initialisiere die Hauptsignalverarbeitungsschicht (Y-Signale) mit CustomLSTM
        self.Y_layer = CustomLSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)

        # Initialisiere die LSTM-Schichten für positive Nebensignale (X_p) mit drei CustomLSTM-Schichten
        self.X_p_layers = nn.ModuleList(
            [CustomLSTM(15, hidden_dim, num_layers=num_layers, dropout=dropout) for _ in range(3)]
        )

        # Initialisiere die LSTM-Schichten für negative Nebensignale (X_n)
        self.X_n_layers = nn.ModuleList(
            [CustomLSTM(15, hidden_dim, num_layers=num_layers, dropout=dropout) for _ in range(3)]
        )

        # Initialisiere die LSTM-Schichten für Indexsignale
        self.Index_layers = nn.ModuleList(
            [CustomLSTM(15, hidden_dim, num_layers=num_layers, dropout=dropout) for _ in range(3)]
        )

        # Multi-Input LSTM mit Gating für die Fusion der Signale
        self.multi_input_lstm = MultiInputLSTMWithGates(hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout)

        # Dual Attention Layer zur Gewichtung der relevanten Informationen nach Zeit und Features
        self.dual_attention_layer = DualAttention(hidden_dim, hidden_dim)

        # Finaler vollständig verbundener Layer (Fully Connected Layer) für die Ausgabe
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, Y, X_p, X_n, Index):
        """
        Führt den Vorwärtsdurchlauf aus, indem verschiedene Signalquellen integriert, Attention-Mechanismen
        angewendet und eine finale Prognose erstellt wird.

        Parameter:
        - Y (Tensor): Hauptsignal-Tensor.
        - X_p (Tensor): Positive Nebensignale (z. B. technische Indikatoren).
        - X_n (Tensor): Negative Nebensignale.
        - Index (Tensor): Indexsignale.

        Rückgabe:
        - output (Tensor): Die finale Prognose des Modells.
        """

        # Verarbeite das Hauptsignal (Y) und berechne die zeitliche Attention
        Y_out, _ = self.Y_layer(Y)
        self_attention = self.dual_attention_layer(Y_out)

        # Erzeuge ein Gate-Signal (self_gate) zur Kontrolle der Nebensignale basierend auf der Attention des Hauptsignals
        self_gate = torch.sigmoid(self_attention).unsqueeze(1)

        # Verarbeitung der positiven Nebensignale (X_p) durch die jeweiligen LSTM-Schichten
        # und Mittelung der Ergebnisse
        X_p_out = torch.mean(torch.stack([layer(X_p)[0] for layer in self.X_p_layers]), dim=0)

        # Verarbeitung der negativen Nebensignale (X_n) und Mittelung
        X_n_out = torch.mean(torch.stack([layer(X_n)[0] for layer in self.X_n_layers]), dim=0)

        # Verarbeitung der Indexsignale und Mittelung
        Index_out = torch.mean(torch.stack([layer(Index)[0] for layer in self.Index_layers]), dim=0)

        # Multi-Input LSTM mit Gate-Kontrolle: integriert die verschiedenen Signalquellen
        fusion_output, _ = self.multi_input_lstm(Y_out, X_p_out, X_n_out, Index_out, self_gate)

        # Wendet eine zweite Attention-Layer an, um die relevantesten Informationen hervorzuheben
        attention_out = self.dual_attention_layer(fusion_output)

        # Finale Ausgabe durch die Fully Connected Layer
        output = self.fc_layers(attention_out)

        return output
