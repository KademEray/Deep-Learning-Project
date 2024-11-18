import torch
import torch.nn as nn
import random
import numpy as np

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
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.4):
        """
        Verbesserter LSTM mit Layernorm und adaptiver Residualverbindung.
        """
        super(CustomLSTM, self).__init__()

        # Sicherstellen, dass input_dim ein einzelner Wert ist
        if isinstance(input_dim, list):  # Füge diese Überprüfung hinzu
            input_dim = input_dim[0]

        # LSTM-Schicht
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )


        # Residualverbindung
        self.residual_fc = nn.Linear(input_dim, hidden_dim)

        # Adaptive Residualgewichtung
        self.residual_weight = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # Layer-Normalisierung
        self.layernorm = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout * 0.5)  # Separates Dropout für Residualverbindung

    def forward(self, x):
        """
        Vorwärtsdurchlauf des LSTM mit adaptiver Residualverbindung.

        Parameter:
        - x (Tensor): Eingabe mit Form (Batch, Sequenz, Features).

        Rückgabe:
        - output (Tensor): LSTM-Ausgabe.
        - hidden (Tensor): Letzter versteckter Zustand der LSTM-Schicht.
        """
        identity = x  # Original-Eingabe speichern für Residualverbindung

        # LSTM-Schicht
        lstm_out, (hidden, _) = self.lstm(x)

        # Residualverbindung
        residual = self.residual_fc(identity)
        residual = self.residual_dropout(residual)  # Residual-Dropout

        # Adaptive Gewichtung
        weighted_residual = residual * self.residual_weight(lstm_out)

        # Layernorm und Dropout
        output = self.layernorm(lstm_out + weighted_residual)  # Residual hinzufügen
        output = self.dropout(output)

        return output, hidden



class MultiInputLSTMWithGates(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        """
        Initialisiert die MultiInputLSTMWithGates-Klasse, eine LSTM-Schicht mit mehreren Eingaben und Gate-Steuerung.

        Parameter:
        - hidden_dim (int): Die Dimension des versteckten Zustands.
        - output_dim (int): Die Dimension der LSTM-Ausgabe.
        - num_layers (int): Die Anzahl der LSTM-Schichten. Standardwert ist 2.
        - dropout (float): Die Abbruchrate für das Dropout zwischen den LSTM-Schichten.
        """
        super(MultiInputLSTMWithGates, self).__init__()

        # LSTM-Schicht, die mehrere Eingaben verarbeitet und alle Eingaben in einer einzigen Sequenz kombiniert.
        # - hidden_dim * 4: Die Eingabedimension entspricht der Summe aller versteckten Dimensionen der vier Eingaben (Y, X_p, X_n, Index).
        # - output_dim: Die Dimension der Ausgabe für den LSTM-Schritt.
        # - num_layers und dropout werden analog der üblichen LSTM-Konfiguration angewendet.
        self.lstm = nn.LSTM(hidden_dim * 4, output_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, Y, X_p, X_n, Index, self_gate):
        X_p = X_p * self_gate
        X_n = X_n * self_gate
        Index = Index * self_gate

        concat_input = torch.cat((Y, X_p, X_n, Index), dim=2)

        output, (hidden, _) = self.lstm(concat_input)

        return output, hidden


class DualAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialisiert die DualAttention-Klasse, die eine doppelte Attention-Schicht mit Zeit- und Feature-Attention bietet.

        Parameter:
        - input_dim (int): Die Dimension der Eingabe.
        - output_dim (int): Die Dimension der Aufmerksamkeit (Attention) für die Ausgabe.
        """
        super(DualAttention, self).__init__()

        # Lineare Schicht für die Zeit-Attention (Zeitdimension)
        self.time_attention = nn.Linear(input_dim, output_dim)

        # Lineare Schicht für die Feature-Attention (Input-Feature-Dimension)
        self.input_attention = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Führt den Vorwärtsdurchlauf durch und berechnet die zeit- und feature-basierte Attention-Gewichtung.

        Parameter:
        - x (Tensor): Die Eingabe, kann entweder die Form (Batch, Sequenz, Features) oder (Batch, Features) haben.

        Rückgabe:
        - attended (Tensor): Das mit zeit- und feature-basierter Attention gewichtete Ergebnis.
        """

        # Unterscheiden, ob die Eingabe eine Sequenz (3D) oder ein Batch (2D) ist
        if x.dim() == 3:
            # Berechnung der Attention-Gewichte für die Zeitdimension (Sequenzlänge)
            # Softmax entlang der Sequenzdimension dim=1
            time_weights = torch.softmax(self.time_attention(x), dim=1)

            # Berechnung der Attention-Gewichte für die Input-Feature-Dimension
            # Softmax entlang der letzten Dimension dim=2 (Features)
            input_weights = torch.softmax(self.input_attention(x), dim=2)

        elif x.dim() == 2:
            # Falls die Eingabe keine Sequenz ist, berechne die Attention-Gewichte für Batch-Dimensionen
            # Zeitdimension-Weights (Batch-Ebene), Softmax entlang dim=0
            time_weights = torch.softmax(self.time_attention(x), dim=0)

            # Feature-Dimension-Gewichtung für 2D-Eingaben (Softmax entlang dim=1)
            input_weights = torch.softmax(self.input_attention(x), dim=1)

            # Füge Dimensionen hinzu, um sie mit den 3D-Gewichtungen konsistent zu machen
            time_weights = time_weights.unsqueeze(1)
            input_weights = input_weights.unsqueeze(0)

        else:
            # Fehler, falls x eine unerwartete Dimension aufweist
            raise ValueError("Eingabe hat eine unerwartete Dimension")

        # Anwenden der berechneten Attention-Gewichte auf die Eingabe x:
        # - x * time_weights berücksichtigt die Zeitdimension
        # - * input_weights berücksichtigt die Feature-Dimension
        attended = x * time_weights * input_weights

        return attended


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