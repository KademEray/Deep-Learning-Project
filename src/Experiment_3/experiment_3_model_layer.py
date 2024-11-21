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
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        """
        Initialisiert die CustomLSTM-Klasse.

        Parameter:
        - input_dim (int): Die Dimension der Eingabe für jedes Zeitschritt.
        - hidden_dim (int): Die Anzahl der LSTM-Einheiten im versteckten Zustand.
        - num_layers (int): Die Anzahl der LSTM-Schichten. Standardwert ist 2.
        - dropout (float): Die Abbruchrate für das Dropout. Wird zwischen den LSTM-Schichten angewendet.
        """
        super(CustomLSTM, self).__init__()

        # LSTM-Schicht: Hier wird eine LSTM-Schicht erstellt, die eine Sequenz verarbeitet.
        # - input_dim gibt die Dimension der Eingabe für jedes Zeitschritt an.
        # - hidden_dim legt die Anzahl der Einheiten im versteckten Zustand fest.
        # - num_layers gibt an, wie viele LSTM-Schichten gestapelt werden.
        # - dropout sorgt dafür, dass die Wahrscheinlichkeit für Dropout in den oberen LSTM-Schichten
        #   bei jedem Zeitschritt 0.2 beträgt, um Überanpassung zu verhindern.
        # - batch_first=True stellt sicher, dass die erste Dimension die Batch-Größe ist.
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        output, (hidden, _) = self.lstm(x)
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

        # Anwenden des Gating-Signals zur Modulation der Nebensignale
        X_p = X_p * self_gate
        X_n = X_n * self_gate
        Index = Index * self_gate

        # Kombinieren der Eingaben entlang der letzten Dimension
        concat_input = torch.cat((Y, X_p, X_n, Index), dim=2)

        # LSTM-Vorwärtsdurchlauf auf den kombinierten Eingang anwenden
        output, (hidden, _) = self.lstm(concat_input)

        return output, hidden[0]


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