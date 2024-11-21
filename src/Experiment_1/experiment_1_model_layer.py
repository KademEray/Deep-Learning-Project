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