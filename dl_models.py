import torch
from torch import nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


class CNN(nn.Module):
    def __init__(self, input_dim, num_filters, filter_sizes, output_dim, dropout):
        super(CNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        text = text.unsqueeze(1)

        conved = [nn.functional.relu(conv(text)) for conv in self.convs]

        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))  # [batch_size, num_filters * len(filter_sizes)]

        return self.fc(cat)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        # self.fc2 = nn.Linear(550, 350)
        # self.fc3 = nn.Linear(350, 150)
        # self.fc4 = nn.Linear(150, output_size)

    def forward(self, input):
        hidden, carry = torch.randn(self.num_layers, self.hidden_size,device=input.device), torch.randn(self.num_layers, self.hidden_size, device=input.device)
        output, (hidden, carry) = self.lstm(input, (hidden, carry))
        output = self.fc1(output)
        # output = self.fc2(output)
        # output = self.fc3(output)
        # output = self.fc4(output)
        return output
