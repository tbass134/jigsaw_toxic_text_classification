import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab, hidden_dim, embedding_dim=300,num_classes=6):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=1
                          )
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        out = self.linear1(feature)
        out = self.linear2(out)
        out = self.fc(out)
        return out