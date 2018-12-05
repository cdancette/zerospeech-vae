import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, input_size, num_components, sigmoid=False, hidden_layers=1):
        super(VAE, self).__init__()

        self.sigmoid = sigmoid
        self.input_size = input_size
        self.num_components = num_components

        fc_encode = [
            nn.Linear(input_size, 400),
            nn.ReLU(),
        ]

        for _ in range(hidden_layers - 1):
            fc_encode.append(nn.Linear(400, 400))
            fc_encode.append(nn.ReLU())

        self.fc_encode = nn.Sequential(*fc_encode)
        self.fc21 = nn.Linear(400, num_components)
        self.fc22 = nn.Linear(400, num_components)

        fc_decode = [
            nn.Linear(num_components, 400),
            nn.ReLU()
        ]

        for _ in range(hidden_layers -1):
            fc_decode.append(nn.Linear(400, 400))
            fc_decode.append(nn.ReLU())

        self.fc_decode = nn.Sequential(*fc_decode)
        self.fc4 = nn.Linear(400, input_size)

    def encode(self, x):
        h1 = self.fc_encode(x)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.fc_decode(z)

        output = self.fc4(h3)
        if self.sigmoid:
            return torch.sigmoid(output)
        return output

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class OldVAE(nn.Module):

    def __init__(self, input_size, num_components, sigmoid=False):
        super(VAE, self).__init__()

        self.sigmoid = sigmoid
        self.input_size = input_size
        self.num_components = num_components

        self.fc1 = nn.Linear(input_size, 400)
        self.fc21 = nn.Linear(400, num_components)
        self.fc22 = nn.Linear(400, num_components)
        self.fc3 = nn.Linear(num_components, 400)
        self.fc4 = nn.Linear(400, input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))

        output = self.fc4(h3)
        if self.sigmoid:
            return torch.sigmoid(output)
        return output

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar