import torch
import torch.nn as nn
import torch.nn.functional as F
sp = nn.Softplus()
MeanAct = lambda x : torch.clamp(torch.exp(x), 1e-5, 1e6)
DispAct = lambda x : torch.clamp(sp(x), 1e-4, 1e4)

class AE_Encoder(nn.Module):
    def __init__(self, input_size, ae_en_1, ae_en_2, ae_en_3, output_size):
        super(AE_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, ae_en_1)
        self.fc2 = nn.Linear(ae_en_1, ae_en_2)
        self.fc3 = nn.Linear(ae_en_2, ae_en_3)
        self.fc4 = nn.Linear(ae_en_3, output_size)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))
        h = self.fc4(h)
        return h

class AE_Decoder(nn.Module):
    def __init__(self, output_size, ae_de_1, ae_de_2, ae_de_3, input_size):
        super(AE_Decoder, self).__init__()
        self.fc1 = nn.Linear(output_size, ae_de_1)
        self.fc2 = nn.Linear(ae_de_1, ae_de_2)
        self.fc3 = nn.Linear(ae_de_2, ae_de_3)
        self.fc4 = nn.Linear(ae_de_3, input_size)
        self.act = nn.ReLU()

    def forward(self, h):
        x_hat = self.act(self.fc1(h))
        x_hat = self.act(self.fc2(x_hat))
        x_hat = self.act(self.fc3(x_hat))
        x_hat = self.fc4(x_hat)
        return x_hat

class fc_layer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, x):
        x = self.fc(x)
        return x

class AE(nn.Module):
    def __init__(self, input_size, ae_en_1, ae_en_2, ae_en_3, output_size, ae_de_1, ae_de_2, ae_de_3):
        super(AE, self).__init__()
        self.ae_encoder = AE_Encoder(input_size, ae_en_1, ae_en_2, ae_en_3, output_size)
        self.ae_decoder = AE_Decoder(output_size, ae_de_1, ae_de_2, ae_de_3, input_size)
        self.fc = fc_layer(input_size)

    def forward(self, x):
        h = self.ae_encoder(x)
        x_hat = self.ae_decoder(h)

        sig = nn.Sigmoid()
        pi = sig(self.fc(x_hat))
        mu = MeanAct(self.fc(x_hat))
        theta = DispAct(self.fc(x_hat))

        return x_hat, h, pi, mu, theta