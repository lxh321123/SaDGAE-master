import torch

class NB(object):
    def __init__(self, x, mu, theta):
        self.x = x
        self.mu = mu
        self.theta = theta

    def loss(self):
        factor1 = torch.exp(torch.lgamma(self.x + self.theta) - torch.lgamma(self.x + 1) - torch.lgamma(self.theta))
        factor2 = (self.theta / (self.theta + self.mu)) ** self.theta * (self.mu / (self.theta + self.mu)) ** self.x
        return factor1 * factor2

class ZINB(object):
    def __init__(self, x, pi, mu, theta):
        self.x = x
        self.pi = pi
        self.mu = mu
        self.theta = theta
        self.nb = NB(self.x, self.mu, self.theta)

    def loss(self):
        NB = self.nb.loss()
        zinb = torch.where(self.x == 0, self.pi, (1 - self.pi) * NB)
        return -torch.log(torch.clamp(zinb, min=1e-5, max=1e6)).mean()