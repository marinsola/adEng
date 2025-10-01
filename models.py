import torch
import torch.nn as nn

from engression.models import StoNet

class additive_engression(nn.Module):
    def __init__(self, K, input_dim, hidden_dim=100, output_dim=1, eng_layer=2,
                 marginal_hidden_dim=100, marginal_layers=2, marginal_dropout=0.0,
                 engressor_hidden_dim=100, engressor_dropout=0.0):
        super(additive_engression, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.engr_layer = eng_layer
        self.output_dim = output_dim
        self.K = K
        self.marginal_hidden_dim = marginal_hidden_dim
        self.marginal_layers = marginal_layers
        self.marginal_dropout = marginal_dropout
        self.engressor_hidden_dim = engressor_hidden_dim
        self.engressor_layers = engressor_layers
        self.engressor_dropout = engressor_dropout

        for k in range(K):
            setattr(self, f"engressor_{k}", StoNet(input_dim, output_dim, engressor_layers, engressor_hidden_dim, resblock=True))

        for p in  range(input_dim):
            layers = [nn.Linear(1, marginal_hidden_dim), nn.LeakyReLU(0.01)]
            for _ in range(marginal_layers - 2):
                layers += [nn.Linear(marginal_hidden_dim, marginal_hidden_dim), nn.LeakyReLU(0.01)]
                if marginal_dropout > 0:
                    layers += [nn.Dropout(marginal_dropout)]
            layers += [nn.Linear(marginal_hidden_dim, output_dim)]
            setattr(self, f"marginal_{p}", nn.Sequential(*layers))

    def predict(self, x, sample_size=100):
        outputs = []
        for p in range(self.input_dim):
            outputs.append(getattr(self, f"marginal_{p}")(x[:, p].unsqueeze(1)))
        outputs = torch.stack(outputs, dim=1)

        engressor_outputs = []
        for k in range(self.K):
            engressor_outputs.append(getattr(self, f"engressor_{k}").predict(x, sample_size=sample_size))
        engressor_outputs = torch.stack(engressor_outputs, dim=1)
        return torch.sum(torch.cat([outputs, engressor_outputs], dim=1), dim=1)

    def forward(self, x, double=False):
        outputs = []
        for p in range(self.input_dim):
            outputs.append(getattr(self, f"marginal_{p}")(x[:, p].unsqueeze(1)))
        outputs = torch.stack(outputs, dim=1)

        if double:
            outputs1, outputs2 = [], []
            for k in range(self.K):
                outs = getattr(self, f"engressor_{k}")(x, double=True)
                outputs1.append(outs[0])
                outputs2.append(outs[1])
            outputs1, outputs2 = torch.stack(outputs1, dim=1), torch.stack(outputs2, dim=1)
            return torch.sum(torch.cat([outputs, outputs1], dim=1), dim=1), torch.sum(torch.cat([outputs, outputs2], dim=1), dim=1)

        else:
            outputs_ = []
            for k in range(self.K):
                outputs_.append(getattr(self, f"engressor_{k}")(x))
            outputs_ = torch.stack(outputs_, dim=1)
            return torch.sum(torch.cat([outputs, outputs_], dim=1), dim=1)
