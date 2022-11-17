"""
Baseline model for Features to point estimates. 
Model takes n features, and a point to return the (n-dimensional) field value.
"""

from torch import nn

# Simple MLP model
class FeatureToPoint(nn.Module):
    def __init__(self, n_in, n_out, n_hidden, n_layers, activation=nn.ReLU):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_in, n_hidden))
        self.layers.append(activation())
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
            self.layers.append(activation())
        self.layers.append(nn.Linear(n_hidden, n_out))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    model = FeatureToPoint(8, 1, 64, 2)
    print(model)