import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_layers, hidden_dim: list[int]) -> None:
        super().__init__()
        assert num_layers == len(hidden_dim), "Check num_layers and hidden_dim."
            
        layers = [nn.Flatten()]
        d_prev = in_features
        for dim in hidden_dim:
            layers += [nn.Linear(d_prev, dim), nn.ReLU()]
            d_prev = dim
        
        self.layers = nn.Sequential(*layers, nn.Linear(d_prev, out_features))
        
        
    def forward(self, x):
        return self.layers(x)