import math
import torch
import torch
import torch.nn as nn
# from torch.distributions.normal import Normal
import numpy as np
# from layers.Layer import WeightGenerator, CustomLinear

# from functools import reduce
# from operator import mul
import torch.nn.functional as F





class SpatialPointModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpatialPointModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTM layer
        _, (hidden, _) = self.lstm(x)  # We only need the last hidden state
        # Pass the last hidden state through a fully connected layer
        out = self.fc(hidden.squeeze(0))
        return out

class Model(nn.Module):
    def __init__(self, configs):        # (self,):
        super(Model, self).__init__()
        self.num_points = configs.num_nodes
        self.input_dim = configs.input_dim  
        self.hidden_dim = configs.hidden_dim
        self.output_dim = configs.pred_len
        self.models = nn.ModuleList([SpatialPointModel(self.input_dim, self.hidden_dim, self.output_dim) for _ in range(self.num_points)])

    def forward(self, x):
        x = x[:,:,:,0]  # x shape: [batch_size, seq_length, num_points,feature_dims]
        batch_size, seq_length, num_points = x.shape
        assert num_points == self.num_points, "Number of spatial points does not match."

        # Process each spatial point separately
        outputs = []
        for i in range(num_points):
            point_data = x[:, :, i].unsqueeze(-1)  # Shape: [batch_size, seq_length, 1]
            point_output = self.models[i](point_data)  # Shape: [batch_size, output_dim]
            outputs.append(point_output)

        # Combine the outputs for all points
        # Shape: [batch_size, output_dim, num_points]
        final_output = torch.stack(outputs, dim=-1)
        return final_output

# # Example parameters
# num_points = 112  # Number of spatial points
# input_dim = 1     # Each point has 1 feature per time step
# hidden_dim = 64   # Size of LSTM hidden layer
# output_dim = 1    # Predicted future SST values for pred_len
# pred_len = 10     # Length of the prediction sequence

# # Instantiate the model
# model = SSTModel(num_points, input_dim, hidden_dim, output_dim)

# # Example input data
# batch_size = 32
# seq_length = 90
# example_input = torch.randn(batch_size, seq_length, num_points)  # Random data

# # Forward pass
# output = model(example_input)
# print("Output shape:", output.shape)  # Expected shape: [32, 1, 112]
