import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    Deep Q-Network for processing Mario visual state observations.
    """
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        c, h, w = state_dim
        
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        def conv_out(size, k, s):
            return (size - (k - 1) - 1) // s + 1

        convw = conv_out(conv_out(conv_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv_out(conv_out(conv_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
