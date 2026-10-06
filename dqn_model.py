"""
Deep Q-Network (DQN) architecture.
Reproduces the exact network described in Section 4.1 of
Mnih et al., "Playing Atari with Deep Reinforcement Learning" (DeepMind, 2013).

Architecture:
    Input:   84 x 84 x 4   (4 stacked grayscale frames, see atari_preprocessing.py)
    Conv1:   16 filters, 8x8, stride 4, ReLU        -> 20 x 20 x 16
    Conv2:   32 filters, 4x4, stride 2, ReLU        ->  9 x  9 x 32
    Flatten:                                        -> 2592
    FC1:     256 units, ReLU                        -> 256
    Output:  linear, one unit per valid action       -> num_actions  (raw Q-values)

Run this file directly to build the network and verify every intermediate
tensor shape against the numbers stated in the paper:
    python3 dqn_model.py
"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    PyTorch implementation of the paper's Q-network Q(s, a; theta).

    Design choices (see DQN_Architecture_Guide.md for the full "why"):
      - Takes the STATE only as input (the 84x84x4 stacked frames).
      - Produces ONE output per action (not one output per (state, action)
        pair). This lets you compute Q-values for every action with a
        single forward pass, which is required by both training (argmax
        over actions for the Bellman target) and acting (argmax over
        actions for the greedy policy).
      - No pooling layers - the paper's conv layers only use strided
        convolutions to reduce spatial size, preserving precise spatial
        information useful for detecting small, fast-moving game objects
        (a ball, a bullet) that pooling could blur or discard.
      - No activation on the output layer - Q-values are unbounded real
        numbers (can be negative, e.g. after reward clipping to -1), so a
        ReLU or sigmoid on the output would be mathematically wrong.
    """

    def __init__(self, in_channels: int = 4, num_actions: int = 4):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.relu = nn.ReLU()

        # Compute the flattened size analytically instead of hard-coding
        # "2592" so the network still works if you ever change the input
        # resolution away from 84x84.
        self._flatten_size = self._infer_flatten_size(in_channels)

        self.fc1 = nn.Linear(self._flatten_size, 256)
        self.out = nn.Linear(256, num_actions)

        self._init_weights()

    def _infer_flatten_size(self, in_channels: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            x = self.relu(self.conv1(dummy))
            x = self.relu(self.conv2(x))
            return x.numel()

    def _init_weights(self):
        # The paper does not specify an initialization scheme; Kaiming
        # (He) initialization is the modern standard choice for ReLU
        # networks and is used here for reproducibility.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 4, 84, 84) float tensor, pixel values already
           normalized to [0, 1] (see preprocessing note in the guide).
        returns: (batch, num_actions) raw Q-value estimates.
        """
        x = self.relu(self.conv1(x))   # -> (batch, 16, 20, 20)
        x = self.relu(self.conv2(x))   # -> (batch, 32, 9, 9)
        x = x.flatten(start_dim=1)     # -> (batch, 2592)
        x = self.relu(self.fc1(x))     # -> (batch, 256)
        q_values = self.out(x)         # -> (batch, num_actions)
        return q_values


def _self_test():
    print("Building DQN and verifying layer shapes against the paper...\n")

    num_actions = 4  # e.g. Breakout has 4 valid actions
    net = DQN(in_channels=4, num_actions=num_actions)

    batch_size = 8
    dummy_input = torch.rand(batch_size, 4, 84, 84)  # simulates preprocessed frames

    # Hook into each layer to print intermediate shapes.
    x = dummy_input
    x1 = net.relu(net.conv1(x))
    x2 = net.relu(net.conv2(x1))
    x_flat = x2.flatten(start_dim=1)
    x_fc = net.relu(net.fc1(x_flat))
    q = net.out(x_fc)

    print(f"Input                : {tuple(dummy_input.shape)}  (expected (batch, 4, 84, 84))")
    print(f"After Conv1 + ReLU    : {tuple(x1.shape)}  (expected (batch, 16, 20, 20))")
    print(f"After Conv2 + ReLU    : {tuple(x2.shape)}  (expected (batch, 32, 9, 9))")
    print(f"After Flatten         : {tuple(x_flat.shape)}  (expected (batch, 2592))")
    print(f"After FC1 + ReLU      : {tuple(x_fc.shape)}  (expected (batch, 256))")
    print(f"Output (Q-values)     : {tuple(q.shape)}  (expected (batch, {num_actions}))")

    n_params = sum(p.numel() for p in net.parameters())
    print(f"\nTotal trainable parameters: {n_params:,}")

    assert tuple(x1.shape) == (batch_size, 16, 20, 20)
    assert tuple(x2.shape) == (batch_size, 32, 9, 9)
    assert tuple(x_flat.shape) == (batch_size, 2592)
    assert tuple(x_fc.shape) == (batch_size, 256)
    assert tuple(q.shape) == (batch_size, num_actions)
    print("\nSelf-test passed: all shapes match the paper's architecture.")

    return net


if __name__ == "__main__":
    _self_test()
