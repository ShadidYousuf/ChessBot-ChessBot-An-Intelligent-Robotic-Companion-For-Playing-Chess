import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, filter_sizes):
        """
        CNN Model.

        Parameters:
            filter_sizes (list): List of filter sizes for convolutional layers
                                 and number of units for the dense layer.
                                 Example: [32, 64, 128] where:
                                  - 32 filters in the first conv layer,
                                  - 64 filters in the second conv layer,
                                  - 128 units in the dense layer.
        """
        super(CNNModel, self).__init__()

        # Convolutional Layer #1
        self.conv1 = nn.Conv2d(
            in_channels=1,  # Grayscale images have 1 channel
            out_channels=filter_sizes[0],
            kernel_size=5,
            padding=2,
        )

        # Convolutional Layer #2
        self.conv2 = nn.Conv2d(
            in_channels=filter_sizes[0],
            out_channels=filter_sizes[1],
            kernel_size=5,
            padding=2,
        )

        # Fully connected (dense) layer
        self.fc1 = nn.Linear(
            in_features=5 * 5 * filter_sizes[1], out_features=filter_sizes[2]
        )

        # Dropout layer
        self.dropout = nn.Dropout(p=0.4)

        # Output layer
        self.fc2 = nn.Linear(
            in_features=filter_sizes[2],
            out_features=2,  # Output logits for binary classification
        )

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 21, 21).

        Returns:
            torch.Tensor: Logits for the two classes.
        """
        # Convert input to float
        x = x.float()

        # Convolutional Layer #1 + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))

        # Convolutional Layer #2 + ReLU + Pooling
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output for the dense layer
        x = torch.flatten(
            x, start_dim=1
        )  # Shape: (batch_size, 5 * 5 * filter_sizes[1])

        # Dense layer with ReLU
        x = F.relu(self.fc1(x))

        # Apply dropout
        x = self.dropout(x)

        # Logits layer
        x = self.fc2(x)

        return x


def get_patch_model():
    patch_model = nn.Sequential(
        nn.Conv2d(
            in_channels=3,  # Grayscale images have 1 channel
            out_channels=32,
            kernel_size=5,
            padding=2,
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(
            in_channels=32,  # Grayscale images have 1 channel
            out_channels=64,
            kernel_size=5,
            padding=2,
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(
            in_channels=64,  # Grayscale images have 1 channel
            out_channels=128,
            kernel_size=5,
            padding=2,
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(in_features=4 * 4 * 128, out_features=1024),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(in_features=1024, out_features=3),
    )
    return patch_model
