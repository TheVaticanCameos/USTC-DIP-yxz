import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential( # input (256, 512), output (128, 256)
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.8)
        )
        ### FILL: add more CONV Layers

        self.conv2 = nn.Sequential( # input (128, 256), output (32, 64)
            nn.Conv2d(8, 32, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.8)
        )
        self.conv3 = nn.Sequential( # input (32, 64), output (8, 16)
            nn.Conv2d(32, 128, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.8)
        )

        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function
        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.8)
        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(32, 8, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.8)
        )
        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Decoder forward pass
        x = self.convT1(x)
        x = self.convT2(x)
        output = self.convT3(x)
        ### FILL: encoder-decoder forward pass
        
        return output
    