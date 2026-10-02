import torch
import torch.nn as nn


class ConditionalGenerator(nn.Module):
    """
    Conditional generator for 64x64 grayscale images.

    The generator receives random noise and a class label
    (0 = square, 1 = circle).
    """

    def __init__(self, latent_dim=100, num_classes=2, image_channels=1):
        super().__init__()

        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        self.network = nn.Sequential(
            nn.Linear(latent_dim * 2, 256 * 8 * 8),
            nn.BatchNorm1d(256 * 8 * 8),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 8, 8)),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, image_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        label_features = self.label_embedding(labels)
        conditioned_input = torch.cat([noise, label_features], dim=1)
        return self.network(conditioned_input)


class ConditionalDiscriminator(nn.Module):
    """
    Conditional discriminator for 64x64 grayscale images.

    The discriminator receives an image and a class label
    (0 = square, 1 = circle).
    """

    def __init__(self, num_classes=2, image_channels=1):
        super().__init__()

        self.label_embedding = nn.Embedding(num_classes, 64 * 64)

        self.network = nn.Sequential(
            nn.Conv2d(image_channels + 1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, images, labels):
        label_map = self.label_embedding(labels)
        label_map = label_map.view(-1, 1, 64, 64)
        conditioned_images = torch.cat([images, label_map], dim=1)
        return self.network(conditioned_images)
