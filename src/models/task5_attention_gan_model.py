
import torch
import torch.nn as nn

IMAGE_SIZE = 32
LATENT_DIM = 100
TEXT_DIM = 768
VISUAL_DIM = 128
ATTENTION_DIM = 128


def masked_mean(text_features, text_mask):
    weights = text_mask.unsqueeze(-1).to(text_features.dtype)
    return (text_features * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class CrossAttention(nn.Module):
    # Learned visual queries attend to BERT token keys and values.
    def __init__(self, visual_dim=VISUAL_DIM, text_dim=TEXT_DIM, attention_dim=ATTENTION_DIM):
        super().__init__()
        self.query = nn.Linear(visual_dim, attention_dim)
        self.key = nn.Linear(text_dim, attention_dim)
        self.value = nn.Linear(text_dim, attention_dim)
        self.output = nn.Linear(attention_dim, visual_dim)
        self.scale = attention_dim ** -0.5

    def forward(self, visual_features, text_features, text_mask):
        q = self.query(visual_features)
        k = self.key(text_features)
        v = self.value(text_features)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        valid_tokens = text_mask.bool().unsqueeze(1)
        scores = scores.masked_fill(~valid_tokens, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v)
        return self.output(attended), weights


class ImageHead(nn.Module):
    # Converts latent and conditioning vectors into a 32x32 grayscale image.
    def __init__(self, latent_dim=LATENT_DIM, condition_dim=VISUAL_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noise, conditioning):
        x = self.fc(torch.cat([noise, conditioning], dim=1))
        x = x.view(x.shape[0], 256, 4, 4)
        return self.deconv(x)


class BaselineGenerator(nn.Module):
    # Text-conditioned GAN baseline without cross-attention.
    def __init__(self, latent_dim=LATENT_DIM, text_dim=TEXT_DIM):
        super().__init__()
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, VISUAL_DIM),
            nn.LayerNorm(VISUAL_DIM),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.image_head = ImageHead(latent_dim, VISUAL_DIM)

    def forward(self, noise, text_features, text_mask):
        pooled = masked_mean(text_features, text_mask)
        conditioning = self.text_projection(pooled)
        return self.image_head(noise, conditioning)


class AttentionGenerator(nn.Module):
    # Text-conditioned GAN generator with token-level cross-attention.
    def __init__(self, latent_dim=LATENT_DIM, text_dim=TEXT_DIM,
                 visual_dim=VISUAL_DIM, attention_dim=ATTENTION_DIM):
        super().__init__()
        self.noise_projection = nn.Linear(latent_dim, 16 * visual_dim)
        self.cross_attention = CrossAttention(visual_dim, text_dim, attention_dim)
        self.combine = nn.Linear(visual_dim * 2, visual_dim)
        self.norm = nn.LayerNorm(visual_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.image_head = ImageHead(latent_dim, visual_dim)

    def forward(self, noise, text_features, text_mask):
        visual = self.noise_projection(noise).view(-1, 16, VISUAL_DIM)
        attended, weights = self.cross_attention(visual, text_features, text_mask)
        conditioned = self.activation(self.combine(torch.cat([visual, attended], dim=-1)))
        conditioned = self.norm(visual + conditioned)
        pooled_visual = conditioned.mean(dim=1)
        images = self.image_head(noise, pooled_visual)
        return images, weights


class TextConditionedDiscriminator(nn.Module):
    # Discriminator scores whether image and text condition match.
    def __init__(self, text_dim=TEXT_DIM, image_size=IMAGE_SIZE):
        super().__init__()
        self.image_size = image_size
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, 16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.image_tower = nn.Sequential(
            nn.Conv2d(17, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0),
        )

    def forward(self, images, text_features, text_mask):
        pooled = masked_mean(text_features, text_mask)
        condition = self.text_projection(pooled).unsqueeze(-1).unsqueeze(-1)
        condition = condition.expand(-1, -1, self.image_size, self.image_size)
        x = torch.cat([images, condition], dim=1)
        return self.image_tower(x).flatten()


def generator_images(generator, noise, text_features, text_mask):
    result = generator(noise, text_features, text_mask)
    return result[0] if isinstance(result, tuple) else result
