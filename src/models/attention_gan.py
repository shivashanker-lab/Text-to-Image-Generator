import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """
    Cross-attention layer for conditioning visual features on text features.

    Visual features provide the queries, while text embeddings provide
    the keys and values.
    """

    def __init__(self, visual_dim=64, text_dim=768, attention_dim=64):
        super().__init__()

        self.query = nn.Linear(visual_dim, attention_dim)
        self.key = nn.Linear(text_dim, attention_dim)
        self.value = nn.Linear(text_dim, attention_dim)

        self.scale = attention_dim ** -0.5
        self.output = nn.Linear(attention_dim, visual_dim)

    def forward(self, visual_features, text_embeddings):
        """
        Parameters
        ----------
        visual_features : torch.Tensor
            Visual features with shape [batch, visual_tokens, visual_dim].

        text_embeddings : torch.Tensor
            Text embeddings with shape [batch, text_tokens, text_dim].

        Returns
        -------
        tuple
            (attended_visual_features, attention_weights)
        """
        queries = self.query(visual_features)
        keys = self.key(text_embeddings)
        values = self.value(text_embeddings)

        attention_scores = torch.matmul(
            queries,
            keys.transpose(-2, -1),
        ) * self.scale

        attention_weights = torch.softmax(attention_scores, dim=-1)

        attended_features = torch.matmul(
            attention_weights,
            values,
        )

        attended_features = self.output(attended_features)

        return attended_features, attention_weights


class AttentionConditioning(nn.Module):
    """
    Residual cross-attention conditioning block.

    Adds text-conditioned information to visual features.
    """

    def __init__(self, visual_dim=64, text_dim=768, attention_dim=64):
        super().__init__()

        self.cross_attention = CrossAttention(
            visual_dim=visual_dim,
            text_dim=text_dim,
            attention_dim=attention_dim,
        )

        self.norm = nn.LayerNorm(visual_dim)

    def forward(self, visual_features, text_embeddings):
        attended_features, attention_weights = self.cross_attention(
            visual_features,
            text_embeddings,
        )

        conditioned_features = self.norm(
            visual_features + attended_features
        )

        return conditioned_features, attention_weights
