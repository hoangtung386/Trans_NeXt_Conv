"""Low-rank gating mechanism for CNN-Transformer interaction."""

from typing import Tuple

from torch import nn, Tensor


class GatingMechanism(nn.Module):
    """Low-rank gating mechanism for CNN-Transformer interaction.

    Generates a gate signal from transformer features to modulate
    CNN decoder features.
    """

    def __init__(self, embed_dim: int, gate_dim: int = 64):
        super().__init__()
        self.gate_dim = gate_dim

        self.gate_proj = nn.Sequential(
            nn.Linear(embed_dim, gate_dim, bias=False),
            nn.Sigmoid(),
        )
        self.feature_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self, trans_features: Tensor, cnn_features: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply gating mechanism.

        Args:
            trans_features: Transformer features [B, N, D].
            cnn_features: CNN features [B, C, H, W].

        Returns:
            Tuple of (gate, query, modulated_cnn).
        """
        gate = self.gate_proj(trans_features.mean(dim=1))
        query = self.feature_proj(trans_features)

        b, c, h, w = cnn_features.shape
        gate_expanded = gate.view(b, 1, 1, 1).expand(-1, c, h, w)
        modulated_cnn = cnn_features * (1 + gate_expanded)

        return gate, query, modulated_cnn
