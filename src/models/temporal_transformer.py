import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalTransformer(nn.Module):
    """Learnable positional encoding + lightweight TransformerEncoder for frame sequences.

    Input:  [B, T, D]  pre-L2-normed CLIP per-frame embeddings (D=512)
    Output: [B, D]     L2-normalized temporally-aware scene embedding

    Supports variable T via src_key_padding_mask (True = padded position).
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_frames: int = 16,
    ):
        super().__init__()
        # Learnable positional encoding — broadcasts over batch dimension
        self.pos_encoding = nn.Parameter(torch.randn(1, max_frames, d_model) * 0.02)

        # Projects per-frame (cx, cy, scale) into d_model space so it can be
        # added as a bias, similar to positional encoding.  Small init so it
        # doesn't overwhelm the CLIP embedding at the start of training.
        self.pos_proj = nn.Linear(3, d_model)
        nn.init.normal_(self.pos_proj.weight, std=0.02)
        nn.init.zeros_(self.pos_proj.bias)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                if p is not self.pos_proj.weight:
                    nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.BoolTensor | None = None,
        positions: torch.Tensor | None = None,
        return_per_frame: bool = False,
    ) -> torch.Tensor:
        """x: [B, T, D], key_padding_mask: [B, T] (True = padded).

        When *positions* [B, T, 3] is given it is projected to d_model and
        added to the input, giving the transformer explicit per-frame object
        location information.

        When *return_per_frame* is True, returns (video_emb, per_frame_outputs)
        where per_frame_outputs is [B, T, D] (before pooling).  This is used
        by the position-prediction reward head during training.
        """
        B, T, D = x.shape
        if T == 0:
            raise ValueError("Empty frame sequence (T=0).")

        # Add learnable positional encoding — slice to actual sequence length
        x = x + self.pos_encoding[:, :T, :]  # [B, T, D]

        # Optionally add projected per-frame positions as an extra bias
        if positions is not None:
            x = x + self.pos_proj(positions)

        # TransformerEncoder with optional padding mask
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, T, D]

        per_frame = x  # save before pooling

        # Masked mean pool over valid positions
        if key_padding_mask is not None:
            mask = (~key_padding_mask).unsqueeze(-1).float()  # [B, T, 1], 1=valid
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)  # [B, D]

        video_emb = F.normalize(x, dim=-1)

        if return_per_frame:
            return video_emb, per_frame
        return video_emb
