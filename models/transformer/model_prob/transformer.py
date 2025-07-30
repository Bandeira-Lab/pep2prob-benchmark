# model/transformer.py

import torch
from torch import nn

class DecoderOnlyTransformer(nn.Module):
    """
    A decoder-only Transformer model for predicting fragment ion probabilities.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        max_len: int = 300,
        pad_id: int = 0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.2
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_head = nn.Linear(d_model, 1)
        self.max_len = max_len
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] tensor of token IDs

        Returns:
            preds: [B, L, 1] predicted probability logits
        """
        B, L = input_ids.size()
        x = self.embedding(input_ids) + self.pos_embedding[:, :L, :]  # [B, L, D]
        x = x.permute(1, 0, 2)  # [L, B, D]

        tgt_mask = self._generate_square_subsequent_mask(L).to(input_ids.device)
        out = self.decoder(tgt=x, memory=x, tgt_mask=tgt_mask)  # [L, B, D]
        out = out.permute(1, 0, 2)  # [B, L, D]

        preds = self.output_head(out)  # [B, L, 1]
        return preds

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate a square causal mask for self-attention.
        Masked positions are filled with -inf.
        """
        mask = torch.full((sz, sz), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        return mask