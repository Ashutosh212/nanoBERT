import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Multi-Head Self-Attention
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads                          # e.g. 2 or 4
        self.d_k = num_hiddens // num_heads                 # dimension per head

        # four learnable linear projections (no bias, as in the original paper)
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=False)  # project input → queries
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=False)  # project input → keys
        self.W_v = nn.Linear(num_hiddens, num_hiddens, bias=False)  # project input → values
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=False)  # merge all heads → output

        self.dropout = nn.Dropout(dropout)                  # regularise attention weights

    # ── helper: split the last dimension into (num_heads, d_k) then flatten batch+heads
    def _split_heads(self, X):
        B, S, _ = X.shape                                   # batch, seq_len, num_hiddens
        X = X.view(B, S, self.num_heads, self.d_k)          # split hidden dim into heads
        X = X.transpose(1, 2)                               # (B, num_heads, S, d_k)
        return X.reshape(B * self.num_heads, S, self.d_k)   # merge batch & heads for bmm

    def forward(self, X, valid_lens=None):
        # X is both query and key/value (self-attention); shape: (B, S, num_hiddens)
        Q = self._split_heads(self.W_q(X))                  # (B*H, S, d_k)
        K = self._split_heads(self.W_k(X))                  # (B*H, S, d_k)
        V = self._split_heads(self.W_v(X))                  # (B*H, S, d_k)

        # ── scaled dot-product attention ──────────────────────────────────────
        # dot every query against every key, then scale to keep gradients stable
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_k)  # (B*H, S, S)

        # mask out padding positions so they cannot be attended to
        if valid_lens is not None:
            # repeat each valid_len once per head so shapes align after _split_heads
            vl = valid_lens.repeat_interleave(self.num_heads)   # (B*H,)
            max_len = scores.shape[-1]
            # True wherever the key position is beyond the valid length (i.e. padding)
            mask = torch.arange(max_len, device=X.device)[None, :] >= vl[:, None]  # (B*H, S)
            mask = mask.unsqueeze(1)                             # (B*H, 1, S) — broadcast over queries
            scores = scores.masked_fill(mask, -1e9)              # push padding to ~0 after softmax

        attn = self.dropout(F.softmax(scores, dim=-1))       # (B*H, S, S)  — normalise over keys
        out  = torch.bmm(attn, V)                            # (B*H, S, d_k) — weighted value sum

        # ── merge heads back ──────────────────────────────────────────────────
        B_H, S, _ = out.shape
        B = B_H // self.num_heads
        out = out.reshape(B, self.num_heads, S, self.d_k)    # separate batch and heads again
        out = out.transpose(1, 2).reshape(B, S, -1)          # (B, S, num_hiddens)
        return self.W_o(out)                                  # final linear mix of all heads


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Position-wise Feed-Forward Network
# ══════════════════════════════════════════════════════════════════════════════

class FFN(nn.Module):
    def __init__(self, num_hiddens, ffn_num_hiddens, dropout):
        super().__init__()
        self.fc1     = nn.Linear(num_hiddens, ffn_num_hiddens)  # expand to wider space
        self.fc2     = nn.Linear(ffn_num_hiddens, num_hiddens)  # project back to model dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        X = self.fc1(X)                   # (B, S, ffn_num_hiddens)
        X = F.gelu(X)                     # GELU activation — used in BERT (smoother than ReLU)
        X = self.dropout(X)               # drop some activations for regularisation
        return self.fc2(X)                # (B, S, num_hiddens)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Transformer Encoder Block  (Attention → Add&Norm → FFN → Add&Norm)
# ══════════════════════════════════════════════════════════════════════════════

class TransformerEncoderBlock(nn.Module):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout):
        super().__init__()
        self.attn   = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.ffn    = FFN(num_hiddens, ffn_num_hiddens, dropout)
        self.norm1  = nn.LayerNorm(num_hiddens)   # normalise after attention sub-layer
        self.norm2  = nn.LayerNorm(num_hiddens)   # normalise after FFN sub-layer
        self.drop1  = nn.Dropout(dropout)          # dropout on attention output
        self.drop2  = nn.Dropout(dropout)          # dropout on FFN output

    def forward(self, X, valid_lens=None):
        # --- sub-layer 1: multi-head self-attention with residual connection ---
        attn_out = self.attn(X, valid_lens)        # attend to all (non-padding) positions
        X = self.norm1(X + self.drop1(attn_out))   # residual + layer-norm keeps training stable

        # --- sub-layer 2: position-wise FFN with residual connection -----------
        ffn_out = self.ffn(X)                      # independent transform at each position
        X = self.norm2(X + self.drop2(ffn_out))    # residual + layer-norm again
        return X                                   # (B, S, num_hiddens)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  BERT Encoder  (embeddings + stack of Transformer blocks)
# ══════════════════════════════════════════════════════════════════════════════

class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, max_len=1000):
        super().__init__()
        # map each token id to a dense vector
        self.token_embedding   = nn.Embedding(vocab_size, num_hiddens)
        # map segment id (0 = sentence A, 1 = sentence B) to a dense vector
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        # learnable positional embeddings (not sinusoidal — BERT learns these)
        # shape (1, max_len, num_hiddens): the 1 lets it broadcast over any batch size
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))
        self.dropout = nn.Dropout(dropout)          # applied once after summing embeddings

        # stack num_blks identical Transformer encoder blocks
        self.blks = nn.ModuleList([
            TransformerEncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout)
            for _ in range(num_blks)
        ])

    def forward(self, tokens, segments, valid_lens=None):
        # tokens:   (B, S)  — token ids
        # segments: (B, S)  — 0 for sentence A tokens, 1 for sentence B tokens
        X  = self.token_embedding(tokens)                   # (B, S, num_hiddens)
        X  = X + self.segment_embedding(segments)           # add sentence-side information
        X  = X + self.pos_embedding[:, :X.shape[1], :]     # add position information (clipped to S)
        X  = self.dropout(X)                                # regularise the summed embeddings
        for blk in self.blks:                               # pass through each Transformer block
            X = blk(X, valid_lens)
        return X                                            # (B, S, num_hiddens)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Masked Language Model head  (predict the original token at masked positions)
# ══════════════════════════════════════════════════════════════════════════════

class MaskLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens):
        super().__init__()
        # small MLP: linear → GELU → LayerNorm → linear → vocab logits
        self.mlp = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens),   # transform the hidden representation
            nn.GELU(),                              # non-linearity
            nn.LayerNorm(num_hiddens),             # stabilise activations
            nn.Linear(num_hiddens, vocab_size),    # project to a score for every vocab token
        )

    def forward(self, X, pred_positions):
        # X:              (B, S, num_hiddens) — full encoder output
        # pred_positions: (B, num_preds)      — which positions were masked
        B, num_preds = pred_positions.shape

        # build row indices [0,0,...,1,1,...,B-1,...] to index into the batch dimension
        batch_idx = torch.arange(B, device=X.device).repeat_interleave(num_preds)  # (B*num_preds,)
        flat_pos  = pred_positions.reshape(-1)                                      # (B*num_preds,)

        # gather the encoder hidden states only at the masked positions
        masked_X = X[batch_idx, flat_pos]                    # (B*num_preds, num_hiddens)
        masked_X = masked_X.reshape(B, num_preds, -1)        # (B, num_preds, num_hiddens)

        return self.mlp(masked_X)                            # (B, num_preds, vocab_size)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Next Sentence Prediction head  (binary: is sentence B the real next one?)
# ══════════════════════════════════════════════════════════════════════════════

class NextSentencePred(nn.Module):
    def __init__(self, num_hiddens):
        super().__init__()
        # two output logits: index 0 = not-next, index 1 = is-next
        self.output = nn.Linear(num_hiddens, 2)

    def forward(self, cls_repr):
        # cls_repr: (B, num_hiddens) — the [CLS] token hidden state (sentence-level summary)
        return self.output(cls_repr)                         # (B, 2)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Full BERT Model
# ══════════════════════════════════════════════════════════════════════════════

class BERTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, max_len=1000):
        super().__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_blks, dropout, max_len)

        # small MLP that transforms the [CLS] vector before NSP classification
        # (matches the "pooler" in the original BERT code)
        # TODO: try removing this and feeding the [CLS] vector directly to the NSP head
        self.hidden = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens),   # dense layer on the [CLS] representation
            nn.Tanh(),                              # tanh keeps values in (-1, 1)
        )
        self.mlm = MaskLM(vocab_size, num_hiddens)   # MLM prediction head
        self.nsp = NextSentencePred(num_hiddens)      # NSP prediction head

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        # ── encode the full sequence ──────────────────────────────────────────
        encoded_X = self.encoder(tokens, segments, valid_lens)   # (B, S, num_hiddens)

        # ── MLM head (only needed during pre-training, skip during fine-tuning) ─
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)     # (B, num_preds, vocab_size)
        else:
            mlm_Y_hat = None

        # ── NSP head — use only the [CLS] token (position 0) ─────────────────
        # [CLS] aggregates the whole sequence; pass through the pooler MLP first
        cls_repr  = self.hidden(encoded_X[:, 0, :])             # (B, num_hiddens)
        nsp_Y_hat = self.nsp(cls_repr)                          # (B, 2)

        return encoded_X, mlm_Y_hat, nsp_Y_hat


# ══════════════════════════════════════════════════════════════════════════════
# quick sanity-check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    B, S       = 4, 64        # batch size, sequence length
    vocab_size = 10_000

    model = BERTModel(
        vocab_size    = vocab_size,
        num_hiddens   = 128,
        ffn_num_hiddens = 256,
        num_heads     = 2,
        num_blks      = 2,
        dropout       = 0.1,
        max_len       = 64,
    )

    tokens        = torch.randint(0, vocab_size, (B, S))   # fake token ids
    segments      = torch.zeros(B, S, dtype=torch.long)    # all sentence-A for now
    valid_lens    = torch.full((B,), S, dtype=torch.long)  # all positions valid
    pred_positions = torch.randint(0, S, (B, 10))           # 10 masked positions per example

    enc, mlm_out, nsp_out = model(tokens, segments, valid_lens, pred_positions)

    print(f'encoder output : {tuple(enc.shape)}')          # (4, 64, 128)
    print(f'MLM logits     : {tuple(mlm_out.shape)}')      # (4, 10, 10000)
    print(f'NSP logits     : {tuple(nsp_out.shape)}')      # (4, 2)
    print('model param count:', sum(p.numel() for p in model.parameters()))
