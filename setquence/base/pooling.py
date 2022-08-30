import warnings

import torch
from reformer_pytorch import Reformer
from torch import __version__ as TORCH_VERSION
from torch import jit as jit
from torch import nn as nn

from setquence.base.attention import MAB, AttentionNoCache, NoCacheSelfAttention


class BasePooler(nn.Module):
    def __init__(self):
        super().__init__()


class BaseJITPooler(jit.ScriptModule):
    def __init__(self):
        super().__init__()


if TORCH_VERSION >= "1.9.0":

    class PoolingPMA(BaseJITPooler):
        def __init__(self, config, req_grad=True, norm_first=False):
            super().__init__()
            self.kseeds = config.k_seeds
            self.hidden_size = config.hidden_size
            self.n_heads = config.n_heads
            self.p_dropout = config.p_dropout

            self.seed = nn.Parameter(torch.Tensor(1, self.kseeds, self.hidden_size), requires_grad=req_grad,)
            nn.init.xavier_uniform_(self.seed)
            self.MutomePMA = MAB(
                self.hidden_size, self.n_heads, batch_first=True, dropout=self.p_dropout, norm_first=norm_first
            )

        @jit.script_method
        def forward(self, input):
            batch_size = input.shape[0]
            seed = self.seed.repeat(batch_size, 1, 1)
            out = self.MutomePMA(seed, input)
            return out


else:

    class PoolingPMA(BasePooler):
        def __init__(self, config, req_grad=True):
            super().__init__()
            self.kseeds = config.k_seeds
            self.hidden_size = config.hidden_size
            self.n_heads = config.n_heads
            self.p_dropout = config.p_dropout

            self.seed = nn.Parameter(torch.Tensor(1, self.kseeds, self.hidden_size), requires_grad=req_grad,)
            nn.init.xavier_uniform_(self.seed)
            self.MutomePMA = MAB(self.hidden_size, self.n_heads, batch_first=True, dropout=self.p_dropout,)

        def forward(self, input):
            batch_size = input.shape[0]
            seed = self.seed.repeat(batch_size, 1, 1)
            out = self.MutomePMA(input, seed)
            return out


class PoolingSeedSimilarity(BasePooler):
    def __init__(self, config, req_grad=True):
        super().__init__()
        self.kseeds = config.k_seeds
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.p_dropout = config.p_dropout

        self.activation_mm = nn.Softmax(dim=-2)
        self.LayerNormSequences = nn.LayerNorm(self.hidden_size)
        self.LayerNormSeeds = nn.LayerNorm(self.hidden_size)
        self.LayerNormActivation = nn.LayerNorm(self.kseeds)

        self.seed = nn.Parameter(torch.Tensor(1, self.kseeds, self.hidden_size), requires_grad=req_grad,)
        nn.init.xavier_uniform_(self.seed)

    def forward(self, input_seqs):
        batch_size = input_seqs.shape[0]
        seed = self.seed.repeat(batch_size, 1, 1)

        _seed = self.LayerNormSeeds(seed)
        _input_sequences = self.LayerNormSequences(input_seqs)

        attn = _seed.matmul(_input_sequences.transpose(-2, -1))
        attn /= attn.shape[-1]
        attn = self.activation_mm(attn)
        attn = attn.sum(-1)
        out_attn = self.LayerNormActivation(attn)

        return out_attn


class PoolingPMATopK(BasePooler):
    def __init__(self, config, req_grad=True):
        super().__init__()
        self.kseeds = config.k_seeds
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.p_dropout = config.p_dropout
        try:
            self.topk = config.topk
        except AttributeError:
            warnings.warn("Could not find 'topk' setting; defaults to 128")
            self.topk = 128

        self.seed = nn.Parameter(torch.Tensor(1, self.kseeds, self.hidden_size), requires_grad=req_grad,)
        nn.init.xavier_uniform_(self.seed)

        d = self.n_heads
        f, drop = nn.Softmax(-1), nn.Dropout(0.1)
        activation = lambda x: drop(f(x * d ** -0.5))
        self.MutomePMATopK = AttentionNoCache(activation)

    def forward(self, input):
        # assumes that tensor format is (Batch, SeqLen, SeqEmbDim)
        batch_size = input.shape[0]
        seed = self.seed.repeat(batch_size, 1, 1)
        n_sequences = input.shape[1]

        # if there are less sequences than top-k
        args = {"topk": min(self.topk, n_sequences), "Q_chunk_size": 2048}
        out = self.MutomePMATopK(seed, input, input, causal_masking=False, args=args)
        return out


class PoolingPMATopKSelf(BasePooler):
    def __init__(self, config, req_grad=True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        assert self.hidden_size % self.n_heads == 0
        self.head_size = int(self.hidden_size / self.n_heads)
        self.p_dropout = config.p_dropout
        self.n_layers = config.n_layers
        try:
            self.topk = config.topk
        except AttributeError:
            warnings.warn("Could not find 'topk' setting; defaults to 128")
            self.topk = 128

        self.layers = nn.ModuleList(
            [
                AttentionNoCacheTransformerBlock(self.n_heads, self.head_size, self.hidden_size)
                for _ in range(self.n_layers)
            ]
        )

    def forward(self, input, args=None):
        # assumes that tensor format is (Batch, SeqLen, SeqEmbDim)
        n_sequences = input.shape[1]

        if args is None:
            args = {"topk": min(self.topk, n_sequences), "Q_chunk_size": 2048}

        Q = input

        # if there are less sequences than top-k
        for layer in self.layers:
            Q = layer(Q, mask=None, causal=False, attn_args=args)

        return Q


class PoolingDecodePMA(BasePooler):
    def __init__(self, max_sequences=500, embedding_size=768):
        super().__init__()
        self.max_sequences = max_sequences
        self.embedding_size = embedding_size
        self.LinearDecode = nn.Linear(self.embedding_size, self.max_sequences * self.embedding_size)
        self.ActivationDecode = nn.ReLU()

    def forward(self, input):
        out = self.ActivationDecode(self.LinearDecode(input))
        out = out.view(self.max_sequences, -1)
        return out


class FFBlock(nn.Module):
    def __init__(self, L_K, d):
        super().__init__()
        self.wi = nn.Linear(d, L_K, bias=False)
        self.wo = nn.Linear(L_K, d, bias=False)
        f, drop = nn.ReLU(), nn.Dropout(0.1)
        self.activation = lambda x: drop(f(x))
        self.Attention = AttentionNoCache(self.activation)

    def forward(self, Q, attn_args=None):
        in_shape = Q.shape
        Q = Q.view(-1, in_shape[-1])
        K = self.wi.weight
        V = self.wo.weight.t()

        out = self.Attention(Q, K, V, args=attn_args)

        return out.view(in_shape)


class AttentionNoCacheTransformerBlock(nn.Module):
    def __init__(self, n_heads, d_head, d_ff):
        super().__init__()
        d_model = n_heads * d_head
        self.attn = NoCacheSelfAttention(n_heads, d_head)
        self.norm_a = nn.LayerNorm(d_model)
        self.ff = FFBlock(d_ff, d_model)
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, Q, mask=None, causal=False, attn_args=None):
        in_shape = Q.shape
        Q = self.norm_a(Q + self.attn(Q, Q, Q, mask=mask, causal=causal, extra_args=attn_args))
        assert in_shape == Q.shape
        Q = self.norm_f(Q + self.ff(Q, attn_args=attn_args))
        assert in_shape == Q.shape
        return Q


class PoolingTransformer(BasePooler):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.p_dropout = config.p_dropout
        self.n_layers = config.n_layers
        self.batch_first = True

        self.EncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=self.n_heads, dropout=self.p_dropout
        )
        self.TranformerEncoder = nn.TransformerEncoder(self.EncoderLayer, num_layers=self.n_layers)

    def forward(self, input):
        # Supports batch_first in torch <= 1.8.0
        is_batched = input.dim() == 3
        if self.batch_first and is_batched:
            input = input.transpose(1, 0)

        # Runs the transformer encoder
        out = self.TranformerEncoder(input)

        # Supports batch_first in torch <= 1.8.0 (for returning out)
        is_batched = out.dim() == 3
        if self.batch_first and is_batched:
            out = out.transpose(1, 0)

        out = torch.mean(out, dim=1).reshape(1, -1, self.hidden_size)

        return out


class PoolingReformer(BasePooler):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.p_dropout = config.p_dropout
        self.n_layers = config.n_layers

        self.ReformerEncoder = Reformer(
            dim=self.hidden_size, depth=self.n_layers, heads=self.n_heads, lsh_dropout=self.p_dropout, causal=True
        )

    def forward(self, input):
        # This assumes that data has shape (batch, tokens, hidden_size)
        out = self.ReformerEncoder(input)
        out = torch.mean(out, dim=1).reshape(1, -1, self.hidden_size)
        return out
