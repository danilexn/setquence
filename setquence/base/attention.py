from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch import __version__ as TORCH_VERSION
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import get_device_states, set_device_states


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0, batch_first=True):
        super().__init__()
        self.MHA = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.batch_first = batch_first

    def forward(self, query, key, value, attn_mask, key_padding_mask, need_weights):
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        out = self.MHA(query, key, value)

        if self.batch_first and is_batched:
            return out[0].transpose(1, 0)
        else:
            return out[0]


class MAB(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
    ):
        super().__init__()

        if TORCH_VERSION >= "1.9.0":
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,)
        else:
            self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._mha_block(self.norm1(x), memory, memory_mask, tgt_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._mha_block(x, memory, memory_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # multihead attention block
    def _mha_block(
        self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.multihead_attn(
            x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.dropout3(x)


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


class Deterministic(nn.Module):
    def __init__(self, net):
        """ensures the second forward pass inside backward pass is
        identical to original forward pass under stochasticity."""

        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)


class _AttentionNoCache(torch.autograd.function.Function):
    @staticmethod
    def forward(ctx, Q, K, V, activation, mask=None, args=None):
        """Computes (args['activation'](Q.K^T)).V
        -----------
        activation: nn.Softmax(-1), nn.ReLU(), etc. (default nn.ReLU())
        NOTE: for vanilla softmax attention remember to supply activation with scaling by d**-0.25 + softmax.
        mask: bool (default None)
        args is a dict() with following optional keys:
            - topk: (default -1)
        """
        assert isinstance(activation, Deterministic), "activation must be wrapped in Deterministic"
        assert mask is None or (mask.dtype == torch.bool and not mask.requires_grad)

        topk = -1
        if args is not None and "topk" in args and args["topk"] > 0:
            topk = args["topk"]
        args.update({"topk": topk})

        dots = Q.matmul(K.transpose(-1, -2))  # [Lq, Lk]
        if mask is not None:
            dots.masked_fill_(mask, max_neg_value(dots))

        top_dots, top_inds = None, None
        if topk > 0:
            mask = None
            top_dots, top_inds = dots.topk(topk, dim=-1, sorted=False)
            attn = dots.zero_().scatter_(-1, top_inds, activation(top_dots, record_rng=activation.training))
            # we're not caching dots so ok to overwrite
        else:
            attn = activation(dots, record_rng=activation.training)  # [Lq, Lk]
        del dots

        out = attn.matmul(V)
        ctx.activation = activation
        ctx.args = args
        ctx.save_for_backward(Q, K, V, mask, top_dots, top_inds)
        return out  # [Lq, d]

    @staticmethod
    def backward(ctx, d_out):
        Q, K, V, mask, top_dots, top_inds = ctx.saved_tensors
        args = ctx.args
        activation, topk = ctx.activation, args["topk"]
        matmul_x_t_y = _AttentionNoCache.matmul_x_t_y

        d_attn = d_out.matmul(V.transpose(-2, -1))  # [Lq, Lk] == [Lq, d] x [d, Lk]

        if topk > 0:
            d_top_attn = d_attn.gather(-1, top_inds)
            # recompute d_top_dots later used for d_dots
            with torch.enable_grad():
                top_dots.requires_grad = True
                top_attn = activation(top_dots, set_rng=True)  # [Lq, topk]
            top_attn.backward(d_top_attn)
            d_top_dots = top_dots.grad
            del top_dots, d_top_attn
            top_attn = top_attn.detach()

            # compute attn
            attn = d_attn.zero_().scatter_(-1, top_inds, top_attn)  # [Lq, Lk]
            d_V = matmul_x_t_y(attn, d_out)  # [Lk, d]  == [Lk, Lq] x [Lq, d]
            del top_attn, d_out
            # compute d_dots
            d_dots = d_attn.scatter_(-1, top_inds, d_top_dots)
            del top_inds, d_top_dots
        else:
            # recompute attn and d_dots
            dots = Q.matmul(K.transpose(-1, -2))  # [Lq, Lk]
            if mask is not None:
                dots.masked_fill_(mask, max_neg_value(dots))
            with torch.enable_grad():
                dots.requires_grad = True
                attn = activation(dots, set_rng=True)  # [Lq, Lk]
            attn.backward(d_attn)
            d_dots = dots.grad
            del dots, d_attn
            d_V = matmul_x_t_y(attn, d_out)  # [Lk, d] == [Lk, Lq] x [Lq, d]
            del attn, d_out

        d_Q = d_dots.matmul(K)  # [Lq, d] == [Lq, Lk] x [Lk, d]
        d_K = matmul_x_t_y(d_dots, Q)  # [Lk, d] == [Lk, Lq] x [Lq, d]
        return d_Q, d_K, d_V, None, None, None

    @staticmethod
    def matmul_x_t_y(x, y):
        """compute x^T.y"""
        a, b, c = x.shape[-1], x.shape[-2], y.shape[-1]
        if b * a <= b * c + c * a:
            return x.transpose(-2, -1).matmul(y)  # [a, c] = [a, b] x [b, c]
        return y.transpose(-2, -1).matmul(x).transpose(-2, -1)  # [a, c] = ([c, b] x [b, a])^T


class AttentionNoCache(nn.Module):
    def __init__(self, activation):
        """activation: nn.Softmax(-1), nn.ReLU(), etc.
        NOTE: for vanilla softmax attention remember to supply activation with scaling by d**-0.5 + softmax.
        If topk will be used later dont use dropout
        """
        super().__init__()
        self.activation = Deterministic(activation)

    def forward(self, Q, K, V, mask=None, causal_masking=False, args=None):
        """Computes self.activation(Q.K^T).V
        -----------
        Q, K, V: [...,Lq,d], [...,Lk,d], [...,Lk,d]
        mask: bool - must be broadcastable. True's are masked (default None)
        causal_masking: will apply causal masking (mask should be None)
        args is a dict() with following optional keys:
            - Q_chunk_size: queries are chunked and looped over to limit max mem usage (default Lq)
            - topk: (default -1)
        """
        assert not causal_masking or mask is None, "mask should not be provided with causal masking"
        Q_chunks, Lq = 1, Q.shape[-2]
        if args is not None and "Q_chunk_size" in args:
            Q_chunk_size = args["Q_chunk_size"] if args["Q_chunk_size"] > 0 else Lq
            Q_chunks = max(1, Lq // Q_chunk_size)

        out = Q.new_zeros(Q.shape[:-1] + (V.shape[-1],))
        for chunk_ids in torch.arange(Lq, device=Q.device).chunk(Q_chunks):
            chunk_mask = None
            if mask is not None:
                # we cant realize a large Lq x Lk mask so we must realize it after chunking
                assert mask.shape[-2] in [1, Lq]
                chunk_mask = mask if mask.shape[-2] == 1 else mask[..., chunk_ids, :]
            elif causal_masking:
                assert Q.shape[-2] == K.shape[-2]
                chunk_mask = torch.triu(
                    torch.ones(len(chunk_ids), K.shape[-2], device=Q.device, dtype=torch.bool),
                    diagonal=1 + chunk_ids[0],
                )  # [Cq, Lk]
            out[..., chunk_ids, :] = _AttentionNoCache.apply(
                Q[..., chunk_ids, :], K, V, self.activation, chunk_mask, args
            )  # [Cq, d]
        return out  # [Lq,d]


class NoCacheSelfAttention(nn.Module):
    def __init__(self, H, d):
        super().__init__()
        self.H, self.d = H, d
        self.wQ, self.wK, self.wV, self.wO = (nn.Linear(H * d, H * d) for _ in range(4))
        f, drop = nn.Softmax(-1), nn.Dropout(0.1)
        self.activation = lambda x: drop(f(x * d ** -0.5))
        self.Attention = AttentionNoCache(self.activation)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.H, self.d)
        return x.view(*new_x_shape).permute(0, 2, 1, 3)

    def forward(self, Q, K, V, mask=None, causal=False, extra_args=None):
        in_shape = Q.shape
        Q = self.transpose_for_scores(self.wQ(Q))
        K = self.transpose_for_scores(self.wK(K))
        V = self.transpose_for_scores(self.wV(V))

        out = self.Attention(Q, K, V, mask=mask, causal_masking=causal, args=extra_args)

        return self.wO(out.permute(0, 2, 1, 3).reshape(in_shape))
