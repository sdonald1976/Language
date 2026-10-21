"""Incremental (KV-cached) T5 decoding, written out explicitly so it exports to ONNX without depending on
transformers' cache classes. It reuses the fine-tuned weights; export_onnx.py checks it numerically
against the reference model.

    cross_kv.onnx      encoder_hidden, attention_mask           -> cross_kv [L, 2, b, h, s, d_kv]
    decoder_step.onnx  token [b,1], position [1], encoder mask,
                       cross_kv, past [L, 2, b, h, t, d_kv]      -> logits [b, V], present [L, 2, b, h, t+1, d_kv]
"""
import math

import torch
import torch.nn.functional as F


def rms_norm(x, weight, eps):
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps))


def relative_position_bucket(relative_position, num_buckets, max_distance):
    """Unidirectional T5 bucketing (decoder self-attention)."""
    relative_position = -torch.clamp(relative_position, max=0)
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    large = max_exact + (
        torch.log(relative_position.float().clamp(min=1) / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    large = torch.clamp(large, max=num_buckets - 1)
    return torch.where(is_small, relative_position, large)


class CrossKV(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.blocks = model.decoder.block
        self.h = model.config.num_heads
        self.dk = model.config.d_kv

    def forward(self, encoder_hidden):
        b, s, _ = encoder_hidden.shape
        out = []
        for block in self.blocks:
            att = block.layer[1].EncDecAttention
            k = att.k(encoder_hidden).view(b, s, self.h, self.dk).transpose(1, 2)
            v = att.v(encoder_hidden).view(b, s, self.h, self.dk).transpose(1, 2)
            out.append(torch.stack([k, v]))
        return torch.stack(out)


class DecoderStep(torch.nn.Module):
    def __init__(self, model, scale_output):
        super().__init__()
        self.model = model
        c = model.config
        self.h, self.dk, self.eps = c.num_heads, c.d_kv, c.layer_norm_epsilon
        self.buckets, self.max_dist = c.relative_attention_num_buckets, c.relative_attention_max_distance
        self.scale = c.d_model ** -0.5 if scale_output else 1.0
        self.gated = c.feed_forward_proj.startswith("gated")

    def attend(self, q, k, v, bias):
        scores = torch.matmul(q, k.transpose(-1, -2)) + bias  # T5 does not scale by 1/sqrt(d)
        return torch.matmul(F.softmax(scores.float(), dim=-1).type_as(scores), v)

    def forward(self, token, position, attention_mask, cross_kv, past):
        dec = self.model.decoder
        b = token.shape[0]
        x = dec.embed_tokens(token)  # [b, 1, d]
        t = past.shape[4]  # number of cached positions
        key_pos = torch.arange(t + 1, device=token.device)
        rel = key_pos[None, :] - position.view(1, 1)  # [1, t+1]
        rel_att = dec.block[0].layer[0].SelfAttention.relative_attention_bias
        self_bias = rel_att(relative_position_bucket(rel, self.buckets, self.max_dist)).permute(2, 0, 1).unsqueeze(0)  # [1, h, 1, t+1]
        cross_bias = (1.0 - attention_mask[:, None, None, :].float()) * torch.finfo(torch.float32).min  # [b, 1, 1, s]
        presents = []
        for i, block in enumerate(dec.block):
            sa = block.layer[0]
            hn = rms_norm(x, sa.layer_norm.weight, self.eps)
            q = sa.SelfAttention.q(hn).view(b, 1, self.h, self.dk).transpose(1, 2)
            k = sa.SelfAttention.k(hn).view(b, 1, self.h, self.dk).transpose(1, 2)
            v = sa.SelfAttention.v(hn).view(b, 1, self.h, self.dk).transpose(1, 2)
            k_all = torch.cat([past[i, 0], k], dim=2)
            v_all = torch.cat([past[i, 1], v], dim=2)
            presents.append(torch.stack([k_all, v_all]))
            o = self.attend(q, k_all, v_all, self_bias).transpose(1, 2).reshape(b, 1, self.h * self.dk)
            x = x + sa.SelfAttention.o(o)

            ca = block.layer[1]
            hn = rms_norm(x, ca.layer_norm.weight, self.eps)
            q = ca.EncDecAttention.q(hn).view(b, 1, self.h, self.dk).transpose(1, 2)
            o = self.attend(q, cross_kv[i, 0], cross_kv[i, 1], cross_bias).transpose(1, 2).reshape(b, 1, self.h * self.dk)
            x = x + ca.EncDecAttention.o(o)

            ff = block.layer[2]
            hn = rms_norm(x, ff.layer_norm.weight, self.eps)
            d = ff.DenseReluDense
            if self.gated:
                inner = d.act(d.wi_0(hn)) * d.wi_1(hn)
            else:
                inner = d.act(d.wi(hn))
            x = x + d.wo(inner)
        x = rms_norm(x, dec.final_layer_norm.weight, self.eps) * self.scale
        logits = self.model.lm_head(x)[:, 0, :]
        return logits, torch.stack(presents)


def empty_past(model, batch):
    c = model.config
    return torch.zeros(c.num_decoder_layers, 2, batch, c.num_heads, 0, c.d_kv)
