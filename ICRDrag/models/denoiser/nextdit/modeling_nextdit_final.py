
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from typing import Any, Tuple, Optional
try:
    from flash_attn.bert_padding import index_first_axis, unpad_input  # noqa
except ImportError:
    index_first_axis = None
    unpad_input = None

from .layers import LLamaFeedForward, RMSNorm
import math


# import inspect
# import itertools
# import json
# import os
# import re
# from collections import OrderedDict
# from functools import partial
# from pathlib import Path
# from typing import Any, Callable, List, Optional, Tuple, Union

# import safetensors
# import torch
# from huggingface_hub import create_repo, split_torch_state_dict_into_shards
# from huggingface_hub.utils import validate_hf_hub_args
# from torch import Tensor, nn

# from diffusers import __version__
# from diffusers.utils import (
#     CONFIG_NAME,
#     FLAX_WEIGHTS_NAME,
#     SAFE_WEIGHTS_INDEX_NAME,
#     SAFETENSORS_WEIGHTS_NAME,
#     WEIGHTS_INDEX_NAME,
#     WEIGHTS_NAME,
#     _add_variant,
#     _get_checkpoint_shard_files,
#     _get_model_file,
#     deprecate,
#     is_accelerate_available,
#     is_torch_version,
#     logging,
# )
# from diffusers.utils.hub_utils import (
#     PushToHubMixin,
#     load_or_create_model_card,
#     populate_model_card,
# )
# from diffusers.models.model_loading_utils import (
#     _determine_device_map,
#     _fetch_index_file,
#     _load_state_dict_into_model,
#     load_model_dict_into_meta,
#     load_state_dict,
# )

# if is_accelerate_available():
#     import accelerate

# logger = logging.get_logger(__name__)

# if is_torch_version(">=", "1.9.0"):
#     _LOW_CPU_MEM_USAGE_DEFAULT = True
# else:
#     _LOW_CPU_MEM_USAGE_DEFAULT = False

# import frasch


def modulate(x, scale):
    return x * (1 + scale)

   

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(0, half, dtype=t.dtype) / half
        ).to(t.device)
        args = t[:, :, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(self.mlp[0].weight.dtype)
        return self.mlp(t_freq)

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, num_patches, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, num_patches * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, 1024), hidden_size),
        )
        
    def forward(self, x, c):
        scale = self.adaLN_modulation(c)
        x = modulate(self.norm_final(x), scale)
        x = self.linear(x)
        return x
    
# def center_one_init_grouped(conv_layer):
#     with torch.no_grad():
#         # 获取卷积权重张量
#         weight = self.conv.weight  # shape: [n_heads, 1, 4, k, k]
#         weight.zero_()
#         center_z = 3
#         center_y = self.conv_kernel_size // 2
#         center_x = self.conv_kernel_size // 2

#         for i in range(self.n_heads):
#             weight[i, 0, center_z, center_y, center_x] = 1.0

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        n_kv_heads=None,
        qk_norm=False,
        y_dim=0,
        base_seqlen=None,
        proportional_attn=False,
        attention_dropout=0.0,
        max_position_embeddings=384,
        point_embedding_dim=0,
        conv_kernel_size=7,
    ):
        
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.qk_norm = qk_norm
        self.y_dim = y_dim
        self.base_seqlen = base_seqlen
        self.proportional_attn = proportional_attn
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings

        self.point_embedding_dim = point_embedding_dim
        self.conv_kernel_size = conv_kernel_size

        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)

        if y_dim > 0:
            self.wk_y = nn.Linear(y_dim, self.n_kv_heads * self.head_dim, bias=False)
            self.wv_y = nn.Linear(y_dim, self.n_kv_heads * self.head_dim, bias=False)
            self.gate = nn.Parameter(torch.zeros(n_heads))

        if point_embedding_dim > 0:
            self.wk_p = nn.Linear(point_embedding_dim, self.n_kv_heads * self.head_dim, bias=False)
            self.wv_p = nn.Linear(point_embedding_dim, self.n_kv_heads * self.head_dim, bias=False)
            self.gate_p = nn.Parameter(torch.zeros(n_heads))

        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
            self.k_norm = nn.LayerNorm(self.n_kv_heads * self.head_dim)
            if y_dim > 0:
                self.ky_norm = nn.LayerNorm(self.n_kv_heads * self.head_dim, eps=1e-6)
            else:
                self.ky_norm = nn.Identity()
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.ky_norm = nn.Identity()
        
        if conv_kernel_size != 0:
            self.conv = nn.Conv3d(self.n_heads, self.n_heads, (7, conv_kernel_size, conv_kernel_size), 1, (3, conv_kernel_size // 2, conv_kernel_size // 2), groups=self.n_heads, bias=False)
            self.center_one_init_grouped()

    def center_one_init_grouped(self):
        with torch.no_grad():
            weight = self.conv.weight
            weight.zero_()
            center_z = 3
            center_y = self.conv_kernel_size // 2
            center_x = self.conv_kernel_size // 2

            for i in range(self.n_heads):
                weight[i, 0, center_z, center_y, center_x] = 1.0


    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        # xq, xk: [batch_size, seq_len, n_heads, head_dim]
        # freqs_cis: [1, seq_len, 1, head_dim]
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

        xq_complex = torch.view_as_complex(xq_)
        xk_complex = torch.view_as_complex(xk_)
        
        freqs_cis = freqs_cis.unsqueeze(2)

        # Apply freqs_cis
        xq_out = xq_complex * freqs_cis
        xk_out = xk_complex * freqs_cis

        # Convert back to real numbers
        xq_out = torch.view_as_real(xq_out).flatten(-2)
        xk_out = torch.view_as_real(xk_out).flatten(-2)

        return xq_out.type_as(xq), xk_out.type_as(xk)
    
    # copied from huggingface modeling_llama.py
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        if index_first_axis is None or unpad_input is None:
            raise ImportError("flash-attn is required only for padded flash-attention helpers.")

        def _get_unpad_data(attention_mask):
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
            return (
                indices,
                cu_seqlens,
                max_seqlen_in_batch,
            )

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.n_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def _scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        B, N_head = query.size(0), query.size(1)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(B, N_head, L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        if self.conv_kernel_size != 0:
            n_tokens = attn_weight.shape[-1]
            start_idx = n_tokens // 4 * 2
            end_idx = n_tokens // 4 * 3
            edited_img_tokens = attn_weight[:, :, start_idx:end_idx, :]
            attn_map1, attn_map2, attn_map3, attn_map4 = torch.chunk(edited_img_tokens, 4, dim=-1)
            attn_map_total = torch.stack([attn_map1, attn_map2, attn_map3, attn_map4], dim=2)
            attn_map_total = self.conv(attn_map_total)
            attn_map1, attn_map2, attn_map3, attn_map4 = torch.chunk(attn_map_total, 4, dim=2)
            edited_img_tokens = torch.concat([attn_map1, attn_map2, attn_map3, attn_map4], dim=-1)
            attn_weight[:, :, start_idx:end_idx, :] = edited_img_tokens[:, :, 0, :, :]

        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value, attn_weight
    
    def F_scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        B, N_head = query.size(0), query.size(1)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(B, N_head, L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value, attn_weight

    def forward(
        self,
        x,
        x_mask,
        freqs_cis,
        y=None,
        y_mask=None,
        init_cache=False,
        point_embedding=None,
    ):
        bsz, seqlen, _ = x.size()
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        if x_mask is None:
            x_mask = torch.ones(bsz, seqlen, dtype=torch.bool, device=x.device)
        inp_dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            xk = xk.repeat_interleave(n_rep, dim=2)
            xv = xv.repeat_interleave(n_rep, dim=2)

        freqs_cis = freqs_cis.to(xq.device)
        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis)

        if self.conv_kernel_size != 0:
            output, attn_weight = self._scaled_dot_product_attention(
                    xq.permute(0, 2, 1, 3),
                    xk.permute(0, 2, 1, 3),
                    xv.permute(0, 2, 1, 3),
                    attn_mask=x_mask.bool().view(bsz, 1, 1, seqlen).expand(-1, self.n_heads, seqlen, -1),
                    scale=None,
                )
            output = output.permute(0, 2, 1, 3).to(inp_dtype)
            
        else:
            output, attn_weight = self.F_scaled_dot_product_attention(
                    xq.permute(0, 2, 1, 3),
                    xk.permute(0, 2, 1, 3),
                    xv.permute(0, 2, 1, 3),
                    attn_mask=x_mask.bool().view(bsz, 1, 1, seqlen).expand(-1, self.n_heads, seqlen, -1),
                    scale=None,
                )
            output = output.permute(0, 2, 1, 3).to(inp_dtype)


        if hasattr(self, "wk_y"):
            yk = self.ky_norm(self.wk_y(y)).view(bsz, -1, self.n_kv_heads, self.head_dim)
            yv = self.wv_y(y).view(bsz, -1, self.n_kv_heads, self.head_dim)
            n_rep = self.n_heads // self.n_kv_heads
            # if n_rep >= 1:
            #     yk = yk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            #     yv = yv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            if n_rep >= 1:
                yk = einops.repeat(yk, "b l h d -> b l (repeat h) d", repeat=n_rep)
                yv = einops.repeat(yv, "b l h d -> b l (repeat h) d", repeat=n_rep)
            output_y = F.scaled_dot_product_attention(
                xq.permute(0, 2, 1, 3),
                yk.permute(0, 2, 1, 3),
                yv.permute(0, 2, 1, 3),
                y_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_heads, seqlen, -1).to(torch.bool),
            ).permute(0, 2, 1, 3)
            output_y = output_y * self.gate.tanh().view(1, 1, -1, 1)
            output = output + output_y
        if hasattr(self, "wk_p") and point_embedding is not None:
            pk = self.ky_norm(self.wk_p(point_embedding)).view(bsz, -1, self.n_kv_heads, self.head_dim)
            pv = self.wv_p(point_embedding).view(bsz, -1, self.n_kv_heads, self.head_dim)
            n_rep = self.n_heads // self.n_kv_heads
            # if n_rep >= 1:
            #     yk = yk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            #     yv = yv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            if n_rep >= 1:
                pk = einops.repeat(pk, "b l h d -> b l (repeat h) d", repeat=n_rep)
                pv = einops.repeat(pv, "b l h d -> b l (repeat h) d", repeat=n_rep)
            output_p = F.scaled_dot_product_attention(
                xq.permute(0, 2, 1, 3),
                pk.permute(0, 2, 1, 3),
                pv.permute(0, 2, 1, 3),
            ).permute(0, 2, 1, 3)
            output_p = output_p * self.gate_p.tanh().view(1, 1, -1, 1)
            output = output + output_p

        output = output.flatten(-2)
        output = self.wo(output)

        return output.to(inp_dtype), attn_weight

class TransformerBlock(nn.Module):
    """
    Corresponds to the Transformer block in the JAX code.
    """
    def __init__(
        self,
        dim,
        n_heads,
        n_kv_heads,
        multiple_of,
        ffn_dim_multiplier,
        norm_eps,
        qk_norm,
        y_dim,
        max_position_embeddings,
        point_embedding_dim,
        conv_kernel_size,
        image_segmentation_experts,
    ):
        super().__init__()
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm, y_dim=y_dim, max_position_embeddings=max_position_embeddings, point_embedding_dim=point_embedding_dim, conv_kernel_size=conv_kernel_size)
        if image_segmentation_experts:
            self.feed_forward_img = LLamaFeedForward(
                dim=dim,
                hidden_dim=4 * dim,
                multiple_of=multiple_of,
                ffn_dim_multiplier=ffn_dim_multiplier,
            )
            self.feed_forward_seg = LLamaFeedForward(
                dim=dim,
                hidden_dim=4 * dim,
                multiple_of=multiple_of,
                ffn_dim_multiplier=ffn_dim_multiplier,
            )
        else:
            self.feed_forward = LLamaFeedForward(
                dim=dim,
                hidden_dim=4 * dim,
                multiple_of=multiple_of,
                ffn_dim_multiplier=ffn_dim_multiplier,
            )
        
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), 4 * dim),
        )
        self.attention_y_norm = RMSNorm(y_dim, eps=norm_eps)
        self.image_segmentation_experts = image_segmentation_experts

    def forward(
        self,
        x,
        x_mask,
        freqs_cis,
        y,
        y_mask,
        adaln_input=None,
        point_embedding=None,
    ):
        if self.image_segmentation_experts:
            if adaln_input is not None:
                scales_gates = self.adaLN_modulation(adaln_input)
                # The modulation projection returns four gate/scale tensors.
                scale_msa, gate_msa, scale_mlp, gate_mlp = scales_gates.chunk(4, dim=-1)
                x = x + torch.tanh(gate_msa) * self.attention_norm2(
                    self.attention(
                        modulate(self.attention_norm1(x), scale_msa), # ok
                        x_mask,
                        freqs_cis,
                        self.attention_y_norm(y), # ok
                        y_mask,
                        point_embedding=point_embedding,
                    )[0]
                )
                # x = x + torch.tanh(gate_mlp) * self.ffn_norm2(
                #     self.feed_forward(
                #         modulate(self.ffn_norm1(x), scale_mlp),
                #     )
                # )
                ffn_norm1_results = modulate(self.ffn_norm1(x), scale_mlp)
                ori_img, ori_seg, end_img, end_seg = torch.chunk(ffn_norm1_results, 4, dim=1)
                img_tokens = torch.concat([ori_img, end_img], dim=1)
                seg_tokens = torch.concat([ori_seg, end_seg], dim=1)
                img_ffn_results = self.feed_forward_img(img_tokens)
                seg_ffn_results = self.feed_forward_seg(seg_tokens)
                ori_img, end_img = torch.chunk(img_ffn_results, 2, dim=1)
                ori_seg, end_seg = torch.chunk(seg_ffn_results, 2, dim=1)
                ffn_results = torch.concat([ori_img, ori_seg, end_img, end_seg], dim=1)
                x = x + torch.tanh(gate_mlp) * self.ffn_norm2(ffn_results)

            else:
                x = x + self.attention_norm2(
                    self.attention(
                        self.attention_norm1(x),
                        x_mask,
                        freqs_cis,
                        self.attention_y_norm(y),
                        y_mask,
                        point_embedding=point_embedding,
                    )[0]
                )
                # x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
                ffn_norm1_results = self.ffn_norm1(x)
                ori_img, ori_seg, end_img, end_seg = torch.chunk(ffn_norm1_results, 4, dim=1)
                img_tokens = torch.concat([ori_img, end_img], dim=1)
                seg_tokens = torch.concat([ori_seg, end_seg], dim=1)
                img_ffn_results = self.feed_forward_img(img_tokens)
                seg_ffn_results = self.feed_forward_seg(seg_tokens)
                ori_img, end_img = torch.chunk(img_ffn_results, 2, dim=1)
                ori_seg, end_seg = torch.chunk(seg_ffn_results, 2, dim=1)
                ffn_results = torch.concat([ori_img, ori_seg, end_img, end_seg], dim=1)
                x = x + self.ffn_norm2(ffn_results)
        else:
            if adaln_input is not None:
                scales_gates = self.adaLN_modulation(adaln_input)
                # The modulation projection returns four gate/scale tensors.
                scale_msa, gate_msa, scale_mlp, gate_mlp = scales_gates.chunk(4, dim=-1)
                x = x + torch.tanh(gate_msa) * self.attention_norm2(
                    self.attention(
                        modulate(self.attention_norm1(x), scale_msa), # ok
                        x_mask,
                        freqs_cis,
                        self.attention_y_norm(y), # ok
                        y_mask,
                        point_embedding=point_embedding,
                    )[0]
                )
                x = x + torch.tanh(gate_mlp) * self.ffn_norm2(
                    self.feed_forward(
                        modulate(self.ffn_norm1(x), scale_mlp),
                    )
                )
            else:
                x = x + self.attention_norm2(
                    self.attention(
                        self.attention_norm1(x),
                        x_mask,
                        freqs_cis,
                        self.attention_y_norm(y),
                        y_mask,
                        point_embedding=point_embedding,
                    )[0]
                )
                x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class NextDiT(ModelMixin, ConfigMixin):
    """
    Diffusion model with a Transformer backbone for image generation.
    """
    config_name = "config.json"
    @register_to_config
    def __init__(
        self,
        input_size=(1, 32, 32),
        patch_size=(1, 2, 2),
        in_channels=16,
        hidden_size=4096,
        depth=32,
        num_heads=32,
        num_kv_heads=None,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        pred_sigma=False,
        caption_channels=4096,
        qk_norm=False,
        norm_type="rms",
        model_max_length=120,
        rotary_max_length=384,
        rotary_max_length_t=None,
        point_embedding_dim=0,
        conv_kernel_size=7,
        image_segmentation_experts = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps
        self.pred_sigma = pred_sigma
        self.caption_channels = caption_channels
        self.qk_norm = qk_norm
        self.norm_type = norm_type
        self.model_max_length = model_max_length
        self.rotary_max_length = rotary_max_length
        self.rotary_max_length_t = rotary_max_length_t
        self.out_channels = in_channels * 2 if pred_sigma else in_channels

        self.point_embedding_dim = point_embedding_dim

        self.x_embedder = nn.Linear(np.prod(self.patch_size) * in_channels, hidden_size)

        self.t_embedder = TimestepEmbedder(min(hidden_size, 1024))
        self.y_embedder = nn.Sequential(
            nn.LayerNorm(caption_channels, eps=1e-6),
            nn.Linear(caption_channels, min(hidden_size, 1024)),
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=hidden_size,
                n_heads=num_heads,
                n_kv_heads=self.num_kv_heads,
                multiple_of=multiple_of,
                ffn_dim_multiplier=ffn_dim_multiplier,
                norm_eps=norm_eps,
                qk_norm=qk_norm,
                y_dim=caption_channels,
                max_position_embeddings=rotary_max_length,
                point_embedding_dim=point_embedding_dim,
                conv_kernel_size=conv_kernel_size,
                image_segmentation_experts=image_segmentation_experts,
            )
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            num_patches=np.prod(patch_size),
            out_channels=self.out_channels,
        )

        assert (hidden_size // num_heads) % 6 == 0, "3d rope needs head dim to be divisible by 6"

        self.freqs_cis = self.precompute_freqs_cis(
            hidden_size // num_heads,
            self.rotary_max_length,
            end_t=self.rotary_max_length_t
        )
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        # self.freqs_cis = self.freqs_cis.to(*args, **kwargs)
        return self
    
    @staticmethod
    def precompute_freqs_cis(
        dim: int,
        end: int,
        end_t: int = None,
        theta: float = 10000.0,
        scale_factor: float = 1.0,
        scale_watershed: float = 1.0,
        timestep: float = 1.0,
    ):
        if timestep < scale_watershed:
            linear_factor = scale_factor
            ntk_factor = 1.0
        else:
            linear_factor = 1.0
            ntk_factor = scale_factor

        theta = theta * ntk_factor
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 6)[: (dim // 6)] / dim)) / linear_factor

        timestep = torch.arange(end, dtype=torch.float32)
        freqs = torch.outer(timestep, freqs).float()
        freqs_cis = torch.exp(1j * freqs)

        if end_t is not None:
            freqs_t = 1.0 / (theta ** (torch.arange(0, dim, 6)[: (dim // 6)] / dim)) / linear_factor
            timestep_t = torch.arange(end_t, dtype=torch.float32)
            freqs_t = torch.outer(timestep_t, freqs_t).float()
            freqs_cis_t = torch.exp(1j * freqs_t)
            freqs_cis_t = freqs_cis_t.view(end_t, 1, 1, dim // 6).repeat(1, end, end, 1)
        else:
            end_t = end
            freqs_cis_t = freqs_cis.view(end_t, 1, 1, dim // 6).repeat(1, end, end, 1)
            
        freqs_cis_h = freqs_cis.view(1, end, 1, dim // 6).repeat(end_t, 1, end, 1)
        freqs_cis_w = freqs_cis.view(1, 1, end, dim // 6).repeat(end_t, end, 1, 1)
        freqs_cis = torch.cat([freqs_cis_t, freqs_cis_h, freqs_cis_w], dim=-1).view(end_t, end, end, -1)
        return freqs_cis

    def forward(
        self, 
        samples, 
        timesteps, 
        encoder_hidden_states,
        encoder_attention_mask,
        scale_factor: float = 1.0, # scale_factor for rotary embedding
        scale_watershed: float = 1.0, # scale_watershed for rotary embedding
        point_embedding=None,
    ):
        if samples.ndim == 4: # B C H W
            samples = samples[:, None, ...] # B F C H W
        
        precomputed_freqs_cis = None
        if scale_factor != 1 or scale_watershed != 1:
            precomputed_freqs_cis = self.precompute_freqs_cis(
                self.hidden_size // self.num_heads,
                self.rotary_max_length,
                end_t=self.rotary_max_length_t,
                scale_factor=scale_factor,
                scale_watershed=scale_watershed,
                timestep=torch.max(timesteps.cpu()).item()
            )
            
        if len(timesteps.shape) == 5:
            t, *_ = self.patchify(timesteps, precomputed_freqs_cis)
            timesteps = t.mean(dim=-1)
        elif len(timesteps.shape) == 1:
            timesteps = timesteps[:, None, None, None, None].expand_as(samples)
            t, *_ = self.patchify(timesteps, precomputed_freqs_cis)
            timesteps = t.mean(dim=-1)
        samples, T, H, W, freqs_cis = self.patchify(samples, precomputed_freqs_cis)
        samples = self.x_embedder(samples)
        t = self.t_embedder(timesteps)

        encoder_attention_mask_float = encoder_attention_mask[..., None].float()
        encoder_hidden_states_pool = (encoder_hidden_states * encoder_attention_mask_float).sum(dim=1) / (encoder_attention_mask_float.sum(dim=1) + 1e-8)
        encoder_hidden_states_pool = encoder_hidden_states_pool.to(samples.dtype)
        y = self.y_embedder(encoder_hidden_states_pool)
        y = y.unsqueeze(1).expand(-1, samples.size(1), -1)

        adaln_input = t + y
                                
        for block in self.layers:
            samples = block(
                samples,
                None,
                freqs_cis,
                encoder_hidden_states,
                encoder_attention_mask,
                adaln_input,
                point_embedding,
            )

        samples = self.final_layer(samples, adaln_input)
        samples = self.unpatchify(samples, T, H, W)

        return samples

    def patchify(self, x, precompute_freqs_cis=None):
        # pytorch is C, H, W
        B, T, C, H, W = x.size()
        pT, pH, pW = self.patch_size
        x = x.view(B, T // pT, pT, C, H // pH, pH, W // pW, pW)
        x = x.permute(0, 1, 4, 6, 2, 5, 7, 3)
        x = x.reshape(B, -1, pT * pH * pW * C)
        if precompute_freqs_cis is None:
            freqs_cis = self.freqs_cis[: T // pT, :H // pH, :W // pW].reshape(-1, * self.freqs_cis.shape[3:])[None].to(x.device)
        else:
            freqs_cis = precompute_freqs_cis[: T // pT, :H // pH, :W // pW].reshape(-1, * precompute_freqs_cis.shape[3:])[None].to(x.device)
        return x, T // pT, H // pH, W // pW, freqs_cis

    def unpatchify(self, x, T, H, W):
        B = x.size(0)
        C = self.out_channels
        pT, pH, pW = self.patch_size
        x = x.view(B, T, H, W, pT, pH, pW, C)
        x = x.permute(0, 1, 4, 7, 2, 5, 3, 6)
        x = x.reshape(B, T * pT, C, H * pH, W * pW)
        return x
