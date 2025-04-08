##############################################################################################
# The code is modified from ActionFormer: https://github.com/happyharrycn/actionformer_release
##############################################################################################

import math
import numpy as np
import logging
import torch
import torch.nn.functional as F
from torch import nn
from .weight_init import trunc_normal_


class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        if isinstance(padding, int):
            assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
            self.padding = (padding, padding)
        elif isinstance(padding, tuple):
            assert len(padding) == 2
            self.padding = padding
        else:
            raise ValueError("padding must be an int or a tuple of two ints")

        # stride
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, 0, dilation, groups, bias, padding_mode)
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        assert T % self.stride == 0

        # apply padding manually
        x = F.pad(x, self.padding, mode='constant', value=0)
        # conv
        out_conv = self.conv(x)
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode='nearest'
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)

        # masking the output, stop grad to mask
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask

class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """
    def __init__(
        self,
        num_channels,
        eps = 1e-5,
        affine = True,
        device = None,
        dtype = None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out


# helper functions for Transformer blocks
def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


# attention / transformers
class MaskedMHA(nn.Module):
    """
    Multi Head Attention with mask

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,          # dimension of the input embedding
        n_head,          # number of heads in multi-head self-attention
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0   # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # calculate query, key, values for all heads in batch
        # (B, nh * hs, T)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # move head forward to be the batch dim
        # (B, nh * hs, T) -> (B, nh, T, hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(mask[:, :, None, :]), float('-inf'))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = att @ (v * mask[:, :, :, None].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * mask.to(out.dtype)
        return out, mask


class MaskedMHCA(nn.Module):
    """
    Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,          # dimension of the output features
        n_head,          # number of heads in multi-head self-attention
        n_qx_stride=1,   # dowsampling stride for query and input
        n_kv_stride=1,   # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.query_norm = LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask


class LocalMaskedMHCA(nn.Module):
    """
    Local Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    The implementation is fairly tricky, code reference from
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/longformer/modeling_longformer.py
    """

    def __init__(
        self,
        n_embd,          # dimension of the output features
        n_head,          # number of heads in multi-head self-attention
        window_size,     # size of the local attention window
        n_qx_stride=1,   # dowsampling stride for query and input
        n_kv_stride=1,   # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
        use_rel_pe=False # use relative position encoding
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.window_size = window_size
        self.window_overlap  = window_size // 2
        # must use an odd window size
        assert self.window_size > 1 and self.n_head >= 1
        self.use_rel_pe = use_rel_pe

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.query_norm = LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # relative position encoding
        if self.use_rel_pe:
            self.rel_pe = nn.Parameter(
                torch.zeros(1, 1, self.n_head, self.window_size))
            trunc_normal_(self.rel_pe, std=(2.0 / self.n_embd)**0.5)

    @staticmethod
    def _chunk(x, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        # x: B x nh, T, hs
        # non-overlapping chunks of size = 2w -> B x nh, T//2w, 2w, hs
        x = x.view(
            x.size(0),
            x.size(1) // (window_overlap * 2),
            window_overlap * 2,
            x.size(2),
        )

        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2

        # B x nh, #chunks = T//w - 1, 2w, hs
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(x, padding):
        """pads rows and then flips rows and columns"""
        # padding value is not important because it will be overwritten
        x = nn.functional.pad(x, padding)
        x = x.view(*x.size()[:-2], x.size(-1), x.size(-2))
        return x

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len):
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        # `== 1` converts to bool or uint8
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        # `== 1` converts to bool or uint8
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))

    @staticmethod
    def _pad_and_diagonalize(x):
        """
        shift every row 1 step right, converting columns into diagonals.
        Example::
              chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                       -1.8348,  0.7672,  0.2986,  0.0285,
                                       -0.7584,  0.4206, -0.0405,  0.1599,
                                       2.0514, -1.1600,  0.5372,  0.2629 ]
              window_overlap = num_rows = 4
             (pad & diagonalize) =>
             [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
               0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
               0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
               0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = x.size()
        # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1).
        x = nn.functional.pad(
            x, (0, window_overlap + 1)
        )
        # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        x = x.view(total_num_heads, num_chunks, -1)
        # total_num_heads x num_chunks x window_overlap*window_overlap
        x = x[:, :, :-window_overlap]
        x = x.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        x = x[:, :, :, :-1]
        return x

    def _sliding_chunks_query_key_matmul(
        self, query, key, num_heads, window_overlap
    ):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This implementation splits the input into overlapping chunks of size 2w with an overlap of size w (window_overlap)
        """
        # query / key: B*nh, T, hs
        bnh, seq_len, head_dim = query.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # B * num_heads, head_dim, #chunks=(T//w - 1), 2w
        chunk_query = self._chunk(query, window_overlap)
        chunk_key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = torch.einsum(
            "bcxd,bcyd->bcxy", (chunk_query, chunk_key))

        # convert diagonals into columns
        # B * num_heads, #chunks, 2w, 2w+1
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
        ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap :
        ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs, value, num_heads, window_overlap
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        bnh, seq_len, head_dim = value.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # step 1: depth convolutions
        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # step 2: query, key, value transforms & reshape
        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        # (B, nh * hs, T) -> (B, nh, T, hs)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # view as (B * nh, T, hs)
        q = q.view(B * self.n_head, -1, self.n_channels).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels).contiguous()

        # step 3: compute local self-attention with rel pe and masking
        q *= self.scale
        # chunked query key attention -> B, T, nh, 2w+1 = window_size
        att = self._sliding_chunks_query_key_matmul(
            q, k, self.n_head, self.window_overlap)

        # rel pe
        if self.use_rel_pe:
            att += self.rel_pe
        # kv_mask -> B, T'', 1
        inverse_kv_mask = torch.logical_not(
            kv_mask[:, :, :, None].view(B, -1, 1))
        # 0 for valid slot, -inf for masked ones
        float_inverse_kv_mask = inverse_kv_mask.type_as(q).masked_fill(
            inverse_kv_mask, -1e4)
        # compute the diagonal mask (for each local window)
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_inverse_kv_mask.new_ones(size=float_inverse_kv_mask.size()),
            float_inverse_kv_mask,
            1,
            self.window_overlap
        )
        att += diagonal_mask

        # ignore input masking for now
        att = nn.functional.softmax(att, dim=-1)
        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        att = att.masked_fill(
            torch.logical_not(kv_mask.squeeze(1)[:, :, None, None]), 0.0)
        att = self.attn_drop(att)

        # step 4: compute attention value product + output projection
        # chunked attn value product -> B, nh, T, hs
        out = self._sliding_chunks_matmul_attn_probs_value(
            att, v, self.n_head, self.window_overlap)
        # transpose to B, nh, hs, T -> B, nh*hs, T
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask


class TransformerBlock(nn.Module):
    """
    A simple (post layer norm) Transformer block
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """
    def __init__(
        self,
        n_embd,                # dimension of the input features
        n_head,                # number of attention heads
        n_ds_strides=(1, 1),   # downsampling strides for q & x, k & v
        n_out=None,            # output dimension, if None, set to input dim
        n_hidden=None,         # dimension of the hidden layer in MLP
        act_layer=nn.GELU,     # nonlinear activation used in MLP, default GELU
        attn_pdrop=0.0,        # dropout rate for the attention map
        proj_pdrop=0.0,        # dropout rate for the projection / MLP
        path_pdrop=0.0,        # drop path rate
        mha_win_size=-1,       # > 0 to use window mha
        use_rel_pe=False,      # if to add rel position encoding to attention
        online_mode=False      # if to use causal attention
    ):
        super().__init__()
        assert len(n_ds_strides) == 2
        # layer norm for order (B C T)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

        # specify the attention module
        if mha_win_size > 1:
            if online_mode:
                self.attn = CausalLocalMultiHeadSelfConvAttention(
                    n_embd,
                    n_head,
                    mha_win_size,
                    n_qx_stride=n_ds_strides[0],
                    n_kv_stride=n_ds_strides[1],
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                )
            else:
                self.attn = LocalMaskedMHCA(
                    n_embd,
                    n_head,
                    window_size=mha_win_size,
                    n_qx_stride=n_ds_strides[0],
                    n_kv_stride=n_ds_strides[1],
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    use_rel_pe=use_rel_pe  # only valid for local attention
                )
        else:
            self.attn = MaskedMHCA(
                n_embd,
                n_head,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop
            )

        # input
        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = \
                n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1)//2
            self.pool_skip = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd
        # ok to use conv1d here with stride=1
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob = path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob = path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, x, mask, pos_embd=None):
        # pre-LN transformer: https://arxiv.org/pdf/2002.04745.pdf
        out, out_mask = self.attn(self.ln1(x), mask)
        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip(x) * out_mask_float + self.drop_path_attn(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.ln2(out)) * out_mask_float)
        # optionally add pos_embd to the output
        if pos_embd is not None:
            out += pos_embd * out_mask_float
        return out, out_mask


class ConvBlock(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """
    def __init__(
        self,
        n_embd,                # dimension of the input features
        kernel_size=3,         # conv kernel size
        n_ds_stride=1,         # downsampling stride for the current layer
        expansion_factor=2,    # expansion factor of feat dims
        n_out=None,            # output dimension, if None, set to input dim
        act_layer=nn.ReLU,     # nonlinear activation used after conv, default ReLU
    ):
        super().__init__()
        # must use odd sized kernel
        assert (kernel_size % 2 == 1) and (kernel_size > 1)
        padding = kernel_size // 2
        if n_out is None:
            n_out = n_embd

         # 1x3 (strided) -> 1x3 (basic block in resnet)
        width = n_embd * expansion_factor
        self.conv1 = MaskedConv1D(
            n_embd, width, kernel_size, n_ds_stride, padding=padding)
        self.conv2 = MaskedConv1D(
            width, n_out, kernel_size, 1, padding=padding)

        # attach downsampling conv op
        if n_ds_stride > 1:
            # 1x1 strided conv (same as resnet)
            self.downsample = MaskedConv1D(n_embd, n_out, 1, n_ds_stride)
        else:
            self.downsample = None

        self.act = act_layer()

    def forward(self, x, mask, pos_embd=None):
        identity = x
        out, out_mask = self.conv1(x, mask)
        out = self.act(out)
        out, out_mask = self.conv2(out, out_mask)

        # downsampling
        if self.downsample is not None:
            identity, _ = self.downsample(x, mask)

        # residual connection
        out += identity
        out = self.act(out)

        return out, out_mask


# drop path: from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """
    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale


# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)


class TemporalGCNConv1d(nn.Module):
    def __init__(
        self,
        n_embd,                # dimension of the input features
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(n_embd, n_embd, kernel_size=5, stride=1, padding=6, dilation=3),
            nn.BatchNorm1d(n_embd),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_embd, n_embd, kernel_size=5, stride=1, padding=6, dilation=3),
            nn.BatchNorm1d(n_embd ),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.main(x)
        
class GCNBlock(nn.Module):
    """
    A simple Graph Convolutional Block
    """
    def __init__(
        self,
        n_embd,                # dimension of the input features
        num_node,              # num of node
        num_classes = 43,      # number of object types
        downsample = False,
        scale_factor = 1.0
    ):
        super().__init__()
        feat_dim = n_embd
        self.feat_dim = feat_dim
        self.num_node = num_node
        self.linear = nn.Linear(4, feat_dim, bias=False)
        self.embedding = nn.Embedding(num_classes, feat_dim, padding_idx=0)
        self.relu = nn.ReLU()
        self.conv2d_1_weight = nn.Conv2d(feat_dim * 2, feat_dim, 1, 1, 0, bias=False)
        self.conv2d_2_weight = nn.Conv2d(feat_dim, feat_dim, 1, 1, 0, bias=False)
        self.conv1d_merge_node_weight = nn.Conv1d(feat_dim, feat_dim, num_node, 1, 0, bias=False)
        self.layernorm = LayerNorm(self.feat_dim)
        self.downsample = downsample
        self.scale_factor = scale_factor

    def generate_graph_feature(self, bbox, bbox_class, edge_map):
        B, T, N, C = bbox_class.size()
        node = torch.cat((bbox, bbox_class), dim=3).permute(0, 3, 1, 2) # becomes (B, C, T, N)
        node = self.conv2d_1_weight(node).permute(0, 2, 3, 1) # becomes (B, T, N, C)
        node = torch.einsum('btij,btjk->btik', edge_map, node)
        node = self.relu(node).permute(0, 3, 1, 2) # becomes (B, C, T, N)
        node = self.conv2d_2_weight(node).permute(0, 2, 3, 1) # becomes (B, T, N, C)
        node = torch.einsum('btij,btjk->btik', edge_map, node)
        node = self.relu(node)
        node = node.permute(0, 1, 3, 2).reshape(-1, self.feat_dim, self.num_node)
        node = self.conv1d_merge_node_weight(node)
        return node.reshape(B, T, self.feat_dim)
    
    def forward(self, bbox, bbox_class, edge_map):
        bbox = bbox.permute(0, 3, 1, 2) # B, T, N, 4
        bbox_class = bbox_class.permute(0, 2, 1) # B, T, N
        edge_map = edge_map.permute(0, 3, 1, 2) # B, T, N, N
        bbox_class_embed = self.embedding(bbox_class)
        bbox_embed = self.linear(bbox)
        feature = self.generate_graph_feature(bbox_class_embed, bbox_embed, edge_map)

        if self.downsample:
            return self.layernorm(F.interpolate(feature.permute(0, 2, 1), scale_factor=self.scale_factor, mode='linear'))
        else:
            return self.layernorm(feature.permute(0, 2, 1))


class OF_module(nn.Module):

    def __init__(
        self,
        n_embd,                # dimension of the input features
    ):
        super().__init__()
        self.main = nn.Sequential(
            # nn.Conv2d(2, 16, 3, 1, 1),
            # nn.ReLU(),
            # nn.Conv2d(16, n_embd, 3, 1, 1),
            # nn.ReLU(),
            # nn.LayerNorm(n_embd)
            nn.Conv1d(2048, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(512, n_embd, 3, 1, 1),
            nn.ReLU(),
            LayerNorm(n_embd)
        )
    def forward(self, x):
        return self.main(x)
    


class CausalLocalMultiHeadSelfConvAttention(nn.Module):
    """
    Causal Local Multi-Head Self-Attention with Depthwise Convolution.
    This implementation ensures that attention is only computed on past and current time steps.
    """

    def __init__(self, embed_dim, num_heads, win_len, n_qx_stride=1, n_kv_stride=1, attn_pdrop=0.0, proj_pdrop=0.0):
        super(CausalLocalMultiHeadSelfConvAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.win_len = win_len
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.scale = math.sqrt(self.head_dim)

        # Set the kernel size and stride for the convolutions
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride
        kernel_size_q = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        kernel_size_kv = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3

        self.kernel_size_q = kernel_size_q
        self.kernel_size_kv = kernel_size_kv

        # Adjusted conv layers for query, key, and value with specified strides
        self.conv_q = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size_q, stride=self.n_qx_stride, padding=0, groups=embed_dim, bias=False)
        self.conv_k = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size_kv, stride=self.n_kv_stride, padding=0, groups=embed_dim, bias=False)
        self.conv_v = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size_kv, stride=self.n_kv_stride, padding=0, groups=embed_dim, bias=False)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, C, T = x.size()

        # Apply convolutions to get queries, keys, and values
        q = self.conv_q(F.pad(x, (self.kernel_size_q - 1, 0))).view(B, self.num_heads, self.head_dim, -1)
        k = self.conv_k(F.pad(x, (self.kernel_size_kv - 1, 0))).view(B, self.num_heads, self.head_dim, -1)
        v = self.conv_v(F.pad(x, (self.kernel_size_kv - 1, 0))).view(B, self.num_heads, self.head_dim, -1)
        
        # Pad keys and values to ensure consistent window length
        k = F.pad(k, (self.win_len, 0), value=0)
        v = F.pad(v, (self.win_len, 0), value=0)

        # Unfold k and v to get local windows
        k_unfolded = k.unfold(-1, self.win_len + 1, 1)  # (B, num_heads, head_dim, T, win_len + 1)
        v_unfolded = v.unfold(-1, self.win_len + 1, 1)  # (B, num_heads, head_dim, T, win_len + 1)

        # Scaled dot-product attention for local windows
        attn_scores = torch.einsum('bhdt,bhdtl->bhdtl', q, k_unfolded) / self.scale  # Adjusted einsum

        # Check NaN
        if torch.isnan(attn_scores).any():
            nan_mask = torch.isnan(attn_scores)
            nan_indices = torch.nonzero(nan_mask, as_tuple=False)
            logging.error("NaN positions: {:s}".format(str(nan_indices)))
            logging.error("NaN values: {:s}".format(str(attn_scores[nan_indices])))
            logging.error("Mask values: {:s}".format(str(mask[nan_indices])))
            logging.error("Query values: {:s}".format(str(q[nan_indices])))
            logging.error("Key values: {:s}".format(str(k_unfolded[nan_indices])))
            raise ValueError("NaN detected in attention scores")

        rt_mask = None
        if mask is not None:
            mask = mask[:, :, ::self.n_kv_stride]
            rt_mask = mask.clone()
            mask = F.pad(mask, (self.win_len, 0), value=0)
            mask_unfolded = mask.unfold(-1, self.win_len + 1, 1).unsqueeze(1)  # (B, 1, 1, T, win_len + 1)
            attn_scores = attn_scores.masked_fill(mask_unfolded == 0, -1e9)

        if torch.isnan(attn_scores).any():
            raise ValueError("NaN detected in attention scores")

        attn_probs = F.softmax(attn_scores, dim=-1)

        if torch.isnan(attn_probs).any():
            raise ValueError("NaN detected in attention probabilities")

        attn_probs = self.attn_drop(attn_probs)

        if torch.isnan(attn_probs).any():
            raise ValueError("NaN detected in attention probabilities")

        attn_output = torch.einsum('bhdtl,bhdtl->bhdt', attn_probs, v_unfolded)  # Adjusted einsum

        # Check NaN
        if torch.isnan(attn_output).any():
            nan_mask = torch.isnan(attn_output)
            nan_indices = torch.nonzero(nan_mask, as_tuple=False)
            logging.error("NaN positions: {:s}".format(str(nan_indices)))
            logging.error("NaN values: {:s}".format(str(attn_output[nan_indices])))
            logging.error("Mask values: {:s}".format(str(mask[nan_indices])))
            logging.error("Query values: {:s}".format(str(q[nan_indices])))
            logging.error("Key values: {:s}".format(str(k_unfolded[nan_indices])))
            raise ValueError("NaN detected in attention output")

        # Concatenate heads and put through final linear layer
        attn_output = attn_output.contiguous().view(B, self.embed_dim, -1)
        out = self.fc_out(attn_output.transpose(1, 2)).transpose(1, 2)  # (B, C, T)
        out = self.proj_drop(out)
        
        # Return the output and the mask
        return out, rt_mask
    






class CausalLocalMultiHeadCrossConvAttention(nn.Module):
    """
    Causal Local Multi-Head Cross-Attention with Depthwise Convolution.
    This implementation ensures that attention is only computed on past and current time steps in k and v.
    """

    def __init__(self, embed_dim, num_heads, win_len, n_qx_stride=1, n_kv_stride=1, attn_pdrop=0.0, proj_pdrop=0.0):
        super(CausalLocalMultiHeadCrossConvAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.win_len = win_len
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.scale = math.sqrt(self.head_dim)

        # Set the kernel size and stride for the convolutions
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride
        kernel_size_q = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        kernel_size_kv = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3

        self.kernel_size_q = kernel_size_q
        self.kernel_size_kv = kernel_size_kv

        # Adjusted conv layers for query, key, and value with specified strides
        self.conv_q = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size_q, stride=self.n_qx_stride, padding=0, groups=embed_dim, bias=False)
        self.conv_k = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size_kv, stride=self.n_kv_stride, padding=0, groups=embed_dim, bias=False)
        self.conv_v = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size_kv, stride=self.n_kv_stride, padding=0, groups=embed_dim, bias=False)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, q_input, k_input, v_input, mask=None):
        B, C, T_q = q_input.size()
        _, _, T_kv = k_input.size()

        # 对查询、键和值进行卷积
        q = self.conv_q(F.pad(q_input, (self.kernel_size_q - 1, 0))).view(B, self.num_heads, self.head_dim, -1)
        k = self.conv_k(F.pad(k_input, (self.kernel_size_kv - 1, 0))).view(B, self.num_heads, self.head_dim, -1)
        v = self.conv_v(F.pad(v_input, (self.kernel_size_kv - 1, 0))).view(B, self.num_heads, self.head_dim, -1)

        # 获取卷积后的长度
        T_q_new = q.size(-1)
        T_kv_new = k.size(-1)

        # 对键和值进行因果填充
        k = F.pad(k, (self.win_len, 0), value=0)
        v = F.pad(v, (self.win_len, 0), value=0)

        # 为每个查询位置 t，获取键和值的局部窗口（确保因果性）
        k_unfolded = k.unfold(-1, self.win_len + 1, 1)
        v_unfolded = v.unfold(-1, self.win_len + 1, 1)

        # 计算注意力得分
        attn_scores = torch.einsum('bhdt,bhdtl->bhdtl', q, k_unfolded) / self.scale

        rt_mask = None
        if mask is not None:
            mask = mask[:, :, ::self.n_kv_stride]
            rt_mask = mask.clone()
            mask = F.pad(mask, (self.win_len, 0), value=0)
            mask_unfolded = mask.unfold(-1, self.win_len + 1, 1).unsqueeze(1)  # (B, 1, 1, T, win_len + 1)
            attn_scores = attn_scores.masked_fill(mask_unfolded == 0, -1e9)

        # 计算注意力概率
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        # 计算注意力输出
        attn_output = torch.einsum('bhdtl,bhdtl->bhdt', attn_probs, v_unfolded)

        # 拼接多头并通过线性层
        attn_output = attn_output.contiguous().view(B, self.embed_dim, -1)
        out = self.fc_out(attn_output.transpose(1, 2)).transpose(1, 2)
        out = self.proj_drop(out)

        return out, rt_mask