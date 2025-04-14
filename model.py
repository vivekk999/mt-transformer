import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.hidden_size)


class PosEncoding(nn.Module):
    def __init__(self, hidden_size: int, seq_len: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Positional encodings matrix of shape (seq_len,hidden_size)
        pe = torch.zeros(seq_len, hidden_size)

        # Positions vector shape (seq_len,1)
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size)
        )

        # Apply PE to even and odd positions with sin and cos functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1,seq_len,hidden_size)

        self.register_buffer("pe", pe)  # persistent buffer

    def forward(self, x):
        # Add positional encodings with dim (bat,seq_len of x,hidden_size) to x
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))  # Multiplicative
        self.beta = nn.Parameter(torch.zeros(hidden_size))  # Additive
        self.eps = eps  # Prevent division by 0

    def forward(self, x):
        # Mean and std calculated arcoss hidden_size | (bat,seq_len,1)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return ((x - mean) / (std + self.eps)) * self.gamma + self.beta


class FFN_block(nn.Module):
    def __init__(self, hidden_size: int, ff_dim: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # First linear transformation: hidden_size => ff_dim
        self.linear1 = nn.Linear(hidden_size, ff_dim)
        # Second linear transformation: ff_dim => model_dim
        self.linear2 = nn.Linear(ff_dim, hidden_size)
        self.relu = nn.ReLU()

    # FFN(x) = relu(xW1+b)W2 + b2
    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(self.relu(x))
        x = self.linear2(x)
        return x


class MHAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, h: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.h = h
        assert hidden_size % h == 0, "hidden_size is not divisible by h"
        self.d_k = hidden_size // h
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)

    @staticmethod
    # softmax(q.k^T/sqrt(dk))*v
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (bat,h,seq_len,d_k) @ (bat,h,d_k,seq_len) => (bat,h,seq_len,seq_len)
        attention_matrix = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_matrix.masked_fill(mask == 0, -1e9)
        attention_matrix = attention_matrix.softmax(dim=-1)  # (bat,h,seq_len,seq_len)
        if dropout is not None:
            attention_matrix = dropout(attention_matrix)
        return (attention_matrix @ value), attention_matrix

    def forward(self, q, k, v, mask):
        # (bat,seq_len,hidden_size) => (bat,seq_len,hidden_size)
        query = self.w_q(q)  # Q . Wq = Q'
        key = self.w_k(k)  # K . Wk = K'
        value = self.w_v(v)  # V . Wv = V'

        # (bat,seq_len,hidden_size) => (bat,seq_len,h,d_k), => (bat,h,seq_len,d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        x, self.attentionScores = MHAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )
        #  Concatenate heads:
        #  (bat,h,seq_len,d_k) => (bat,seq_len,h,d_k) => (bat,seq_len,hidden_size)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)


class SkipConnection(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sub_layer):
        # (sublayer input x) + { norm(x) -> sublayer -> dropout}
        return x + self.dropout(sub_layer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        self_attn_block: MHAttentionBlock,
        ffn_block: FFN_block,
        dropout: float,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.self_attn_block = self_attn_block
        self.ffn_block = ffn_block
        self.skip_conn = nn.ModuleList(
            [SkipConnection(hidden_size, dropout) for _ in range(2)]
        )

    def forward(self, x, encoder_mask):  # Encoder mask for masking padding tokens
        x = self.skip_conn[0](x, lambda x: self.self_attn_block(x, x, x, encoder_mask))
        x = self.skip_conn[1](x, self.ffn_block)
        return x


class Encoder(nn.Module):
    def __init__(self, hidden_size: int, num_hidden_layers: nn.ModuleList):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.norm = LayerNorm(hidden_size)

    def forward(self, x, mask):
        for layer in self.num_hidden_layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        self_attn: MHAttentionBlock,
        cross_attn: MHAttentionBlock,
        ffn_block: FFN_block,
        dropout: float,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ffn_block = ffn_block
        self.skip_conn = nn.ModuleList(
            [SkipConnection(hidden_size, dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        #  Masked MultiHeadAttention block
        x = self.skip_conn[0](x, lambda x: self.self_attn(x, x, x, decoder_mask))
        #  Self Attention Block with query == x, (k,v) == encoder_output
        x = self.skip_conn[1](
            x,
            lambda x: self.cross_attn(x, encoder_output, encoder_output, encoder_mask),
        )
        x = self.skip_conn[2](x, self.ffn_block)
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_size: int, num_hidden_layers: nn.ModuleList):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.norm = LayerNorm(hidden_size)

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        for layer in self.num_hidden_layers:
            x = layer(x, encoder_output, encoder_mask, decoder_mask)
        return self.norm(x)


class LinearLayer(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        src_emb: InputEmbeddings,
        tgt_emb: InputEmbeddings,
        src_pos: PosEncoding,
        tgt_pos: PosEncoding,
        encoder: Encoder,
        decoder: Decoder,
        linear_layer: LinearLayer,
    ):
        super().__init__()
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.encoder = encoder
        self.decoder = decoder
        self.linear_layer = linear_layer

    def encode(self, input, encoder_mask):
        # Input Embeddings -> Positional Encoding -> Encoder Block
        input = self.src_pos(self.src_emb(input))
        return self.encoder(input, encoder_mask)

    def decode(self, encoder_output, encoder_mask, target, decoder_mask):
        # Input Embeddings -> Positional Encoding -> Decoder Block
        target = self.tgt_pos(self.tgt_emb(target))
        return self.decoder(target, encoder_output, encoder_mask, decoder_mask)

    def linear_proj(self, x):
        # Projection for token -> vocab word
        return self.linear_layer(x)


# Build the transformer with default hyperparameter config


def build(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    hidden_size: int = 512,
    dropout: float = 0.1,
    attention_heads: int = 8,
    num_hidden_layers: int = 6,
    ff_dim: int = 2048,
) -> Transformer:
    # Embedding Layers:
    src_emb = InputEmbeddings(hidden_size, src_vocab_size)
    tgt_emb = InputEmbeddings(hidden_size, tgt_vocab_size)

    # Positional Encoding:
    src_pos = PosEncoding(hidden_size, src_seq_len, dropout)
    tgt_pos = PosEncoding(hidden_size, tgt_seq_len, dropout)

    # Encoder blocks
    encoder_blocks = []
    for _ in range(num_hidden_layers):
        encoder_self_attn = MHAttentionBlock(hidden_size, attention_heads, dropout)
        ffn_block = FFN_block(hidden_size, ff_dim, dropout)
        encoder_block = EncoderBlock(hidden_size, encoder_self_attn, ffn_block, dropout)
        encoder_blocks.append(encoder_block)

    # Decoder_blocks
    decoder_blocks = []
    for _ in range(num_hidden_layers):
        decoder_self_attn = MHAttentionBlock(hidden_size, attention_heads, dropout)
        decoder_cross_attn = MHAttentionBlock(hidden_size, attention_heads, dropout)
        ffn_block = FFN_block(hidden_size, ff_dim, dropout)
        decoder_block = DecoderBlock(
            hidden_size, decoder_self_attn, decoder_cross_attn, ffn_block, dropout
        )
        decoder_blocks.append(decoder_block)

    # Encoder and Decoder
    encoder = Encoder(hidden_size, nn.ModuleList(encoder_blocks))
    decoder = Decoder(hidden_size, nn.ModuleList(decoder_blocks))

    # Linear Layer
    linear_layer = LinearLayer(hidden_size, tgt_vocab_size)

    # Initialize the transformer
    transformer = Transformer(
        src_emb, tgt_emb, src_pos, tgt_pos, encoder, decoder, linear_layer
    )

    # Parameter Initilization
    for p in transformer.parameters():
        # Initilise for tensors with dim>1 ie only weights and not biases
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
