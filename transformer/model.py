import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import modules

# encoder block
#   N subblocks
class encoder_block(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, dropout:float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads

        # 结构
        self.multi_head_attention = modules.Multi_Head_Attention(d_model, num_heads)
        self.feed_forward = modules.Feed_Forward_Network(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # multi-head attention
        attention = self.multi_head_attention(x, x, x)
        x = x + attention
        x = self.norm1(x)

        # feed forward
        x = x + self.feed_forward(x)
        x = self.norm2(x)

        return x

# decoder block
#   N subblocks
class decoder_block(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, dropout:float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads

        # 结构
        self.masked_multi_head_attention = modules.Multi_Head_Attention(d_model, num_heads)
        self.multi_head_attention = modules.Multi_Head_Attention(d_model, num_heads)
        self.feed_forward = modules.Feed_Forward_Network(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, mask):
        # masked multi-head attention
        attention = self.masked_multi_head_attention(x, x, x, mask)
        x = x + attention
        x = self.norm1(x)

        # multi-head attention
        attention = self.multi_head_attention(x, encoder_output, encoder_output)
        x = x + attention
        x = self.norm2(x)

        # feed forward
        x = x + self.feed_forward(x)
        x = self.norm3(x)

        return x
    
# transformer
class transformer(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, num_encoder_blocks:int, num_decoder_blocks:int, dropout:float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads
        self.num_encoder_blocks = num_encoder_blocks
        self.num_decoder_blocks = num_decoder_blocks

        # encoder
        self.encoder = nn.ModuleList([encoder_block(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_blocks)])

        # decoder
        self.decoder = nn.ModuleList([decoder_block(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_blocks)])

    def forward(self, src, tgt, src_mask):
        # encoder
        for encoder_block in self.encoder:
            src = encoder_block(src)

        # decoder
        for decoder_block in self.decoder:
            tgt = decoder_block(tgt, src, src_mask)

        return tgt