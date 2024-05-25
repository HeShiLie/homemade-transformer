import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# define the positional_encoding in section 3.5
# 用函数实现吧，它也没参数可学
def positional_encoding(input_sequence:torch.tensor, d_model:int):
    '''
    input: input_sequence, d_model
    input_sequence is a tensor shaped of [batch_size, seq_len, d_model];
    d_model is the dimension of the model as well the dimention of x;

    output: PE, a tensor shaped of [seq_len, d_model]
    这种PE既学到了x在序列中的位置信息,也学到了x里的元素的相互位置信息
    '''
    # max_len of the sequence
    max_len = input_sequence.size()[1]

    # every sequence in the batch shares the same positional encoding
    # PE should be a tensor of shape [batch_size, seq_len, d_model], but we adapt it to [seq_len, d_model]
    # thus, PE(pos)should be a dicrete sin function, of which parameters depend on pos. 
    PosEncoding = 10000**(torch.arange(0, d_model, 2)/d_model)

    PE = torch.zeros(max_len, d_model)
    for pos in range(max_len):
        PE[pos, 0::2] = torch.sin(pos/PosEncoding)
        PE[pos, 1::2] = torch.cos(pos/PosEncoding)

    return PE

# Multi-head Attention
#   first, scaled dot-product attention
#   (since it's also a non-parametric function, we implement it as a function)
#       
#       first of first, define two ways of making masks
#          (implement by 2 functions)
#           1. padding mask, which is used to mask the padding tokens of value 0.
def create_padding_mask(input_sequence:torch.tensor):
    '''
    input: input_sequence
    input_sequence is a tensor of shape [batch_size, seq_len, seq_len] (also the shape of scaled_QK)
    output: mask, a tensor of shape [batch_size, seq_len, seq_len]

    differen from other multi-head implementations, we implement on only one head.
    '''
    mask = (input_sequence == 0).unsqueeze(1)
    return mask
#          2. look-ahead mask, which is used to mask the future tokens. 
def create_look_ahead_mask(input_sequence:torch.tensor):
    '''
    input: input_sequence, position_to_mask
    input_sequence is a tensor of shape [batch_size, seq_len, seq_len] (also the shape of scaled_QK)
    position_to_mask is the position that we want to mask in the sequence

    output: mask, a tri-angular tensor of shape [batch_size, seq_len, seq_len]
    '''
    # get the seq_len
    seq_len = input_sequence.size()[1]
    # get the tri-angular tensor
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask

def scaled_dot_production(Query : torch.tensor, Key : torch.tensor, Value : torch.tensor, mask=None):
    '''
    input: Query, Key, Value, mask
    Query, Key, Value are all tensors of shape [batch_size, seq_len, d_k]
    mask is a boolean value, if True, the function will apply mask to the attention weights
    d_k is the dimension of the Query, Key, Value, not the dimension of the model

    output: attention, a tensor of shape [batch_size, seq_len, d_k]
    '''

    # first deal with the left part, Q and K
    # get the d_k for scaling
    d_k = Query.size()[-1]

    # note that the shape of the scaled_QK is [batch_size, seq_len, seq_len]
    scaled_QK = torch.matmul(Query, Key.transpose(-2, -1))/torch.sqrt(torch.tensor(d_k).float())

    # apply mask
    if mask is not None:
        scaled_QK =scaled_QK.masked_fill(mask == 0, -1e9)

    # softmax
    scaled_QK = F.softmax(scaled_QK, dim=-1)

    # final attention
    attention = torch.matmul(scaled_QK, Value)
    return attention

#   second, multi-head attention
#       (since it's a parametric function, we implement it as a subclass of nn.Module)
class Multi_Head_Attention(nn.Module):
    def __init__(self, d_model:int, num_heads:int):
        super(Multi_Head_Attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # get the d_k, d_k = d_model/num_heads
        self.d_k = d_model//num_heads

        # pre-linear layers, 
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        # post-linear layer
        self.WO = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask = None):
        '''
        input: Q, K, V, mask
        Q, K, V are all tensors of shape [batch_size, seq_len, d_model]
        mask is a boolean value, if True, the function will apply mask to the attention weights

        output: attention, a tensor of shape [batch_size, seq_len, d_model]
        '''
        # get the batch_size
        batch_size = Q.size()[0]

        # pre-linear layers
        Q = self.WQ(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2)
        K = self.WK(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2)
        V = self.WV(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2)

        # apply scaled dot-production
        # since our scaled_dot_production is implemented on one head, we need to loop over the heads
        #   and then concatenate the results
        #   the shape of input to the scaled_dot_production is [batch_size, seq_len, seq_len]
        #       thus, we need to transpose the Q, K, V to [num_heads, batch_size, seq_len, d_k]
        #       and the squeeze the first dimension
        attention_list = []
        for i in range(self.num_heads):
            attention = scaled_dot_production(Q[0].squeeze(0), K[0].squeeze(0), V[0].squeeze(0), mask)
            attention_list.append(attention)
        # get an attention of shape [batch_size, seq_len, d_model]
        attention = torch.cat(attention_list, dim=-1)

        # post-linear layer
        attention = attention.contiguous()
        attention = self.WO(attention)

        return attention

# Feed-Forward Networks
# since it is parametrilized, we should implement it as a subclass of nn.Module
class Feed_Forward_Network(nn.Module):
    def __init__(self, d_model, d_ff, drop_out = .1):
        '''
        d_model is the lenth of every elements in the sequece
        d_ff is the inner dimension of the matrices
        '''
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)
        self.linear2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        x = self.ReLU(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x