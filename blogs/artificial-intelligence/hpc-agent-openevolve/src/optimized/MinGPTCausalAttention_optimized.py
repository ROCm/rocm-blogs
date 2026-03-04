import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # Separate linear layers for Q, K, V
        self.query_linear = nn.Linear(n_embd, n_embd)
        self.key_linear = nn.Linear(n_embd, n_embd)
        self.value_linear = nn.Linear(n_embd, n_embd)
        # output projection
        self.output_linear = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()

        # Compute Q, K, V with separate projections
        q = self.query_linear(x)
        k = self.key_linear(x)
        v = self.value_linear(x)
        
        # Reshape for multi-head attention and transpose for matrix multiplication
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        
        # Vectorized attention score computation using matrix multiplication
        # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Apply causal mask
        attention_scores = attention_scores.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
        
        # Vectorized weighted sum using matrix multiplication
        # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        output = attention_weights @ v
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.resid_dropout(self.output_linear(output))
        
        return output

batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
