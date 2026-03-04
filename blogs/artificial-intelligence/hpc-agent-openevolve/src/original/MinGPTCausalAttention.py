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
        
        # Reshape for multi-head attention
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim)
        k = k.view(B, T, self.n_head, head_dim)
        v = v.view(B, T, self.n_head, head_dim)
        
        # Manual attention score computation
        attention_scores = torch.zeros(B, self.n_head, T, T, device=x.device)
        
        for batch_idx in range(B):
            for head_idx in range(self.n_head):
                for pos_i in range(T):
                    for pos_j in range(T):
                        score = torch.sum(q[batch_idx, pos_i, head_idx, :] * k[batch_idx, pos_j, head_idx, :])
                        attention_scores[batch_idx, head_idx, pos_i, pos_j] = score / math.sqrt(head_dim)
        
        # Apply causal mask
        attention_scores = attention_scores.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
        
        # Manual weighted sum
        output = torch.zeros(B, self.n_head, T, head_dim, device=x.device)
        for batch_idx in range(B):
            for head_idx in range(self.n_head):
                for pos_i in range(T):
                    for pos_j in range(T):
                        output[batch_idx, head_idx, pos_i, :] += attention_weights[batch_idx, head_idx, pos_i, pos_j] * v[batch_idx, pos_j, head_idx, :]
        
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
