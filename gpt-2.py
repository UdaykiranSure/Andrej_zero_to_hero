import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, masked = False):

        assert d_k%n_head == 0, "attention key matrix dimenstion should be multiple of num of heads"
        assert d_v%n_head == 0, "attention value matrix dimenstion should be multiple of num of heads"

        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.query = nn.Linear(d_model,d_k, bias = False)
        self.key = nn.Linear(d_model,d_k, bias = False)
        self.value = nn.Linear(d_model,d_v, bias = False)
        self.proj = nn.Linear(d_v, d_model, bias= False)
        self.masked = masked

        # if masked:
        #     self.register_buffer('mask', torch.tril(torch.ones(seq_len,seq_len)).view(1,1,seq_len,seq_len))

    def forward(self, q, k, v):
        B, out_seq_len, d_model = q.shape
        B, in_seq_len,  d_model = k.shape
        Q = self.query(q).view(B,self.n_head, out_seq_len,self.d_k//self.n_head)       #(B, n_heads, seq_len, d_k//n_heads)
        K = self.key(k).view(B,self.n_head, in_seq_len,self.d_k//self.n_head)         #(B, n_heads, seq_len, d_k//n_heads)
        V = self.value(v).view(B,self.n_head, in_seq_len,self.d_v//self.n_head)       #(B, n_heads, seq_len, d_v//n_heads)
        
        wei = (Q @ (K.transpose(-1, -2)))                         #(B, n_heads, seq_len, seq_len)
        if self.masked:
            mask = torch.tril(torch.ones(out_seq_len,in_seq_len)).view(1,1,out_seq_len,in_seq_len)
            wei = torch.masked_fill(wei,mask == 0,-torch.inf)
        att = F.softmax(wei,3) @ V                                # (B, n_heads, seq_len, d_v//n_heads)
        out = att.transpose(1,2).reshape(B , out_seq_len, d_v)        # (B, seq_len, d_v)
        out = self.proj(out)                                      # (B, seq_len, d_model)

        return out


class FFN(nn.Module):
    def __init__(self, d_model):
        super(FFN, self).__init__()
        self.d_model = d_model
        self.l1 = nn.Linear(d_model, d_model)
        self.l2 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        return out
