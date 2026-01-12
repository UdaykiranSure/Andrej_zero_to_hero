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



class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, vocab_size, n_head):
        super(DecoderBlock, self).__init__()

        self.self_att = MultiHeadAttention(d_model, d_k, d_v, n_head, masked= True)
        self.cross_att = MultiHeadAttention(d_model, d_k, d_v, n_head)
        self.ffn = FFN(d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, enc_out):
        """
        params: 
        x: positional encoded output token sequence. #(seq_len, d_model)
        enc_out: encoder output hidden represenations of input sequence

        outputs:
        out: encoder output representation of input sequence  #(seq_len, d_model)
        """

        out = self.self_att(x,x,x)
        att1_out = self.ln(x + out)
        att2_out = self.cross_att(att1_out, enc_out, enc_out)
        out = self.ln(att2_out + att1_out)
        out = self.ln(self.ffn(out) + out)

        return out

# d_model = 32
# d_k = d_v = 16
# n_heads = 4
# seq_len = 3
# vocab_size = 200
# enc = EncoderBlock(d_model,d_k, d_v, vocab_size, n_heads)

# x = torch.rand(4,seq_len,d_model)
# enc_out = enc(x)

# dec  = DecoderBlock(d_model, d_k, d_v, vocab_size, n_heads)
# out = dec(x,enc_out)
# out.shape


class Model(nn.Module):
    def __init__(self, d_model, d_k, d_v, vocab_size, n_head, n_layers = 1, ):
        """
        d_modle: dimensional size of hidden representations
        d_k:  dimentional size of key and query projection matrix
        d_v: dimentional size of value projection matrix
        vocab_size: size of vocabulary in the training data
        """

        super(Model,self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.vocab_size = vocab_size

        # Initialise input embedding matrix
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        
        self.enc_blocks = nn.ModuleList([EncoderBlock(d_model, d_k, d_v, vocab_size, n_head) for _ in range(n_layers)])
        self.dec_blocks = nn.ModuleList([DecoderBlock(d_model, d_k, d_v, vocab_size, n_head) for _ in range(n_layers)])

        # model ouput linear layer
        self.model_out = nn.Linear(d_model, vocab_size)



    def _positional_encode(self, seq_len,d_model):
        """
        TODO: can this be cached while initialising the model, so don't need to calculate at every forward pass

        params:
        inp_len : number of tokens in the input
        d_model: dimension of hidden layer

        ouputs:
        pe : postional vector of size (d_model, seq_len)

        PE(pos, 2i) = sin( pos / 10000^(2*i/d_model) )
        PE(pos, 2i + 1) = cos( pos / 10000(2*i/d_model) )
        """
        pos_embs = torch.zeros(seq_len, d_model)
        for pos in torch.arange(seq_len):
            for i in torch.arange(0, d_model, 2):
                pos_embs[pos, i]     = torch.sin( pos / 10000** ((2*i)/d_model))
                pos_embs[pos, i + 1] = torch.cos( pos/ 10000** ((2*i)/d_model))
        return pos_embs



    def forward(self, inputs, outputs=None):
        



# d_model = 32
# d_k = d_v = 16
# n_head = 4
# seq_len = 3
# vocab_size = 200

# inputs = torch.randint(0,50,(10,5))
# trs = Model(d_model, d_k, d_v, vocab_size, n_head)
# out = trs(inputs)
# out.shape
# out.shape
# pos = torch.ones(2,5)
# torch.sin(pos)

# embs = torch.nn.Embedding(25,8)
# inputs = torch.zeros((10,1),dtype=torch.int)
# embs(inputs).shape

# pos_embs = torch.ones((seq_len,d_model))
# for pos in range(0,seq_len-1):
#     for i in range(0, d_model, 2):
#         pos_embs[pos, i]     = torch.sin( pos / 10000 ** ((2*i)/d_model))
#         pos_embs[pos, i + 1] = torch.cos( pos/ 10000** ((2*i)/d_model))

