import torch
import torch.nn as nn
import torch.nn.functional as F


class CasualSelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, masked = False):

        assert d_k%n_head == 0, "attention key matrix dimenstion should be multiple of num of heads"
        assert d_v%n_head == 0, "attention value matrix dimenstion should be multiple of num of heads"

        super(CasualSelfAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head

        self.query = nn.Linear(d_model,d_k, bias = False)
        self.key = nn.Linear(d_model,d_k, bias = False)
        self.value = nn.Linear(d_model,d_v, bias = False)

        self.proj = nn.Linear(d_v, d_model, bias= False)
        self.proj.NANOGPT_SCALE_INIT = 1

        # if masked:
        #     self.register_buffer('mask', torch.tril(torch.ones(seq_len,seq_len)).view(1,1,seq_len,seq_len))

    def forward(self, x):
        B, seq_len, d_model = q.shape
        q = self.query(x).view(B,self.n_head, seq_len,self.d_k//self.n_head)       #(B, n_heads, seq_len, d_k//n_heads)
        k = self.key(x).view(B,self.n_head, seq_len,self.d_k//self.n_head)         #(B, n_heads, seq_len, d_k//n_heads)
        v = self.value(x).view(B,self.n_head, seq_len,self.d_v//self.n_head)       #(B, n_heads, seq_len, d_v//n_heads)

        # Scratch implementaion
        # wei = (q @ (k.transpose(-1, -2)))                         #(B, n_heads, seq_len, seq_len)
        # if self.masked:
        #     mask = torch.tril(torch.ones(seq_len, seq_len)).view(1,1,seq_len,seq_len)
        #     wei = torch.masked_fill(wei,mask == 0,-torch.inf)
        # att = F.softmax(wei,3) @ v                              # (B, n_heads, seq_len, d_v//n_heads)

        # Pytorch's Flash attention
        out = F.scaled_dot_product_attention(q,k,v, is_causal=True)

        out = out.transpose(1,2).reshape(B , seq_len, d_v)        # (B, seq_len, d_v)
        out = self.proj(out)                                      # (B, seq_len, d_model)
        return out


class FFN(nn.Module):
    def __init__(self, d_model):
        super(FFN, self).__init__()
        self.d_model = d_model
        self.l1 = nn.Linear(d_model, 4*d_model)
        self.gelu = nn.GELU(approximate='tanh')
        self.l2 = nn.Linear(d_model, 4*d_model)
        self.l2.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return x



class Block(nn.Module):
    def __init__(self, d_model, d_k, d_v, vocab_size, n_head):
        super(DecoderBlock, self).__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CasualSelfAttention(d_model, d_k, d_v, n_head, masked= True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)

    def forward(self, x, enc_out):
        """
        params: 
        x: positional encoded output token sequence. #(seq_len, d_model)
        enc_out: encoder output hidden represenations of input sequence

        outputs:
        out: encoder output representation of input sequence  #(seq_len, d_model)
        """

        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    d_model: int = 768 # embedding dimension
    d_q: int = 768 # embedding dimension
    d_k: int = 768 # embedding dimension
    d_v: int = 768 # embedding dimension
    

class GPT(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, vocab_size, n_head, block_size, n_layers = 1 ):
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
        self.d_q = d_q
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layers = n_layers
        
        self.transfomer = nn.ModuelDict(dict(
            wte = nn.Embedding(self.vocab_size, self.d_model),
            wpe = nn.Embedding(self.vocab_size, self.d_model),
            h = nn.ModuleList([Block(d_model, d_k, d_v, vocab_size, n_head) for _ in range(n_layers)]),
            ln_f = nn.LayerNorm(d_model)
        ))

        # model ouput linear layer
        self.lm_head = nn.Linear(d_model, vocab_size)

        # weight sharing scheme
        self.transfomer.wte.weight = self.lm_head.weight

        #initialise parameters
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.n_layers)** -0.5 
            torch.nn.init.normal_(module.weight ,mean=0.0, std = std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, inputs, targets= None):
        B, seq_len = inputs.shape

        assert seq_len == self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"

        pos = torch.arange(0, seq_len, dtype=torch.long, device = inputs.device)
        pos_emb = self.transfomer.wpe(pos)
        tok_emb = self.transfomer.wte(inputs)
        x = tok_emb + pos_emb
        for block in self.transfomer.h:
            x = block(x)
        x = self.transfomer.ln_f(x)

        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss





class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y










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

