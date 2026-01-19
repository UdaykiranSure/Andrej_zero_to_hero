import torch
import torch.nn as nn
import torch.nn.functional as F
import time


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
        B, seq_len, d_model = x.shape
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

        out = out.transpose(1,2).reshape(B , seq_len, self.d_v)        # (B, seq_len, d_v)
        out = self.proj(out)                                      # (B, seq_len, d_model)
        return out


class FFN(nn.Module):
    def __init__(self, d_model):
        super(FFN, self).__init__()
        self.d_model = d_model
        self.l1 = nn.Linear(d_model, 4*d_model)
        self.gelu = nn.GELU(approximate='tanh')
        self.l2 = nn.Linear(4*d_model, d_model)
        self.l2.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return x



class Block(nn.Module):
    def __init__(self, d_model, d_k, d_v, vocab_size, n_head):
        super(Block, self).__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CasualSelfAttention(d_model, d_k, d_v, n_head, masked= True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
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
    def __init__(self, d_model, d_q, d_k, d_v, vocab_size,  block_size, n_head, n_layers = 1 ):
        """
        d_modle: dimensional size of hidden representations
        d_k:  dimentional size of key and query projection matrix
        d_v: dimentional size of value projection matrix
        vocab_size: size of vocabulary in the training data
        """

        super(GPT ,self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_q = d_q
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layers = n_layers
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vocab_size, self.d_model),
            wpe = nn.Embedding(self.vocab_size, self.d_model),
            h = nn.ModuleList([Block(d_model, d_k, d_v, vocab_size, n_head) for _ in range(n_layers)]),
            ln_f = nn.LayerNorm(d_model)
        ))

        # model ouput linear layer
        self.lm_head = nn.Linear(d_model, vocab_size)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

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
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(inputs)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layers=12, n_head=12, d_model=768),  # 124M params
            'gpt2-medium':  dict(n_layers=24, n_head=16, d_model=1024), # 350M params
            'gpt2-large':   dict(n_layers=36, n_head=20, d_model=1280), # 774M params
            'gpt2-xl':      dict(n_layers=48, n_head=25, d_model=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['d_q'] = config_args['d_k'] = config_args['d_v'] = config_args['d_model']

        # create a from-scratch initialized minGPT model
        # config = GPTConfig(**config_args)

        model = GPT(**config_args)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model







class DataLoader:
    def __init__(self, B, T, text,device):
        self.B = B
        self.T = T
        self.text = torch.tensor(text)
        self.device = device
        # self.process_rank = process_rank
        # self.num_processes = num_processes
        # assert split in {'train', 'val'}

        # # get the shard filenames
        # data_root = "edu_fineweb10B"
        # shards = os.listdir(data_root)
        # shards = [s for s in shards if split in s]
        # shards = sorted(shards)
        # shards = [os.path.join(data_root, s) for s in shards]
        # self.shards = shards
        # assert len(shards) > 0, f"no shards found for split {split}"
        # if master_process:
        #     print(f"found {len(shards)} shards for split {split}")
        # self.reset()
        self.current_position = 0

    # def reset(self):
    #     # state, init at shard zero
    #     self.current_shard = 0
    #     self.tokens = load_tokens(self.shards[self.current_shard])
    #     self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.text[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        self.current_position += B * T * self.num_processes
        # advance the position in the tensor
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position > len(self.tokens):
            # self.current_shard = (self.current_shard + 1) % len(self.shards)
            # self.tokens = load_tokens(self.shards[self.current_shard])
            # self.current_position = B * T * self.process_rank
            self.current_position = 0
        return x, y





max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)








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

