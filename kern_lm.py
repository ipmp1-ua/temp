import torch
import torch.nn as nn
from torch.nn import functional as F
import re, sys, math
import random
import os
from music21 import *
import converter21

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

n_embd = 256
ff_dim = n_embd
n_head = 4
n_layer = 8
dropout = 0.1
try_correct = True
stoi_path = '/workspace/stoi.txt'

temperature = 0.75   
p = 0.9
top_p_sampling = True

block_size = 256 # what is the maximum context length for predictions?

max_seq_len = 1512 # maximum sequence length for training

device = 'cuda' if torch.cuda.is_available() else 'cpu'



special_tokens = {
    '<t>': '\t',
    '<n>': '\n',
    '<end>': '\n\n================================\n\n', 
}


# Load chars and vocab mapping only if we're training or they don't already exist
chars = None
stoi = None
itos = None

def load_vocab_from_file(file_path='kern.txt'):
    global chars, stoi, itos, vocab_size
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(kern_tokenizer(text))))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, stoi, itos, vocab_size

def load_vocab_from_checkpoint(checkpoint_path):
    global chars, stoi, itos, vocab_size
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'vocab' in checkpoint:
        # If vocab is stored in the checkpoint
        chars = checkpoint['vocab']['chars']
        stoi = checkpoint['vocab']['stoi']
        itos = checkpoint['vocab']['itos']
        vocab_size = len(chars)
    else:
        # If using older checkpoint without vocab, attempt to load from a vocab file
        if os.path.exists(stoi_path):
            # Load vocabulary mapping from stoi.txt if available
            chars, stoi, itos, vocab_size = load_vocab_from_stoi(stoi_path)
        else:
            raise ValueError("Checkpoint doesn't contain vocabulary and stoi.txt not found")
    return chars, stoi, itos, vocab_size

def load_vocab_from_stoi(stoi_path=stoi_path):
    global chars, stoi, itos, vocab_size
    with open(stoi_path, 'r', encoding='utf-8') as f:
        stoi_content = f.read()
    
    # Parse the stoi.txt file - format is: token index
    stoi = {}
    for line in stoi_content.strip().split('\n'):
        if line:
            parts = line.split(' ')
            if len(parts) >= 2:
                token = parts[0]
                idx = int(parts[1])
                stoi[token] = idx
    
    chars = sorted(stoi.keys(), key=lambda x: stoi[x])
    itos = {v: k for k, v in stoi.items()}
    vocab_size = len(chars)
    return chars, stoi, itos, vocab_size

def kern_tokenizer(text):
    tokens = re.findall(r'<t>|<n>|<end>|[^<]+', text)
    return tokens

def initialize_vocab():
    global chars, stoi, itos, vocab_size
    if chars is None:
        # Try to load from stoi.txt first
        if os.path.exists(stoi_path):
            chars, stoi, itos, vocab_size = load_vocab_from_stoi(stoi_path)
        # If still not loaded and kern.txt exists, load from there
        elif os.path.exists('kern.txt'):
            chars, stoi, itos, vocab_size = load_vocab_from_file('kern.txt')
        else:
            raise ValueError("No vocabulary source found (stoi.txt or kern.txt)")
    return chars, stoi, itos, vocab_size

# Initialize if we have data files available
try:
    initialize_vocab()
except Exception as e:
    print(f"Vocabulary initialization deferred: {e}")
    # We'll initialize vocab when loading a checkpoint

def encode(s):
    # Ensure vocabulary is loaded before encoding
    if stoi is None:
        initialize_vocab()
    # Convert a string to a list of integers using the stoi mapping
    return [stoi[c] for c in s]

def decode(l):
    # Ensure vocabulary is loaded before decoding
    if itos is None:
        initialize_vocab()
    # Convert a list of integers to a string using the itos mapping
    return ''.join([itos[i] for i in l])

converter21.register()

class RelativeHead(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.register_buffer(
            'rel_pos_indices',
            torch.arange(block_size).unsqueeze(0) - torch.arange(block_size).unsqueeze(1)
        )
        self.rel_pos_embedding = nn.Parameter(torch.zeros(2 * block_size - 1, head_size))
        nn.init.normal_(self.rel_pos_embedding, std=0.02)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        
        content_scores = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)  # (B, T, T)
        
        pos_indices = self.rel_pos_indices[:T, :T].unsqueeze(0)  # (1, T, T)
        pos_indices_shifted = pos_indices + (block_size - 1)
        rel_pos_emb = self.rel_pos_embedding[pos_indices_shifted]  # (1, T, T, head_size)
        
        position_scores = (q.unsqueeze(2) * rel_pos_emb).sum(dim=-1)  # (B, T, T)
        
        wei = content_scores + position_scores
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = wei @ v  # (B, T, head_size)
        return out


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([RelativeHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Concatenate outputs from all heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Final projection and dropout
        out = self.proj(out)
        out = self.dropout(out)
        return out    


class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        
        wei = q @ k.transpose(-2, -1) * C **-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, ff_dim), 
            nn.ReLU(),
            nn.Linear(ff_dim, n_embd), 
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class KernLM(nn.Module):
    @classmethod
    def from_checkpoint(cls, checkpoint_path):
        global vocab_size
        # Try to load vocab from checkpoint first
        try:
            load_vocab_from_checkpoint(checkpoint_path)
        except:
            # If that fails, ensure we have vocab initialized
            initialize_vocab()
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model = cls()
        model.load_state_dict(state_dict)
        return model

    def __init__(self):
        global stoi, itos, vocab_size
        self.w2i = stoi
        self.i2w = itos
        self.vocab_size = vocab_size
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #self.position_embedding_table = nn.Embedding(block_size,n_embd)


        positional_encoding = self._get_sinusoidal_positional_encoding(max_seq_len, n_embd)
        self.register_buffer('positional_encoding', positional_encoding)

        self.blocks = nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd,vocab_size)
        

    def _get_sinusoidal_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  
    
    def forward(self, idx, targets=None):
        B,T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        #pos_emb = self.position_embedding_table(torch.arange(T,device=device)) # (T,C)
        pos_emb = self.positional_encoding[:T].unsqueeze(0)
        x = tok_emb+pos_emb
        x = self.blocks(x)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, out_file_path, top_p_sampling=True, temperature=1.0):
        initial_context = idx.clone()
        out_dir = 'output/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        i = 0
        get_filename = lambda i: f"{out_dir}{out_file_path}_{i}.krn"

        out_file = open(get_filename(i), 'w', encoding='utf-8')
        # idx is (B, T) array of indices in the current context
        self.eval()
        decoded = decode(idx[0].tolist())
        for token, replacement in special_tokens.items():
            decoded = decoded.replace(token, replacement)
        out_file.write(decoded)
        out_file.flush()

        for _ in range(max_new_tokens):
            # crop idx
            idx_cond = idx[:,-block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            if top_p_sampling:
                logits = logits / temperature
        
                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                
                # Calculate cumulative probability
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Create mask for tokens within the top-p probability mass
                mask = cumulative_probs <= p
                
                # Always keep at least the top token
                mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], dim=-1)
                
                # Get cutoff index - the last True value in the mask
                cutoff_idx = torch.sum(mask, dim=-1, keepdim=True)
                
                # Create a new mask of the proper shape
                mask_final = torch.zeros_like(probs)
                
                # Place the sorted and filtered probabilities back in their original positions
                for i in range(probs.size(0)):  # For each item in the batch
                    # Get the number of tokens to keep for this sample
                    num_to_keep = cutoff_idx[i].item()
                    # Get the indices of tokens to keep
                    indices_to_keep = sorted_indices[i, :num_to_keep]
                    # Put the filtered probabilities back in their original positions
                    mask_final[i, indices_to_keep] = probs[i, indices_to_keep]
                
                # Renormalize
                filtered_probs = mask_final / mask_final.sum(dim=-1, keepdim=True)
            
            else:
                filtered_probs = F.softmax(logits / temperature, dim=-1)
            # Sample from the filtered distribution
            idx_next = torch.multinomial(filtered_probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

            output = decode([idx[0, -1].item()])
            if output in special_tokens:
                output = special_tokens[output]
            if output == special_tokens['<end>']:
                idx = initial_context.clone().to(device)
                out_file.close()
                if try_correct:

                    s = converter.parse(get_filename(i), format='humdrum')
                    if s.isWellFormedNotation():
                        sc = omr.correctors.ScoreCorrector(s)
                        s2 = sc.run()
                        s2.write('humdrum', fp=get_filename(i))
                        print("File written to", get_filename(i))
                    else:
                        print("File", get_filename(i), "is not well formed")
                        



                i+= 1
                out_file = open(get_filename(i), 'w', encoding='utf-8')
                decoded = decode(idx[0].tolist())
                for token, replacement in special_tokens.items():
                    decoded = decoded.replace(token, replacement)
                out_file.write(decoded)
                out_file.flush()
                continue

            out_file.write(output)
            out_file.flush()  # Flush after each token
        return idx

