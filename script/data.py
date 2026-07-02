import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

block_size = 8
batch_size = 32
T = block_size
n_embd = 32
C = n_embd
n_layer = 4
num_heads = 4
max_new_tokens = 200
Base_path = Path(__file__).parent.parent #/nanogpt
data_path = Base_path / "data" / "input.txt"
text = Path(data_path).read_text()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {s: i for i,s in enumerate(chars)}
itos = {i: s for s,i in stoi.items()}
def encode(a: str) -> list[int]:
    b = []
    for t in a:
        c = stoi[t]
        b.append(c)
    return b

def decode(a: list[int]) -> str:
    b = []
    for t in a:
        c = itos[t]
        b.append(c)
    b = ''.join(b)
    return b

text_converted = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(text_converted))
val_data = text_converted[n:]
train_data = text_converted[:n]

def get_batch(split:str):
    data_batch = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data_batch) - block_size, (batch_size,))
    x = torch.stack([data_batch[i : i + block_size] for i in ix])
    y = torch.stack([data_batch[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


#creazione della classe dell attention
class Attention(nn.Module):

    def __init__(self, head_size):
        """Costruttore statico che inizializza le risorse"""
        super().__init__() #chiamata al costruttore della classe madre
        self.head_size = head_size
        self.query = nn.Linear(C, head_size, bias=False) 
        self.key = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False) #definisco le operazioni lineari, con numeri casuali all inizializzazione, queste non hanno bias dato che sono degli scalari
        self.register_buffer('tril', torch.tril(torch.ones(T, T))) #registro il buffer, questo crea una matrice di uni

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #chiamo le operazioni lineari
        q = self.query(x)
        v = self.value(x)
        score = q@k.transpose(-2, -1) #ottengo una matrice tokenxtoken
        scores = score * (self.head_size ** -0.5) #ammorbidisco i numeri
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #metto tutti i numeri della diagonale superiori a 0
        wei = F.softmax(scores, dim=-1) #applico il softmax
        out = wei @ v #output
        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Attention(head_size) for _ in range(num_heads)]) #il modulo contiene una lista di moduli
        self.proj = nn.Linear(head_size*num_heads, C)

    def forward(self, x):
        out = [h(x) for h in self.heads] #chiamo ogni testa e gli faccio elaborare x
        out = torch.cat(out, dim=2) #concateno in un vettore i risultati delle teste
        out = self.proj(out) #risultato finale
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C, 4*C),
            nn.ReLU(),
            nn.Linear(4*C, C)
        )

    def forward(self, x):
        out = self.net(x)
        return out

class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.mha = MultiHeadAttention(num_heads, head_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, n_layer, C):
        super().__init__()
        self.n_layer = n_layer
        self.token_embedding_table = nn.Embedding(vocab_size, C)
        self.position_embedding_table = nn.Embedding(block_size, C)
        self.blocks = nn.Sequential(*[Block(C, num_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, loss = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

model = GPT(n_layer, C)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(15000):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(step, loss.item())

context = torch.zeros((1, 1), dtype=torch.long)
out = generate(model, context, max_new_tokens)
print(decode(out[0].tolist()))