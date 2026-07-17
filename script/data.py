import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

batch_size = 32 #quante sequenze processi in paralleloa ogni step, se n alto rende il gradiente meno rumoroso
block_size = 64  #è la lunghezza del contesto, il modello vede fino a n token per prevedere il n+1,
n_embd = 128     #è la larghezza della rete
n_layer = 6     #è la profondita della rete
num_heads = 4   #numero di teste
head_size = n_embd // num_heads #ogni testa lavora in un sottospazio di x dimensioni
max_new_tokens = 200    #numero di token da generare nella risposta finale
dropout = 0.3 #30%
device = 'cuda' if torch.cuda.is_available() else 'cpu' #device = gpu senno cpu
print(f"Device = {device}")
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
    x, y = x.to(device), y.to(device)
    return x, y


#creazione della classe dell attention
class Attention(nn.Module):

    def __init__(self, head_size):
        """Costruttore statico che inizializza le risorse"""
        super().__init__() #chiamata al costruttore della classe madre
        self.head_size = head_size
        self.dropout = nn.Dropout(dropout)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False) #definisco le operazioni lineari, con numeri casuali all inizializzazione, queste non hanno bias dato che sono degli scalari
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #registro il buffer, questo crea una matrice di uni

    def forward(self, x):
        _, block_size, _ = x.shape
        k = self.key(x) #chiamo le operazioni lineari
        q = self.query(x)
        v = self.value(x)
        score = q@k.transpose(-2, -1) #ottengo una matrice tokenxtoken
        scores = score * (self.head_size ** -0.5) #ammorbidisco i numeri
        scores = scores.masked_fill(self.tril[:block_size, :block_size] == 0, float('-inf')) #metto tutti i numeri della diagonale superiori a 0
        wei = F.softmax(scores, dim=-1) #applico il softmax
        wei = self.dropout(wei)
        out = wei @ v #output
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Attention(head_size) for _ in range(num_heads)]) #il modulo contiene una lista di moduli
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = [h(x) for h in self.heads] #chiamo ogni testa e gli faccio elaborare x
        out = torch.cat(out, dim=2) #l output è la somma dei contributi delle singole teste ognuna proiettata dalla sua fetta indipendente di proj
        out = self.proj(out) #risultato finale
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
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
    def __init__(self, n_layer, n_embd):
        super().__init__()
        self.n_layer = n_layer
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        batch_size, block_size = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(block_size, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            batch_size, block_size, vocab_size = logits.shape
            logits = logits.view(batch_size*block_size, vocab_size)
            targets = targets.view(batch_size*block_size)
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

model = GPT(n_layer, n_embd)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

eval_iters = 200   # su quanti batch mediare la misura

@torch.no_grad()                     #dice a PyTorch "non costruire il grafo dei gradienti qui dentro". Stai misurando, non imparando.
def estimate_loss(model):
    out = {}
    model.eval()    #mette il modello in modalita valutazione
    for split in ['train', 'val']:
        loss_tot = torch.zeros(eval_iters)    #crea un tensore di zeri che accumula le loss
        for k in range(eval_iters):
            x, y = get_batch(split)# BUCO 3: pesca un batch dallo split corrente
            logits, loss = model(x,y)# BUCO 4: passa il batch nel modello e ottieni logits e loss
            loss_tot[k] = loss.item()# BUCO 5: salva la loss di questo batch nella posizione k del tensore
        out[split] = loss_tot.mean()# BUCO 6: fai la media del tensore e mettila in out[split]
    model.train()# BUCO 7: rimetti il modello in modalità training  ← il più infido, non scordarlo
    return out

for step in range(15000):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        losses = estimate_loss(model)
        print(step, losses['train'], losses['val'])

context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = generate(model, context, max_new_tokens)
print(decode(out[0].tolist()))
