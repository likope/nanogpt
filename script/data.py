import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

block_size = 8
batch_size = 32
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

token_embedding_table = nn.Embedding(vocab_size, vocab_size)
xb, yb = get_batch("train")
logits = token_embedding_table(xb)          # (4, 8, 65)
B, T, C = logits.shape                       # 4, 8, 65
loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    def forward(self, xb, targets=None):
        logits = self.token_embedding_table(xb)
        if targets is None:
            return logits
        else:
            B, T, C = logits.shape                       # 4, 8, 65
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
            return logits, loss
    def generate(self, xb, max_new_tokens):
        for i in range(max_new_tokens):
            logits = self(xb)
            s = logits[:, -1, :]
            probs = torch.softmax(s, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            xb = torch.cat((xb, ix), dim=1)
        return xb

model = BigramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

n_steps = 10000

for step in range (n_steps):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(step, loss.item())
    
# start with a single token (the newline char, id 0) in a (1,1) tensor
context = torch.zeros((1, 1), dtype=torch.long)

# generate 300 new characters
out = model.generate(context, max_new_tokens=300)

# out is (1, 301) ids — pull row 0, turn to a list, decode to text
print(decode(out[0].tolist()))