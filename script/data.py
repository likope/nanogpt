import torch
from pathlib import Path

block_size = 8
batch_size = 4
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

xb, yb = get_batch("train")
print(xb[0])
print(yb[0])