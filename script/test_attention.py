import torch
import torch.nn as nn
import torch.nn.functional as F

"""
File per capire come funziona l attention.
"""

torch.manual_seed(1337) #imposto un seed
B, T, C = 1, 4, 8 #imposto dei parametri fittizi, B = Quante sequenze stai processando in parallelo, T = Quanti token ci sono in sequenza, C = vettore
x = torch.randn(B, T, C) #genero una matrice casuale

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

B, T, C = 4, 8, 32
x = torch.randn(B, T, C)
block = Block(n_embd=32, num_heads=4)
out = block(x)
print(out.shape)   # deve stampare torch.Size([4, 8, 32])