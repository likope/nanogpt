import torch
import torch.nn as nn
import torch.nn.functional as F

"""
File per capire come funziona l attention.
"""

torch.manual_seed(1337) #imposto un seed
B, T, C = 1, 4, 8 #imposto dei parametri fittizi
x = torch.randn(B, T, C) #genero una matrice casuale

#creazione della classe dell attention
class Attention(nn.Module):

    def __init__(self, head_size):
        """Costruttore non dinamico che inizializza le risorse"""
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
    
h = Attention(head_size=4)
out = h(x)
print(out.shape)                                   # (1, 4, 4)
print('tril' in dict(h.named_buffers()))           # True
print('tril' in dict(h.named_parameters()))        # False