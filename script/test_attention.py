import torch
import torch.nn as nn
import torch.nn.functional as F

"""
File per capire come funziona l attention.
"""

torch.manual_seed(1337) #imposto un seed
B, T, C = 1, 4, 8 #imposto dei parametri fittizi
x = torch.randn(B, T, C) #genero una matrice casuale
head_size = 4 #parametro
key = nn.Linear(C, head_size, bias=False) #eseguo un operazione lineare ma senza bias, dato che voglio solo gli scalari
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x) #chiamo le operazioni lineari
q = query(x)
v = value(x)

score = q@k.transpose(-2, -1) #ottengo una matrice tokenxtoken
scores = score * (head_size ** -0.5)

A = torch.ones(1, 4, 4)
A = torch.tril(A)

scores = scores.masked_fill(A == 0, float('-inf'))
wei = F.softmax(scores, dim=-1)
out = wei @ v

print(out.shape)
print(wei[0].sum(dim=-1))
print(wei[0])