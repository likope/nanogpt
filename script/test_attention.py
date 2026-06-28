import torch
import torch.nn as nn

torch.manual_seed(1337)
B, T, C = 1, 4, 8
x = torch.randn(B, T, C)
head_size = 4
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)
q = query(x)
v = value(x)
print(x.shape, k.shape, q.shape, v.shape)