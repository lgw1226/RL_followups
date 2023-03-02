import torch

x = torch.arange(0, 9, 1).view(-1, 3)
y = torch.normal(0, 1, size=x.size())

batch = (x, y)
a, b, c = [*zip(*batch)]
print(a, b, c)