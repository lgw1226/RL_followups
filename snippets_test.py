import rlutil
import torch

x = torch.arange(10)
y = torch.rand(10)

tuple_of_tensors = (x, y)
print(tuple_of_tensors)

ret = rlutil.tuple_of_tensors_to_tensor(tuple_of_tensors)
print(ret)