# import random
# from collections import namedtuple

# Point = namedtuple("Point", ('x', 'y'))
# points = []
# for i in range(5):
#     points.append(Point(i, i + 1))

# for i in zip(*points):
#     print(i)

# l = Point((0, 1, 2, 3, 4), (1, 2, 3, 4, 5))
# print(l)

# s = None
# f = lambda x: x is not None
# print(f(s))

# import torch

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())

for i in reversed(range(3)):
    print(i)