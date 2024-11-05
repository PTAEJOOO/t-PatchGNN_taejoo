import torch
import torch.nn as nn
import torch.nn.functional as F

# a = torch.randn([4,6,3])

# b = torch.randn([4,4,3])

# print(a)

# print(b)

# target_mask = torch.ones_like(a).to(torch.int64)
# mask = torch.ones_like(b).to(torch.int64)

# for r,m in zip(target_mask, mask):
#     print(m[r])
#     break

# a = torch.randn([2,3,4,2])
# b = a.view(2,12,2)
# print(a)
# print(b)

t = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
ind = torch.tensor([[0, 0, 0, 1, 2], [1, 1, 1, 2, 0], [2, 2, 2, 0, 1]])
k = torch.gather(t, 1, ind)

print(t.shape, ind.shape, k.shape)
print(k)