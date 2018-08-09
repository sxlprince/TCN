import numpy as np
import torch

# a = [600] * 3 + [300]
#
# print (a)
# i = 0
# a = 4 if i == 0 else 3
# print (a)

# a = np.array([[[2,3,4],
#      [1,3,5]],
#      [[4,5,6],
#       [4,7,9]]])
# b = a.transpose(1,2)
#
# print (b)
#
# print (a[:, :, :-1])

# a = np.array([2,3,4,5])
#
# print (a.size(0))

# a = np.array([1,2,3,4,5,6])
# b = a.reshape(2,3)
# print (b)

# a = np.array([1,2,3])
# b = np.array([1,2,3])
# print (a.shape)
# c = np.dot(a,b)
# print (c)

# a = 'e m skfd , fs .'
# b = a.split()
# print (b)

ids = torch.LongTensor(6)
print (ids)

sxl = ids.view(2,-1)
print (sxl)

