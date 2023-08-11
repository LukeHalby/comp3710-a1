import torch as t
import numpy as np
import matplotlib.pyplot as plt

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

C = t.Tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
C = C.to(device)
product = C
for i in range(6):
    product = t.kron(product, C)

plt.imshow(product.cpu().numpy())
plt.tight_layout(pad=0)
plt.show()
