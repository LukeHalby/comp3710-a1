import torch as t
import numpy as np
import matplotlib.pyplot as plt

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

x = t.Tensor(X)
y = t.Tensor(Y)

x = x.to(device)
y = y.to(device)
# First question
#z = t.sin(2*(x + y))
# Second question
z = t.exp((-(x**2 + y**2))/2.0) * t.sin(2*(x + y))

plt.imshow(z.cpu().numpy())

plt.tight_layout()
plt.show()