import torch as t
import numpy as np
import matplotlib.pyplot as plt

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

# First question, zoomed in mandelbrot
# Y, X = np.mgrid[-0.2:0.2:0.0005, -2:-1.5:0.0005]
# Second question, Julia set
Y, X = np.mgrid[-0.2:0.2:0.0005, -0.5:0.5:0.0005]

x = t.Tensor(X)
y = t.Tensor(Y)
z = t.complex(x, y)
zs = z.clone()
ns = t.zeros_like(z)

z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)

# First question
# for i in range(200):
#     zs_ = zs*zs + z
#     not_diverged = t.abs(zs_) < 4.0
#     ns += not_diverged
#     zs = zs_

# Second question
for i in range(200):
    zs_ = zs*zs + t.complex(t.Tensor([-0.162]), t.Tensor([1.04]))
    not_diverged = t.abs(zs_) < 4.0
    ns += not_diverged
    zs = zs_

fig = plt.figure(figsize=(16, 10))


def processFractal(a):
    """Display an array of iteration counts as a
    colorful picture of a fractal."""
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
                          30 + 50*np.sin(a_cyclic),
                          155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a


plt.imshow(processFractal(ns.cpu().numpy()))
plt.tight_layout(pad=0)
plt.show()
