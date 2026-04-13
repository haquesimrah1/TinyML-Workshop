# TinyML-Workshop

In this workshop, we'll demo TinyML concepts by using python to approximate a sine wave, which will then be used to control an LED. The end result will be an LED that appears to be "breathing".
Due to the hardware resistrictions on the Arduino Nano, we will not use a TensorFlow library. However, we will still use TinyML conceptual ideas, such as supervised regression via a feedforward nerual network (which we demo through the sine function)

# CODE FOR GENERATING AND TRAINING DATA ON GOOGLE COLAB

1) Open new file on google colab
2) Copy and paste code into the google colab file:

Code block 1:
```
import numpy as np
import matplotlib.pyplot as plt  
```
Code block 2:
```
np.random.seed(42)
N = 1000

x_raw = np.linspace(0, 2 * np.pi, N)
y_raw = np.sin(x_raw) + np.random.normal(0, 0.05, N)

X_MIN = float(x_raw.min())
X_MAX = float(x_raw.max())

x = ((x_raw - X_MIN) / (X_MAX - X_MIN)).reshape(-1, 1)  # (1000, 1)
y = y_raw.reshape(-1, 1)

plt.plot(x_raw, y_raw, alpha=0.4, label="noisy data")
plt.plot(x_raw, np.sin(x_raw), label="true sin")
plt.legend(); plt.title("Training data"); plt.show()
```
Code block 3:
```
def relu(z):
    return np.maximum(0, z)

def forward(x, params):
    W1, b1, W2, b2, W3, b3 = params
    h1  = relu(x @ W1 + b1)
    h2  = relu(h1 @ W2 + b2)
    out = h2 @ W3 + b3
    return out, h1, h2

def mse(pred, target):
    return np.mean((pred - target) ** 2)

def step(x, y, params, lr):
    W1, b1, W2, b2, W3, b3 = params
    out, h1, h2 = forward(x, params)
    N = x.shape[0]

    dout = 2 * (out - y) / N

    dW3 = h2.T @ dout;        db3 = dout.sum(axis=0)
    dh2 = (dout @ W3.T) * (h2 > 0)

    dW2 = h1.T @ dh2;         db2 = dh2.sum(axis=0)
    dh1 = (dh2 @ W2.T) * (h1 > 0)

    dW1 = x.T @ dh1;          db1 = dh1.sum(axis=0)

    W1 -= lr*dW1;  b1 -= lr*db1
    W2 -= lr*dW2;  b2 -= lr*db2
    W3 -= lr*dW3;  b3 -= lr*db3

    return [W1, b1, W2, b2, W3, b3]  
```
Code block 4:
```
HIDDEN = 16
LR     = 0.01
EPOCHS = 3000

def init(in_sz, h_sz, out_sz):
    W1 = np.random.randn(in_sz, h_sz) * np.sqrt(2.0 / in_sz)
    b1 = np.zeros((1, h_sz))
    W2 = np.random.randn(h_sz, h_sz)  * np.sqrt(2.0 / h_sz)
    b2 = np.zeros((1, h_sz))
    W3 = np.random.randn(h_sz,     1) * np.sqrt(2.0 / h_sz)
    b3 = np.zeros((1, 1))
    return [W1, b1, W2, b2, W3, b3]

params = init(1, HIDDEN, 1)
losses = []

for epoch in range(EPOCHS + 1):
    params = step(x, y, params, LR)
    if epoch % 200 == 0:
        loss = mse(forward(x, params)[0], y)
        losses.append((epoch, loss))
        print(f"epoch {epoch:4d}   loss {loss:.6f}")

plt.plot([e for e,_ in losses], [l for _,l in losses])
plt.xlabel("epoch"); plt.ylabel("MSE loss")
plt.title("Training loss"); plt.show()
```
Code block 5:
```
pred, _, _ = forward(x, params)

plt.plot(x_raw, y_raw,         alpha=0.3, label="noisy data")
plt.plot(x_raw, np.sin(x_raw), label="true sin",       linewidth=2)
plt.plot(x_raw, pred,          label="network output",  linewidth=2, linestyle="--")
plt.legend(); plt.title("Model fit"); plt.show() 
```
Code block 6:
```
from google.colab import files

W1, b1, W2, b2, W3, b3 = params

def c_array(name, arr):
    rows, cols = arr.shape
    vals = ", ".join(f"{v:.8f}f" for v in arr.flatten())
    return (
        f"// {name}: {rows}x{cols}\n"
        f"const float {name}[] = {{{vals}}};\n"
        f"const int {name}_rows = {rows}, {name}_cols = {cols};\n\n"
    )

header = "#pragma once\n\n"
header += f"// Input normalization — must match Python exactly\n"
header += f"const float X_MIN = {X_MIN:.8f}f;\n"
header += f"const float X_MAX = {X_MAX:.8f}f;\n\n"
header += c_array("W1", W1)
header += c_array("b1", b1)
header += c_array("W2", W2)
header += c_array("b2", b2)
header += c_array("W3", W3)
header += c_array("b3", b3)

with open("weights.h", "w") as f:
    f.write(header)

print("Preview of weights.h:")
print(header[:500], "...")

files.download("weights.h")   # triggers browser download immediately
```



