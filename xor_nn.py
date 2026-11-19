import numpy as np
import matplotlib.pyplot as plt

# ---------- Dataset ----------
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

# ---------- Activations (derivative written in terms of the activation output) ----------
def sigmoid(z): return 1 / (1 + np.exp(-z))
acts = {
    "sigmoid": (sigmoid,     lambda a: a * (1 - a)),
    "tanh":    (np.tanh,     lambda a: 1 - a**2),
    "relu":    (lambda z: np.maximum(0, z), lambda a: (a > 0).astype(float)),
}

# ---------- Feedforward network: 2 -> hidden -> 1 ----------
class XORNet:
    def __init__(self, hidden=2, act="sigmoid", lr=0.5, seed=1):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1, (2, hidden)); self.b1 = np.zeros((1, hidden))
        self.W2 = rng.normal(0, 1, (hidden, 1)); self.b2 = np.zeros((1, 1))
        self.f, self.df = acts[act]
        self.lr = lr

    def forward(self, X):
        self.h = self.f(X @ self.W1 + self.b1)
        self.out = sigmoid(self.h @ self.W2 + self.b2)
        return self.out

    def train(self, X, y, epochs=10000):
        losses = []
        for _ in range(epochs):
            out = self.forward(X)
            losses.append(np.mean((y - out) ** 2))
            d_out = (out - y) * out * (1 - out)          # output-layer error
            d_h = (d_out @ self.W2.T) * self.df(self.h)  # hidden-layer error
            self.W2 -= self.lr * self.h.T @ d_out
            self.b2 -= self.lr * d_out.sum(0, keepdims=True)
            self.W1 -= self.lr * X.T @ d_h
            self.b1 -= self.lr * d_h.sum(0, keepdims=True)
        return losses

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

# ---------- Baseline: single-layer perceptron (fails on XOR) ----------
def perceptron(X, y, epochs=100, lr=0.1):
    w, b, acc = np.zeros(2), 0.0, []
    t = y.ravel()
    for _ in range(epochs):
        for xi, ti in zip(X, t):
            p = int(xi @ w + b > 0)
            w += lr * (ti - p) * xi
            b += lr * (ti - p)
        acc.append(np.mean([int(xi @ w + b > 0) == ti for xi, ti in zip(X, t)]))
    return acc

# ---------- Plot helpers ----------
def plot_boundary(net, title, ax):
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    Z = net.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=50, cmap="RdBu_r", alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="coolwarm", edgecolors="k", s=120)
    ax.set_title(title)

if __name__ == "__main__":
    # 1. Perceptron failure
    acc = perceptron(X, y)
    plt.figure(); plt.plot(acc); plt.ylim(0, 1.05)
    plt.title("Single-layer perceptron accuracy on XOR (never reaches 1.0)")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.savefig("perceptron_fail.png")

    # 2. Main model: 2-2-1, sigmoid
    net = XORNet(hidden=2, act="sigmoid", lr=0.5)
    losses = net.train(X, y, epochs=10000)
    print("Predictions:\n", net.forward(X).round(3))
    print("Classes:", net.predict(X).ravel())

    plt.figure(); plt.plot(losses)
    plt.title("Training loss (MSE) vs epoch"); plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.savefig("loss_curve.png")

    fig, ax = plt.subplots(); plot_boundary(net, "Decision boundary (2-2-1, sigmoid)", ax)
    fig.savefig("decision_boundary.png")

    # 3. Experiments: activation x hidden size
    print("\nActivation | Hidden | Final loss | Correct")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for a, lr, ax in zip(["sigmoid", "tanh", "relu"], [0.5, 0.5, 0.1], axes):
        for h in [2, 3, 4]:
            n = XORNet(hidden=h, act=a, lr=lr, seed=1)
            l = n.train(X, y, 10000)
            ok = (n.predict(X).ravel() == y.ravel()).all()
            print(f"{a:10s} | {h:6d} | {l[-1]:.5f}   | {ok}")
            ax.plot(l, label=f"{h} hidden")
        ax.set_title(a); ax.set_xlabel("Epoch"); ax.legend()
    fig.savefig("activation_comparison.png")

    # 4. Learning-rate experiment
    plt.figure()
    for lr in [0.05, 0.1, 0.5, 1.0]:
        l = XORNet(hidden=2, act="sigmoid", lr=lr).train(X, y, 10000)
        plt.plot(l, label=f"lr={lr}")
    plt.legend(); plt.title("Effect of learning rate"); plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.savefig("lr_comparison.png")
    plt.show()