import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import cg


# ---------- tvoje pomoćne funkcije ----------
def downscale_2x2(X):
    return (
        X.reshape(-1, 28, 28)
         .reshape(-1, 14, 2, 14, 2)
         .mean(axis=(2, 4))
         .reshape(-1, 196)
    )


# ---------- tvoj kernelbasedMANDyClassifier ----------
class kernelbasedMANDyClassifier:
    def __init__(self, alpha=0.59, reg=1e-4, dtype=np.float32):
        self.alpha = alpha
        self.reg = reg
        self.dtype = dtype
        self.X_train = None
        self.Z = None

    def compute_gram_blockwise(self, X, block_size=1000):
        m, d = X.shape
        G = np.zeros((m, m), dtype=self.dtype)

        for i in range(0, m, block_size):
            i_end = min(i + block_size, m)
            Xi = X[i:i_end]

            for j in range(0, m, block_size):
                j_end = min(j + block_size, m)
                Xj = X[j:j_end]

                diff = Xi[:, None, :] - Xj[None, :, :]
                K_block = np.prod(np.cos(self.alpha * diff), axis=2)

                G[i:i_end, j:j_end] = K_block

        return G

    def fit(self, X, y):
        X = X.astype(self.dtype)
        self.X_train = X

        m = X.shape[0]
        num_classes = np.max(y) + 1

        Y = np.zeros((m, num_classes), dtype=self.dtype)
        Y[np.arange(m), y] = 1

        print(f"Computing Gram matrix (m={m})...")
        G = self.compute_gram_blockwise(X, block_size=1000)
        G += self.reg * np.eye(m, dtype=self.dtype)

        print("Solving system...")
        self.Z = np.zeros((m, num_classes), dtype=self.dtype)

        for c in range(num_classes):
            z, info = cg(G, Y[:, c], maxiter=200)
            self.Z[:, c] = z

    def predict(self, X, block_size=500):
        X = X.astype(self.dtype)

        n_test = X.shape[0]
        num_classes = self.Z.shape[1]

        scores = np.zeros((n_test, num_classes), dtype=self.dtype)

        for i in range(0, n_test, block_size):
            i_end = min(i + block_size, n_test)
            Xi = X[i:i_end]

            diff = Xi[:, None, :] - self.X_train[None, :, :]
            K_block = np.prod(np.cos(self.alpha * diff), axis=2)

            scores[i:i_end] = K_block @ self.Z

        return np.argmax(scores, axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ---------- tvoj ARRClassifier (ostavljen kakav je) ----------
class ARRClassifier:
    def __init__(self, d=196, n_basis=2, tt_rank=5, reg=1e-4, n_sweeps=5):
        self.d = d
        self.n_basis = n_basis
        self.r = tt_rank
        self.reg = reg
        self.n_sweeps = n_sweeps
        self.models = []

    def feature_map(self, X, alpha=0.59):
        return np.stack([np.cos(alpha * X), np.sin(alpha * X)], axis=2)

    def init_tt(self):
        cores = []
        cores.append(np.random.randn(1, self.n_basis, self.r))
        for _ in range(self.d - 2):
            cores.append(np.random.randn(self.r, self.n_basis, self.r))
        cores.append(np.random.randn(self.r, self.n_basis, 1))
        return cores

    def right_orthogonalize(self, cores):
        for mu in reversed(range(1, self.d)):
            G = cores[mu]
            r1, n, r2 = G.shape
            Q, R = np.linalg.qr(G.reshape(r1, n * r2).T)
            Q = Q.T
            cores[mu] = Q.reshape(Q.shape[0], n, r2)
            cores[mu - 1] = np.einsum('ijk,kl->ijl', cores[mu - 1], R.T)
        return cores

    def build_right_stack(self, Phi, cores):
        m = Phi.shape[0]
        Q_stack = [None] * self.d
        Q = np.ones((m, 1))
        for mu in reversed(range(self.d)):
            G = cores[mu]
            phi = Phi[:, mu, :]
            tmp = np.einsum('mi,rij->mrj', phi, G)
            Q = np.einsum('mrj,mj->mr', tmp, Q)
            Q_stack[mu] = Q
        return Q_stack

    def build_micromatrix(self, Phi, P_mu, Q_mu, mu):
        phi_mu = Phi[:, mu, :]
        M = np.einsum('mi,mj,mk->mijk', P_mu, phi_mu, Q_mu)
        return M.reshape(P_mu.shape[0], -1)

    def solve_local(self, M, v, r_prev, r_next):
        A = M.T @ M + self.reg * np.eye(M.shape[1])
        b = M.T @ v
        w = np.linalg.solve(A, b)

        W = w.reshape(r_prev * self.n_basis, r_next)
        U, S, Vt = np.linalg.svd(W, full_matrices=False)

        rank = min(self.r, len(S))
        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]
        return U, S, Vt

    def train_one_class(self, Phi, v):
        cores = self.init_tt()
        cores = self.right_orthogonalize(cores)

        for sweep in range(self.n_sweeps):
            Q_stack = self.build_right_stack(Phi, cores)
            P = np.ones((Phi.shape[0], 1))

            for mu in range(self.d - 1):
                M = self.build_micromatrix(Phi, P, Q_stack[mu + 1], mu)
                r_prev = cores[mu].shape[0]
                r_next = cores[mu].shape[2]

                U, S, Vt = self.solve_local(M, v, r_prev, r_next)

                cores[mu] = U.reshape(r_prev, self.n_basis, -1)
                cores[mu + 1] = np.einsum('ij,jkl->ikl', np.diag(S) @ Vt, cores[mu + 1])

                tmp = np.einsum('mi,rij->mrj', Phi[:, mu, :], cores[mu])
                P = np.einsum('mr,mrj->mj', P, tmp)

            for mu in reversed(range(1, self.d)):
                pass

        return cores

    def fit(self, X, y):
        Phi = self.feature_map(X)
        n_classes = np.max(y) + 1

        for i in range(n_classes):
            print(f"Training class {i}")
            v = (y == i).astype(float)
            cores = self.train_one_class(Phi, v)
            self.models.append(cores)

    def predict(self, X):
        Phi = self.feature_map(X)
        scores = []

        for cores in self.models:
            result = np.ones((X.shape[0], 1))
            for mu in range(self.d):
                G = cores[mu]
                tmp = np.einsum('mi,rij->mrj', Phi[:, mu, :], G)
                result = np.einsum('mr,mrj->mj', result, tmp)
            scores.append(result.squeeze())

        scores = np.stack(scores, axis=1)
        return np.argmax(scores, axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ---------- helperi za dataset ----------
def load_openml(name: str):
    ds = fetch_openml(name, version=1)
    X = ds.data.to_numpy().astype(np.float32) / 255.0
    y = ds.target.astype(int).to_numpy() if hasattr(ds.target, "to_numpy") else ds.target.astype(int)
    return X, y


def run_curve(X, y, train_sizes, test_size=5000, seed=42, model_kind="mandy"):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    X_test_down = downscale_2x2(X_test)

    accs = []
    for n in train_sizes:
        X_train = X_train_full[:n]
        y_train = y_train_full[:n]
        X_train_down = downscale_2x2(X_train)

        if model_kind == "mandy":
            clf = kernelbasedMANDyClassifier(alpha=0.59, reg=1e-3)
        elif model_kind == "arr":
            clf = ARRClassifier(d=196, n_basis=2, tt_rank=5, reg=1e-2, n_sweeps=5)
        else:
            raise ValueError("unknown model_kind")

        print(f"\n=== {model_kind.upper()} train_size={n} ===")
        clf.fit(X_train_down, y_train)
        acc = clf.score(X_test_down, y_test)
        print("Accuracy:", acc)
        accs.append(acc)

    return np.array(accs)


def main():
    # ⚠️ kernelbasedMANDyClassifier radi Gram matricu O(n^2).
    # 10k je već masivno. Za glatke krivulje koristi ove veličine:
    train_sizes = [1000, 2000, 3000, 5000, 7000, 10000]

    print("Loading MNIST...")
    X_mnist, y_mnist = load_openml("mnist_784")

    print("Loading Fashion-MNIST...")
    X_fmnist, y_fmnist = load_openml("Fashion-MNIST")

    # MANDy curve
    acc_mandy_mnist = run_curve(X_mnist, y_mnist, train_sizes, model_kind="mandy")
    acc_mandy_fmnist = run_curve(X_fmnist, y_fmnist, train_sizes, model_kind="mandy")

    # ARR curve (ako želiš i njega na grafu, odkomentiraj)
    acc_arr_mnist = run_curve(X_mnist, y_mnist, train_sizes, model_kind="arr")
    acc_arr_fmnist = run_curve(X_fmnist, y_fmnist, train_sizes, model_kind="arr")

    # Plot (jedan graf, 2 dataset-a, 2 metode)
    plt.figure(figsize=(10, 4))

    # lijevo MNIST
    plt.subplot(1, 2, 1)
    plt.plot(train_sizes, acc_mandy_mnist * 100, marker="o", label="MANDy MNIST")
    plt.plot(train_sizes, acc_arr_mnist * 100, marker="x", label="ARR MNIST")
    plt.xlabel("training data size")
    plt.ylabel("classification rate (%)")
    plt.ylim(75, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # desno FMNIST
    plt.subplot(1, 2, 2)
    plt.plot(train_sizes, acc_mandy_fmnist * 100, marker="s", label="MANDy FMNIST")
    plt.plot(train_sizes, acc_arr_fmnist * 100, marker="o", label="ARR FMNIST")
    plt.xlabel("training data size")
    plt.ylabel("classification rate (%)")
    plt.ylim(75, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()