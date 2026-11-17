"""Model masalah: node, data task, dan fungsi makespan + energi.
Jalankan  `python fitness.py`  untuk uji kebenaran (cocok dengan hitungan tangan)."""
import numpy as np
import pandas as pd
import config as C


def build_nodes(seed=C.SEED_NODE):
    """Node 0..N_FOG-1 = fog, sisanya = cloud."""
    rng = np.random.default_rng(seed)
    nf, nc = C.N_FOG, C.N_CLOUD
    return {
        "S":      np.r_[rng.uniform(*C.FOG_MIPS, nf),  np.full(nc, C.CLOUD_MIPS)],
        "P_act":  np.r_[rng.uniform(*C.FOG_P_ACT, nf), np.full(nc, C.CLOUD_P_ACT)],
        "P_idle": np.r_[rng.uniform(*C.FOG_P_IDLE, nf), np.full(nc, C.CLOUD_P_IDLE)],
        "B":      np.r_[np.full(nf, C.FOG_BW),         np.full(nc, C.CLOUD_BW)],
        "tau":    np.r_[np.full(nf, C.FOG_LAT),        rng.uniform(*C.CLOUD_LAT, nc)],
    }


class Instance:
    """Satu masalah: N task dan M node. Solusi x = array panjang N, isi indeks node 0..M-1."""

    def __init__(self, L, D, nodes):
        self.L = np.asarray(L, float)          # panjang task (MI)
        self.D = np.asarray(D, float)          # ukuran data (Mb)
        self.__dict__.update(nodes)            # S, P_act, P_idle, B, tau
        self.N = len(self.L)
        self.M = len(self.S)

    @classmethod
    def from_excel(cls, n_task, nodes=None):
        df = pd.read_excel(C.DATA_FILE, sheet_name=C.SKENARIO[n_task])
        return cls(df["Panjang_Instruksi_MI"], df["Ukuran_Data_Mb"], nodes or build_nodes())

    def evaluate_pop(self, X):
        """X: (P, N) bilangan bulat. Mengembalikan (P, 2) = [makespan (s), energi (J)]."""
        X = np.atleast_2d(X).astype(int)
        T = self.L / self.S[X] + self.D / self.B[X] + self.tau[X]       # waktu proses tiap task
        F = np.stack([(T * (X == j)).sum(axis=1) for j in range(self.M)], axis=1)  # waktu selesai node
        mk = F.max(axis=1)
        en = (self.P_act * F + self.P_idle * (mk[:, None] - F)).sum(axis=1)
        return np.stack([mk, en], axis=1)

    def evaluate(self, x):
        return tuple(self.evaluate_pop(x)[0])


if __name__ == "__main__":
    # --- Uji 1: hitungan tangan (4 task, 2 node) ---
    nodes = {"S": [1000, 2000], "P_act": [30, 60], "P_idle": [20, 40], "B": [500, 100], "tau": [0, 0.01]}
    nodes = {k: np.array(v, float) for k, v in nodes.items()}
    kecil = Instance([1000, 2000, 2000, 1000], [5, 5, 5, 5], nodes)
    mk, en = kecil.evaluate([0, 1, 0, 1])
    print(f"Uji tangan : makespan={mk:.3f} (harus 3.020), energi={en:.2f} (harus 243.80)")
    assert abs(mk - 3.02) < 1e-9 and abs(en - 243.8) < 1e-6, "Fungsi evaluate SALAH"
    print("Uji tangan : LOLOS")

    # --- Uji 2: node + data Excel ---
    nd = build_nodes()
    print("\nNode (0-5 fog, 6-8 cloud):")
    print(pd.DataFrame(nd).round(3).to_string())
    inst = Instance.from_excel(100)
    print(f"\nData skenario 100 task dimuat: N={inst.N}, M={inst.M}")
    x = np.arange(inst.N) % inst.M
    print("Contoh round-robin -> makespan, energi =", tuple(round(float(v), 2) for v in inst.evaluate(x)))



def evaluate_pop(self, X):
    T = self.L / self.S[X] + self.D / self.B[X] + self.tau[X]
    F = np.stack([(T * (X == j)).sum(axis=1) for j in range(self.M)], axis=1)
    mk = F.max(axis=1)
    en = (self.P_act * F + self.P_idle * (mk[:, None] - F)).sum(axis=1)
    return np.stack([mk, en], axis=1)

