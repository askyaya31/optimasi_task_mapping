"""NSGA-II adaptif: hanya operator MUTASI yang diganti (crossover & seleksi tetap standar).
 1) Mutasi adaptif   : peluang mutasi naik saat keragaman populasi rendah, turun saat tinggi.
 2) Mutasi sadar fog-cloud : node tujuan dipilih condong ke node yang waktu prosesnya T_ij kecil
    (memperhitungkan panjang task, ukuran data, bandwidth, latensi).
Antarmuka sama dengan nsga2_standar:  Xf, Ff = run(inst, seed)
Varian ablasi: run(inst, seed, adaptif=True, sadar=False) dst."""
import numpy as np
import config as C
from nsga2_standar import run as _run_nsga2


def keragaman(X, rng, n_pasang=200):
    """Keragaman populasi 0-1: rata-rata proporsi gen yang berbeda antara dua individu acak."""
    i = rng.integers(0, len(X), n_pasang)
    j = rng.integers(0, len(X), n_pasang)
    return float((X[i] != X[j]).mean())


class MutasiAdaptif:
    def __init__(self, inst, adaptif=True, sadar=True):
        self.adaptif, self.sadar = adaptif, sadar
        self.pm = C.PM                                   # peluang awal = sama dengan NSGA-II standar
        self.riwayat = []                                # (generasi, keragaman, pm) untuk analisis
        if sadar:
            T = (inst.L[:, None] / inst.S[None, :]
                 + inst.D[:, None] / inst.B[None, :] + inst.tau[None, :])    # T_ij: (N, M)
            W = (1.0 / T) ** C.BETA
            self.cum = np.cumsum(W / W.sum(axis=1, keepdims=True), axis=1)  # peluang kumulatif per task

    def __call__(self, Xc, ctx):
        rng, M = ctx["rng"], ctx["M"]
        N = Xc.shape[1]
        if self.adaptif:
            d = keragaman(ctx["X"], rng)
            if d < C.DIV_LOW:
                self.pm = min(C.PM_MAX, self.pm * C.PM_UP)
            elif d > C.DIV_HIGH:
                self.pm = max(C.PM_MIN, self.pm * C.PM_DOWN)
            pm = self.pm
            self.riwayat.append((ctx["gen"], d, pm))
        else:
            pm = ctx["pm"]

        k = max(1, int(round(C.MUT_FRAC * N)))
        for i in np.where(rng.random(len(Xc)) < pm)[0]:
            pos = rng.choice(N, size=k, replace=False)
            baru = rng.integers(0, M, size=k)                         # pilihan acak
            if self.sadar:
                pakai = rng.random(k) < C.BIAS_Q                      # sebagian gen memakai pilihan condong
                u = rng.random(k)
                condong = (u[:, None] > self.cum[pos]).sum(axis=1).clip(0, M - 1)
                baru = np.where(pakai, condong, baru)
            Xc[i, pos] = baru
        return Xc


def run(inst, seed=0, pop_size=C.POP_SIZE, n_gen=C.N_GEN, adaptif=True, sadar=True):
    return _run_nsga2(inst, seed, pop_size, n_gen, mutation_fn=MutasiAdaptif(inst, adaptif, sadar))