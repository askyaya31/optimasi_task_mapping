"""GA satu-objektif dengan weighted sum. Satu run = beberapa bobot (C.GA_WEIGHTS), tiap bobot
satu GA penuh; hasil akhir = solusi non-dominated dari semua bobot (jadi punya 'front' untuk HV/IGD).
KEADILAN: bila C.BUDGET_SAMA = True, generasi tiap bobot = n_gen / jumlah bobot, sehingga total evaluasi
sama dengan NSGA-II. Bila False, GA memakai (jumlah bobot) kali lebih banyak evaluasi (sebutkan di artikel)."""
import numpy as np
import config as C
from nsga2_standar import mutasi_standar
from metrik import non_dominated_mask


def _satu_bobot(inst, rng, w, lo, hi, pop_size, n_gen, pc, pm):
    N, M = inst.N, inst.M

    def fit(X):
        F = inst.evaluate_pop(X)
        Z = (F - lo) / (hi - lo)
        return w * Z[:, 0] + (1 - w) * Z[:, 1], F

    X = rng.integers(0, M, (pop_size, N))
    f, F = fit(X)
    for _ in range(n_gen):
        a, b = rng.integers(0, pop_size, (2, pop_size))
        p1 = np.where(f[a] <= f[b], a, b)
        a, b = rng.integers(0, pop_size, (2, pop_size))
        p2 = np.where(f[a] <= f[b], a, b)
        cross = rng.random(pop_size) < pc
        mask = rng.random((pop_size, N)) < 0.5
        Xc = np.where(cross[:, None] & mask, X[p2], X[p1])
        Xc = mutasi_standar(Xc, dict(rng=rng, M=M, pm=pm))
        fc, Fc = fit(Xc)
        Xa, fa, Fa = np.vstack([X, Xc]), np.r_[f, fc], np.vstack([F, Fc])
        keep = np.argsort(fa)[:pop_size]                     # (mu + lambda): elitisme
        X, f, F = Xa[keep], fa[keep], Fa[keep]
    return X[0], F[0]


def run(inst, seed=0, pop_size=C.POP_SIZE, n_gen=C.N_GEN, pc=None, pm=None, weights=C.GA_WEIGHTS):
    pc = C.PC if pc is None else pc
    pm = C.PM if pm is None else pm
    rng = np.random.default_rng(seed)
    ref = inst.evaluate_pop(rng.integers(0, inst.M, (1000, inst.N)))
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    gen = max(1, n_gen // len(weights)) if C.BUDGET_SAMA else n_gen
    hasil = [_satu_bobot(inst, rng, w, lo, hi, pop_size, gen, pc, pm) for w in weights]
    X = np.array([h[0] for h in hasil])
    F = np.array([h[1] for h in hasil])
    m = non_dominated_mask(F)
    return X[m], F[m]