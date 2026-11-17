"""PSO diskret (posisi kontinu 0..M, di-floor jadi indeks node) dengan weighted sum.
Satu run = beberapa bobot seperti ga_weighted.py (aturan keadilan BUDGET_SAMA yang sama)."""
import numpy as np
import config as C
from metrik import non_dominated_mask


def _satu_bobot(inst, rng, w, lo, hi, n_part, n_iter):
    N, M = inst.N, inst.M

    def fit(pos):
        X = np.clip(np.floor(pos), 0, M - 1).astype(int)
        F = inst.evaluate_pop(X)
        Z = (F - lo) / (hi - lo)
        return w * Z[:, 0] + (1 - w) * Z[:, 1], F, X

    pos = rng.uniform(0, M, (n_part, N))
    vel = rng.uniform(-1, 1, (n_part, N))
    f, F, X = fit(pos)
    pbest, pbest_f = pos.copy(), f.copy()
    g = f.argmin()
    gbest, gbest_f, gX, gF = pos[g].copy(), f[g], X[g].copy(), F[g].copy()

    for t in range(n_iter):
        iw = C.PSO_W[0] - (C.PSO_W[0] - C.PSO_W[1]) * t / max(1, n_iter - 1)
        r1, r2 = rng.random((2, n_part, N))
        vel = iw * vel + C.PSO_C1 * r1 * (pbest - pos) + C.PSO_C2 * r2 * (gbest - pos)
        vel = np.clip(vel, -0.5 * M, 0.5 * M)
        pos = np.clip(pos + vel, 0, M - 1e-9)
        f, F, X = fit(pos)
        better = f < pbest_f
        pbest[better], pbest_f[better] = pos[better], f[better]
        k = f.argmin()
        if f[k] < gbest_f:
            gbest, gbest_f, gX, gF = pos[k].copy(), f[k], X[k].copy(), F[k].copy()
    return gX, gF


def run(inst, seed=0, pop_size=C.POP_SIZE, n_gen=C.N_GEN, weights=C.GA_WEIGHTS):
    rng = np.random.default_rng(seed)
    ref = inst.evaluate_pop(rng.integers(0, inst.M, (1000, inst.N)))
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    iters = max(1, n_gen // len(weights)) if C.BUDGET_SAMA else n_gen
    hasil = [_satu_bobot(inst, rng, w, lo, hi, pop_size, iters) for w in weights]
    X = np.array([h[0] for h in hasil])
    F = np.array([h[1] for h in hasil])
    m = non_dominated_mask(F)
    return X[m], F[m]