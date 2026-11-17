"""NSGA-II standar untuk task mapping fog-cloud (ditulis sendiri dengan numpy).
Antarmuka:  X_front, F_front = run(inst, seed, pop_size, n_gen)
  - X_front: solusi non-dominated terakhir (array indeks node), F_front: [makespan, energi]
Hook `mutation_fn(Xc, ctx)` dipakai nsga2_adaptif.py untuk mengganti operator mutasi.
Uji cepat:  python nsga2_standar.py 100"""
import os
import sys
import time
import numpy as np
import config as C
from fitness import Instance


# ---------- bagian inti NSGA-II ----------
def nondominated_sort(F):
    """Peringkat Pareto: 0 = front pertama. Minimasi semua objektif."""
    dom = (F[:, None, :] <= F[None, :, :]).all(axis=2) & (F[:, None, :] < F[None, :, :]).any(axis=2)
    counts = dom.sum(axis=0).astype(int)          # berapa solusi yang mendominasi solusi j
    rank = np.full(len(F), -1)
    front, r = np.where(counts == 0)[0], 0
    while front.size:
        rank[front] = r
        counts = counts - dom[front].sum(axis=0)
        front = np.where((counts == 0) & (rank == -1))[0]
        r += 1
    return rank


def crowding_distance(F, rank):
    n, m = F.shape
    cd = np.zeros(n)
    for r in np.unique(rank):
        idx = np.where(rank == r)[0]
        if len(idx) <= 2:
            cd[idx] = np.inf
            continue
        for k in range(m):
            order = idx[np.argsort(F[idx, k])]
            lo, hi = F[order[0], k], F[order[-1], k]
            cd[order[0]] = cd[order[-1]] = np.inf
            if hi > lo:
                cd[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / (hi - lo)
    return cd


def tournament(rank, cd, n, rng):
    a = rng.integers(0, len(rank), n)
    b = rng.integers(0, len(rank), n)
    a_menang = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (cd[a] >= cd[b]))
    return np.where(a_menang, a, b)


def mutasi_standar(Xc, ctx):
    """Mutasi acak: individu dimutasi dengan peluang pm; pada individu itu beberapa task
    dipindah ke node acak."""
    rng, M = ctx["rng"], ctx["M"]
    N = Xc.shape[1]
    k = max(1, int(round(C.MUT_FRAC * N)))
    for i in np.where(rng.random(len(Xc)) < ctx["pm"])[0]:
        pos = rng.choice(N, size=k, replace=False)
        Xc[i, pos] = rng.integers(0, M, size=k)
    return Xc


def run(inst, seed=0, pop_size=C.POP_SIZE, n_gen=C.N_GEN, pc=None, pm=None,
        mutation_fn=None, return_history=False):
    pc = C.PC if pc is None else pc      # dibaca saat dipanggil, supaya tuning.py bisa mengubahnya
    pm = C.PM if pm is None else pm
    rng = np.random.default_rng(seed)
    N, M = inst.N, inst.M
    mutation_fn = mutation_fn or mutasi_standar

    X = rng.integers(0, M, (pop_size, N))
    F = inst.evaluate_pop(X)
    rank = nondominated_sort(F)
    cd = crowding_distance(F, rank)
    history = []

    for g in range(n_gen):
        p1 = tournament(rank, cd, pop_size, rng)
        p2 = tournament(rank, cd, pop_size, rng)
        cross = rng.random(pop_size) < pc
        mask = rng.random((pop_size, N)) < 0.5                 # crossover seragam
        Xc = np.where(cross[:, None] & mask, X[p2], X[p1])

        ctx = dict(rng=rng, inst=inst, X=X, F=F, rank=rank, cd=cd, gen=g, n_gen=n_gen, pm=pm, M=M)
        Xc = mutation_fn(Xc, ctx)
        Fc = inst.evaluate_pop(Xc)

        Xa, Fa = np.vstack([X, Xc]), np.vstack([F, Fc])
        ra = nondominated_sort(Fa)
        ca = crowding_distance(Fa, ra)
        keep = np.lexsort((-ca, ra))[:pop_size]                # rank kecil dulu, lalu crowding besar
        X, F = Xa[keep], Fa[keep]
        rank = nondominated_sort(F)
        cd = crowding_distance(F, rank)
        if return_history:
            history.append(F[rank == 0].copy())

    front = rank == 0
    Xf, Ff = X[front], F[front]
    _, uniq = np.unique(Ff.round(9), axis=0, return_index=True)   # buang duplikat
    Xf, Ff = Xf[uniq], Ff[uniq]
    return (Xf, Ff, history) if return_history else (Xf, Ff)


# ---------- uji cepat ----------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_task = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    inst = Instance.from_excel(n_task)

    t0 = time.time()
    Xf, Ff = run(inst, seed=0)
    dt = time.time() - t0
    Ff = Ff[np.argsort(Ff[:, 0])]
    print(f"NSGA-II standar, {n_task} task, seed 0, waktu {dt:.1f} detik")
    print(f"Jumlah solusi di Pareto front : {len(Ff)}")
    print(f"Makespan  : {Ff[:, 0].min():.2f} - {Ff[:, 0].max():.2f} detik")
    print(f"Energi    : {Ff[:, 1].min():.0f} - {Ff[:, 1].max():.0f} Joule")

    rng = np.random.default_rng(1)
    Fr = inst.evaluate_pop(rng.integers(0, inst.M, (1000, n_task)))
    print(f"Pembanding acak terbaik makespan: {Fr[:, 0].min():.2f} detik, terbaik energi: {Fr[:, 1].min():.0f} J")

    os.makedirs(C.OUT_DIR, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.scatter(Fr[:, 0], Fr[:, 1], s=8, c="lightgray", label="1000 solusi acak")
    plt.plot(Ff[:, 0], Ff[:, 1], "o-", c="tab:blue", ms=4, label="Pareto front NSGA-II")
    plt.xlabel("Makespan (detik)")
    plt.ylabel("Energi (Joule)")
    plt.title(f"NSGA-II standar ({n_task} task)")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(C.OUT_DIR, f"nsga2_standar_{n_task}.png")
    plt.savefig(out, dpi=150)
    print("Grafik :", out)
