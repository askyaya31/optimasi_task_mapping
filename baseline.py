"""Baseline: semua-ke-fog, semua-ke-cloud, round robin, dan N solusi acak + grafik trade-off.
Pemakaian:  python baseline.py            (100 task)
            python baseline.py 200        (skenario lain: 50/100/200/500)"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from fitness import Instance


def non_dominated(F):
    """Mask solusi non-dominated (minimasi dua objektif)."""
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        dom = (F <= F[i]).all(axis=1) & (F < F[i]).any(axis=1)
        keep[i] = not dom.any()
    return keep


def main(n_task=100, n_random=1000, seed=1):
    inst = Instance.from_excel(n_task)
    nf, M = C.N_FOG, inst.M
    idx = np.arange(n_task)

    rng = np.random.default_rng(seed)
    F_rand = inst.evaluate_pop(rng.integers(0, M, size=(n_random, n_task)))

    nama = {
        "Semua ke fog": idx % nf,
        "Semua ke cloud": nf + idx % C.N_CLOUD,
        "Round robin": idx % M,
    }
    hasil = {k: inst.evaluate(v) for k, v in nama.items()}

    tabel = pd.DataFrame(
        [(k, *v) for k, v in hasil.items()]
        + [("Acak (rata-rata)", *F_rand.mean(axis=0)),
           ("Acak (terbaik makespan)", *F_rand[F_rand[:, 0].argmin()]),
           ("Acak (terbaik energi)", *F_rand[F_rand[:, 1].argmin()])],
        columns=["Baseline", "Makespan (s)", "Energi (J)"]).round(2)
    print(f"\n=== Baseline, {n_task} task, {n_random} solusi acak ===")
    print(tabel.to_string(index=False))

    os.makedirs(C.OUT_DIR, exist_ok=True)
    tabel.to_csv(os.path.join(C.OUT_DIR, f"baseline_{n_task}.csv"), index=False)

    mask = non_dominated(F_rand)
    korelasi = np.corrcoef(F_rand[:, 0], F_rand[:, 1])[0, 1]
    print(f"\nDiagnosa trade-off: korelasi makespan-energi = {korelasi:.3f}; "
          f"non-dominated = {int(mask.sum())} dari {n_random}")
    if mask.sum() <= 2:
        print("PERINGATAN: kedua objektif hampir tidak bertentangan -> Pareto front nyaris satu titik. "
              "Cek PROFIL_DAYA di config.py.")
    plt.figure(figsize=(7, 5))
    plt.scatter(F_rand[:, 0], F_rand[:, 1], s=8, c="lightgray", label=f"{n_random} solusi acak")
    plt.scatter(F_rand[mask, 0], F_rand[mask, 1], s=14, c="tab:blue", label="non-dominated (acak)")
    for (k, v), c in zip(hasil.items(), ["tab:green", "tab:red", "tab:orange"]):
        plt.scatter(*v, s=90, c=c, marker="*", edgecolor="k", label=k)
    plt.xlabel("Makespan (detik)")
    plt.ylabel("Energi (Joule)")
    plt.title(f"Trade-off makespan vs energi ({n_task} task)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(C.OUT_DIR, f"baseline_{n_task}.png")
    plt.savefig(out, dpi=150)
    print(f"\nGrafik : {out}\nTabel  : {os.path.join(C.OUT_DIR, f'baseline_{n_task}.csv')}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
