"""Membandingkan profil daya A, B, C: apakah makespan dan energi benar-benar bertentangan?
Jalankan:  python bandingkan_profil.py        (100 task) atau  python bandingkan_profil.py 200
Keluaran: tabel di layar + hasil/perbandingan_profil_<n>.png dan .csv"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from fitness import Instance, build_nodes
from baseline import non_dominated


def nodes_untuk(profil):
    p = C.PROFIL[profil]
    C.FOG_P_ACT, C.FOG_P_IDLE = p["FOG_P_ACT"], p["FOG_P_IDLE"]
    C.CLOUD_P_ACT, C.CLOUD_P_IDLE = p["CLOUD_P_ACT"], p["CLOUD_P_IDLE"]
    return build_nodes()


def main(n_task=100, n_acak=1000):
    rng = np.random.default_rng(1)
    X = rng.integers(0, C.N_FOG + C.N_CLOUD, size=(n_acak, n_task))      # solusi acak yang SAMA untuk semua profil
    judul = {"A": "A: angka slide proposal (cloud 60 W)", "B": "B: cloud 200 W", "C": "C: cloud 400 W"}
    baris = []
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, p in zip(ax, "ABC"):
        F = Instance.from_excel(n_task, nodes_untuk(p)).evaluate_pop(X)
        mask = non_dominated(F)
        korelasi = np.corrcoef(F[:, 0], F[:, 1])[0, 1]
        baris.append(dict(profil=p, cloud_aktif_W=C.PROFIL[p]["CLOUD_P_ACT"], korelasi=round(korelasi, 3),
                          non_dominated=int(mask.sum()), dari=n_acak))
        a.scatter(F[:, 0], F[:, 1], s=6, c="lightgray")
        a.scatter(F[mask, 0], F[mask, 1], s=14, c="tab:blue")
        a.set_title(f"Profil {judul[p]}\nkorelasi={korelasi:.2f}, non-dominated={int(mask.sum())}", fontsize=9)
        a.set_xlabel("Makespan (detik)")
        a.set_ylabel("Energi (Joule)")
    plt.tight_layout()
    os.makedirs(C.OUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(C.OUT_DIR, f"perbandingan_profil_{n_task}.png"), dpi=150)
    df = pd.DataFrame(baris)
    df.to_csv(os.path.join(C.OUT_DIR, f"perbandingan_profil_{n_task}.csv"), index=False)
    print(df.to_string(index=False))
    print("\nSemakin kecil korelasi dan semakin banyak non-dominated, semakin nyata trade-off makespan vs energi.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)