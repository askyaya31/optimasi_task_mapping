"""Grafik untuk artikel: Pareto front (run dengan HV median per algoritma) dan boxplot HV/IGD.
Contoh:  python grafik.py   atau   python grafik.py --folder uji_coba
Keluaran: hasil/<folder>/grafik/*.png"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
import metrik as K
from analisis import muat, hitung_per_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="eksperimen")
    folder = os.path.join(C.OUT_DIR, ap.parse_args().folder)
    data = muat(folder)
    df = hitung_per_run(data)
    out = os.path.join(folder, "grafik")
    os.makedirs(out, exist_ok=True)

    for n, algos in data.items():
        sub = df[df.ukuran == n]
        # 1) Pareto front dari run dengan HV median
        plt.figure(figsize=(7, 5))
        for nama, d in algos.items():
            hv = sub[sub.algoritma == nama].sort_values("run").HV.to_numpy()
            r = int(np.argsort(hv)[len(hv) // 2])
            F = d["F"][r][np.argsort(d["F"][r][:, 0])]
            plt.plot(F[:, 0], F[:, 1], "o-", ms=3, lw=1, label=nama)
        plt.xlabel("Makespan (detik)"); plt.ylabel("Energi (Joule)")
        plt.title(f"Pareto front, run dengan HV median ({n} task)")
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(out, f"pareto_{n}.png"), dpi=150); plt.close()
        # 2) boxplot HV dan IGD
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for a, m, jdl in zip(ax, ("HV", "IGD"), ("Hypervolume (besar = baik)", "IGD (kecil = baik)")):
            nama = list(algos)
            a.boxplot([sub[sub.algoritma == k][m] for k in nama])
            a.set_xticklabels(nama)      # cara ini jalan di semua versi matplotlib
            a.set_title(f"{jdl}, {n} task"); a.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"boxplot_{n}.png"), dpi=150); plt.close()
    print("Grafik tersimpan di", out)


if __name__ == "__main__":
    main()