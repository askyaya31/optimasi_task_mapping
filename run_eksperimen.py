"""Menjalankan semua algoritma x ukuran task x N run, menyimpan semua Pareto front ke file .pkl.
Contoh:
  python run_eksperimen.py --ukuran 100 --run 3 --gen 60 --folder uji_coba     (uji cepat, 1 menit)
  python run_eksperimen.py --ablasi                                            (eksperimen penuh, 30 run)
Seed run ke-r = r (sama untuk semua algoritma). Hasil: hasil/<folder>/hasil_<ukuran>.pkl"""
import argparse
import os
import pickle
import time
import numpy as np
import config as C
from fitness import Instance
import nsga2_standar
import nsga2_adaptif
import ga_weighted
import pso

UTAMA = ["NSGA2_standar", "NSGA2_adaptif", "GA", "PSO"]
ABLASI = ["NSGA2_adaptif_saja", "NSGA2_sadar_saja"]


def daftar_algo(g):
    return {
        "NSGA2_standar":      lambda inst, s: nsga2_standar.run(inst, s, n_gen=g),
        "NSGA2_adaptif":      lambda inst, s: nsga2_adaptif.run(inst, s, n_gen=g),
        "NSGA2_adaptif_saja": lambda inst, s: nsga2_adaptif.run(inst, s, n_gen=g, adaptif=True, sadar=False),
        "NSGA2_sadar_saja":   lambda inst, s: nsga2_adaptif.run(inst, s, n_gen=g, adaptif=False, sadar=True),
        "GA":                 lambda inst, s: ga_weighted.run(inst, s, n_gen=g),
        "PSO":                lambda inst, s: pso.run(inst, s, n_gen=g),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ukuran", type=int, nargs="+", default=list(C.UKURAN))
    ap.add_argument("--run", type=int, default=C.N_RUN)
    ap.add_argument("--gen", type=int, default=C.N_GEN)
    ap.add_argument("--algo", nargs="+", default=None, help="nama algoritma (default: 4 utama)")
    ap.add_argument("--ablasi", action="store_true", help="tambahkan 2 varian ablasi")
    ap.add_argument("--folder", default="eksperimen")
    a = ap.parse_args()

    algos = daftar_algo(a.gen)
    pilih = a.algo or (UTAMA + (ABLASI if a.ablasi else []))
    folder = os.path.join(C.OUT_DIR, a.folder)
    os.makedirs(folder, exist_ok=True)

    for n in a.ukuran:
        inst = Instance.from_excel(n)
        path = os.path.join(folder, f"hasil_{n}.pkl")
        data = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
        for nama in pilih:
            fronts, waktu = [], []
            for r in range(1, a.run + 1):
                t0 = time.time()
                _, F = algos[nama](inst, r)
                waktu.append(time.time() - t0)
                fronts.append(np.asarray(F))
                print(f"[{n:3d} task] {nama:20s} run {r:2d}/{a.run}  titik={len(F):3d}  {waktu[-1]:.1f}s", flush=True)
            data[nama] = {"F": fronts, "waktu": waktu}
            pickle.dump(data, open(path, "wb"))          # simpan setiap selesai satu algoritma
        print(f"== tersimpan: {path}\n")


if __name__ == "__main__":
    main()