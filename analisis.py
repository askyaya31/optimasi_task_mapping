"""Menghitung HV, IGD, makespan/energi terbaik per run, ringkasan mean±std, dan uji statistik.
Contoh:  python analisis.py                         (folder eksperimen)
         python analisis.py --folder uji_coba
Keluaran (di hasil/<folder>/): metrik_per_run.csv, ringkasan.csv, uji_statistik.csv"""
import argparse
import glob
import os
import pickle
import numpy as np
import pandas as pd
import config as C
import metrik as K


def muat(folder):
    data = {}
    for p in sorted(glob.glob(os.path.join(folder, "hasil_*.pkl"))):
        n = int(os.path.basename(p).split("_")[1].split(".")[0])
        data[n] = pickle.load(open(p, "rb"))
    return data


def hitung_per_run(data):
    baris = []
    for n, algos in data.items():
        ref = K.front_gabungan([F for d in algos.values() for F in d["F"]])    # front acuan per skenario
        lo, hi = ref.min(axis=0), ref.max(axis=0)
        R = K.normalisasi(ref, lo, hi)
        for nama, d in algos.items():
            for r, (F, t) in enumerate(zip(d["F"], d["waktu"]), 1):
                Z = K.normalisasi(F, lo, hi)
                baris.append(dict(ukuran=n, algoritma=nama, run=r, HV=K.hypervolume_2d(Z), IGD=K.igd(R, Z),
                                  makespan_min=F[:, 0].min(), energi_min=F[:, 1].min(),
                                  n_titik=len(F), waktu_s=t))
    return pd.DataFrame(baris)


def ringkasan(df):
    kolom = ["HV", "IGD", "makespan_min", "energi_min", "n_titik", "waktu_s"]
    g = df.groupby(["ukuran", "algoritma"])[kolom].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()


def uji(df):
    """NSGA2_adaptif vs setiap algoritma lain; NSGA2_standar vs varian ablasi. Rank-sum dua sisi."""
    baris = []
    for n, sub in df.groupby("ukuran"):
        ada = set(sub.algoritma)
        pasangan = [("NSGA2_adaptif", b) for b in sorted(ada - {"NSGA2_adaptif"})]
        pasangan += [("NSGA2_standar", b) for b in ("NSGA2_adaptif_saja", "NSGA2_sadar_saja") if b in ada]
        pasangan = [(x, y) for x, y in pasangan if x in ada and y in ada]
        for metrik, lebih_besar_baik in (("HV", True), ("IGD", False)):
            for x, y in pasangan:
                a = sub[sub.algoritma == x][metrik].to_numpy()
                b = sub[sub.algoritma == y][metrik].to_numpy()
                p = K.uji_rank_sum(a, b)
                pbonf = min(1.0, p * len(pasangan))
                lebih_baik = x if (np.median(a) > np.median(b)) == lebih_besar_baik else y
                baris.append(dict(ukuran=n, metrik=metrik, A=x, B=y, median_A=np.median(a), median_B=np.median(b),
                                  p=p, p_bonferroni=pbonf, signifikan_0_05=bool(pbonf < 0.05), lebih_baik=lebih_baik))
    return pd.DataFrame(baris)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="eksperimen")
    folder = os.path.join(C.OUT_DIR, ap.parse_args().folder)
    data = muat(folder)
    if not data:
        raise SystemExit(f"Tidak ada hasil_*.pkl di {folder}. Jalankan run_eksperimen.py dulu.")
    df = hitung_per_run(data)
    rs, ut = ringkasan(df), uji(df)
    df.to_csv(os.path.join(folder, "metrik_per_run.csv"), index=False)
    rs.to_csv(os.path.join(folder, "ringkasan.csv"), index=False)
    ut.to_csv(os.path.join(folder, "uji_statistik.csv"), index=False)

    pd.set_option("display.width", 200)
    for n in sorted(df.ukuran.unique()):
        s = rs[rs.ukuran == n]
        print(f"\n===== {n} task =====")
        tampil = pd.DataFrame({
            "algoritma": s.algoritma,
            "HV (mean±std)": [f"{m:.3f} ± {d:.3f}" for m, d in zip(s.HV_mean, s.HV_std)],
            "IGD (mean±std)": [f"{m:.3f} ± {d:.3f}" for m, d in zip(s.IGD_mean, s.IGD_std)],
            "makespan min": s.makespan_min_mean.round(2),
            "energi min": s.energi_min_mean.round(0),
            "waktu (s)": s.waktu_s_mean.round(1)})
        print(tampil.to_string(index=False))
    print("\n===== Uji statistik (rank-sum, p dikoreksi Bonferroni) =====")
    print(ut.round(4).to_string(index=False))
    print(f"\nTersimpan di {folder}")


if __name__ == "__main__":
    main()