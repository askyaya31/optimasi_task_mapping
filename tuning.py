"""Tuning parameter pada seed TERPISAH (101-105), BUKAN seed eksperimen (1-30).
Tahap A: cari PC dan PM terbaik untuk NSGA-II standar.
Tahap B: dengan PC, PM itu, cari BETA dan BIAS_Q terbaik untuk NSGA-II adaptif (komponen sadar fog-cloud).
Jalankan:  python tuning.py A     lalu tulis PC, PM terbaik ke config.py
           python tuning.py B     lalu tulis BETA, BIAS_Q terbaik ke config.py   (tiap tahap sekitar 3-5 menit)"""
import itertools
import numpy as np
import config as C
from fitness import Instance
import nsga2_standar as S
import nsga2_adaptif as A
import metrik as K

SEEDS = [101, 102, 103, 104, 105]
UKURAN = (100, 500)


def skor(fronts):
    """fronts: {nama: [F per seed]} -> {nama: HV rata-rata} (normalisasi dari gabungan semua front)."""
    ref = K.front_gabungan([F for L in fronts.values() for F in L])
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    return {k: float(np.mean([K.hypervolume_2d(K.normalisasi(F, lo, hi)) for F in L])) for k, L in fronts.items()}


def tahap(nama_param, kandidat, buat_front):
    total = {k: 0.0 for k in kandidat}
    for n in UKURAN:
        inst = Instance.from_excel(n)
        fr = {k: [buat_front(inst, k, s) for s in SEEDS] for k in kandidat}
        for k, v in skor(fr).items():
            total[k] += v / len(UKURAN)
    print(f"\n{nama_param}  (HV rata-rata, makin besar makin baik)")
    for k, v in sorted(total.items(), key=lambda kv: -kv[1]):
        print(f"   {k}  ->  {v:.4f}")
    return max(total, key=total.get)


if __name__ == "__main__":
    import sys
    tahap_pilih = sys.argv[1].upper() if len(sys.argv) > 1 else "AB"

    if "A" in tahap_pilih:       # Tahap A: NSGA-II standar
        kand = list(itertools.product((0.7, 0.9, 1.0), (0.1, 0.2, 0.4)))      # (PC, PM)
        best = tahap("TAHAP A: NSGA-II standar (PC, PM)", kand,
                     lambda inst, k, s: S.run(inst, s, pc=k[0], pm=k[1])[1])
        C.PC, C.PM = best
        print(f"\n>>> Tulis di config.py:  PC = {best[0]}   PM = {best[1]}")

    if "B" in tahap_pilih:       # Tahap B: komponen sadar fog-cloud (memakai PC, PM di config.py)
        kand = list(itertools.product((1.0, 2.0, 3.0), (0.5, 1.0)))            # (BETA, BIAS_Q)

        def buat(inst, k, s):
            C.BETA, C.BIAS_Q = k
            return A.run(inst, s)[1]

        best2 = tahap("TAHAP B: NSGA-II adaptif (BETA, BIAS_Q)", kand, buat)
        print(f"\n>>> Tulis di config.py:  BETA = {best2[0]}   BIAS_Q = {best2[1]}")
    print("\nCatatan: ambang keragaman (DIV_LOW, DIV_HIGH) dan PM_UP/PM_DOWN bisa dituning dengan cara yang sama.")