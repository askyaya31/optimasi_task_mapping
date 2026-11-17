"""Metrik kualitas Pareto front dan uji statistik. Dua objektif, minimasi."""
import numpy as np
from scipy.stats import mannwhitneyu
import config as C

def non_dominated_mask(F):
    F = np.asarray(F, float)
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        dom = (F <= F[i]).all(axis=1) & (F < F[i]).any(axis=1)
        keep[i] = not dom.any()
    return keep


def front_gabungan(daftar_F):
    """Gabungkan banyak front lalu ambil yang non-dominated (dan unik)."""
    F = np.unique(np.vstack(daftar_F).round(9), axis=0)
    return F[non_dominated_mask(F)]


def normalisasi(F, lo, hi):
    return (np.asarray(F, float) - lo) / np.where(hi > lo, hi - lo, 1.0)


def hypervolume_2d(F, ref=(C.HV_REF, C.HV_REF)):
    """Luas daerah yang didominasi front terhadap titik acuan (F sudah ternormalisasi). Besar = bagus."""
    F = np.asarray(F, float)
    F = F[(F < np.array(ref)).all(axis=1)]
    if len(F) == 0:
        return 0.0
    F = F[np.argsort(F[:, 0])]
    hv, y_prev = 0.0, ref[1]
    for x, y in F:
        if y < y_prev:
            hv += (ref[0] - x) * (y_prev - y)
            y_prev = y
    return float(hv)


def igd(F_acuan, F):
    """Rata-rata jarak dari tiap titik front acuan ke titik terdekat milik F (ternormalisasi). Kecil = bagus."""
    R, F = np.asarray(F_acuan, float), np.asarray(F, float)
    d = np.sqrt(((R[:, None, :] - F[None, :, :]) ** 2).sum(axis=2))
    return float(d.min(axis=1).mean())


def uji_rank_sum(a, b):
    """Wilcoxon rank-sum (Mann-Whitney U), dua sisi. Mengembalikan p-value."""
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:      # semua nilai identik
        return 1.0