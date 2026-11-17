"""Konfigurasi bersama. Ubah angka HANYA di sini supaya semua file memakai parameter yang sama."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "Dataset_Task_FogCloud.xlsx")
OUT_DIR = os.path.join(BASE_DIR, "hasil")

# jumlah task -> nama sheet di Excel
SKENARIO = {
    50: "Skenario_50_Task",
    100: "Skenario_100_Task",
    200: "Skenario_200_Task",
    500: "Skenario_500_Task",
}

# ---- Node (USULAN AWAL, samakan dengan paper Karami hal. 6-8 setelah dicek) ----
SEED_NODE = 42           # seed tetap supaya node sama di semua anggota tim
N_FOG, N_CLOUD = 6, 3

# Profil daya (HASIL UJI, 100 task, NSGA-II):
#  "A" = angka di slide proposal. Makespan dan energi nyaris tidak bertentangan
#        (korelasi 0.999, Pareto front ~1 titik) -> NSGA-II tidak ada gunanya.
#  "B" = cloud aktif 200 W. Trade-off ada tapi lemah (energi hanya beda ~2% sepanjang front, ~8 titik).
#  "C" = cloud aktif 400 W. Trade-off jelas (energi beda ~51%, makespan ~64%, ~48 titik). DEFAULT.
PROFIL_DAYA = "C"
PROFIL = {
    "A": dict(FOG_P_ACT=(30.0, 50.0), FOG_P_IDLE=(20.0, 30.0), CLOUD_P_ACT=60.0,  CLOUD_P_IDLE=36.0),
    "B": dict(FOG_P_ACT=(30.0, 50.0), FOG_P_IDLE=(5.0, 10.0),  CLOUD_P_ACT=200.0, CLOUD_P_IDLE=60.0),
    "C": dict(FOG_P_ACT=(30.0, 50.0), FOG_P_IDLE=(5.0, 10.0),  CLOUD_P_ACT=400.0, CLOUD_P_IDLE=60.0),
}
_p = PROFIL[PROFIL_DAYA]

FOG_MIPS = (2000.0, 3500.0)    # kecepatan (MIPS), diacak seragam per node
FOG_P_ACT = _p["FOG_P_ACT"]    # daya aktif (W)
FOG_P_IDLE = _p["FOG_P_IDLE"]  # daya idle (W)
FOG_BW = 500.0                 # bandwidth (Mbps)
FOG_LAT = 0.0                  # latensi (detik)

CLOUD_MIPS = 6000.0
CLOUD_P_ACT = _p["CLOUD_P_ACT"]
CLOUD_P_IDLE = _p["CLOUD_P_IDLE"]
CLOUD_BW = 100.0
CLOUD_LAT = (0.05, 0.10)       # latensi 50-100 ms (ASUMSI kita), dalam detik

# ---- Parameter algoritma evolusi (dipakai NSGA-II dan GA) ----
POP_SIZE = 100          # ukuran populasi (acuan Karami)
N_GEN = 400             # jumlah generasi (acuan Karami)
PC = 0.7                # peluang crossover (DITUNING, lihat tuning.py)
PM = 0.4                # peluang sebuah individu dimutasi (DITUNING, lihat tuning.py)
MUT_FRAC = 0.05         # sebagian kecil gen (task) yang dipindah node pada individu yang dimutasi (min 1 gen)

# ---- NSGA-II adaptif (nsga2_adaptif.py) ----
DIV_LOW, DIV_HIGH = 0.55, 0.65   # ambang keragaman populasi (0-1); DITUNING
PM_MIN, PM_MAX = 0.05, 0.60      # batas bawah/atas peluang mutasi individu
PM_UP, PM_DOWN = 1.3, 0.8        # pengali saat keragaman rendah / tinggi
BETA = 1.0                       # kuat "condong" ke node cepat: peluang ~ (1/T_ij)^BETA; 0 = acak
BIAS_Q = 0.5                     # porsi gen mutasi yang memakai pilihan condong (sisanya acak)

# ---- GA dan PSO (weighted sum) ----
GA_WEIGHTS = (0.1, 0.3, 0.5, 0.7, 0.9)   # bobot makespan; bobot energi = 1 - w
BUDGET_SAMA = True    # True: generasi GA/PSO dibagi jumlah bobot, jadi total evaluasi = NSGA-II (adil)
PSO_W = (0.9, 0.4)                       # inersia: turun linear dari 0.9 ke 0.4
PSO_C1 = PSO_C2 = 2.0

# ---- Eksperimen ----
N_RUN = 30
UKURAN = (50, 100, 200, 500)
HV_REF = 1.1             # titik acuan Hypervolume (setelah normalisasi 0-1)
