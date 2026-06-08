"""
afficher_dikau.py
=================
Script d'affichage de la classification de Dikau.
Lance directement depuis PyCharm.

Changer : TERRAIN = "plan" | "plateau" | "sinc_card" | "double_sin" | "reel"
"""

import numpy as np
import matplotlib.pyplot as plt
import fonction_theoriques as ft
import pente
from pente import calculer_courbures
from dikau import Dikau, classer_mnt
import visualisation as vis
import visualisation_dikau as vd

# ─────────────────────────────────────────────
TERRAIN       = "reel"
FICHIER_REEL  = "morneRouge.txt"
RESOL         = 1.0
EPSILON_P = 0.08    # au lieu de 2e-3
EPSILON_C = 0.5     # au lieu de 2e-2
# ─────────────────────────────────────────────

# ===========================================================================
# CHARGEMENT
# ===========================================================================

if TERRAIN == "reel":
    Z     = np.loadtxt(FICHIER_REEL)
    titre = "Morne Rouge"
else:
    N = 101
    x = np.arange(N, dtype=float)
    y = np.arange(N, dtype=float)
    X_tmp, Y_tmp = np.meshgrid(x, y)
    fn_z, _ = {
        "plan":       (ft.plan,       ft.grad_plan),
        "plateau":    (ft.plateau,    ft.grad_plateau),
        "sinc_card":  (ft.sinc_card,  ft.grad_sinc),
        "double_sin": (ft.double_sin, ft.grad_double_sin),
    }[TERRAIN]
    Z     = fn_z(X_tmp, Y_tmp)
    titre = TERRAIN

N_rows, N_cols = Z.shape
x = np.arange(N_cols) * RESOL
y = np.arange(N_rows) * RESOL
X, Y = np.meshgrid(x, y)

# ===========================================================================
# CALCULS
# ===========================================================================

print("Calcul des courbures...")
kv, kh, kmin, kmax, slope, G = calculer_courbures(Z, dx=RESOL)

print("Classification Dikau...")
mat_dikau = classer_mnt(kv, kh, kmin, kmax, slope,
                        epsilon_p=EPSILON_P, epsilon_c=EPSILON_C)

# Stats
total = np.sum(~np.isnan(mat_dikau))
print(f"\nDistribution ({total} pixels classes) :")
for x_cls in Dikau:
    cnt = int(np.sum(mat_dikau == x_cls.value))
    if cnt > 0:
        print(f"  {x_cls.name:20s} : {cnt:5d} ({100*cnt/total:.1f}%)")

print("Calcul des courbures...")
kv, kh, kmin, kmax, slope, G = calculer_courbures(Z, dx=RESOL)

# --- DIAGNOSTIC SEUILS ---
print(f"slope  : moy={slope.mean():.4f}  std={slope.std():.4f}")
print(f"kV     : moy={kv.mean():.4f}  std={kv.std():.4f}")
print(f"kH     : moy={kh.mean():.4f}  std={kh.std():.4f}")
print(f"kV p10/p90 : {np.percentile(kv,10):.4f} / {np.percentile(kv,90):.4f}")
print(f"kH p10/p90 : {np.percentile(kh,10):.4f} / {np.percentile(kh,90):.4f}")
# -------------------------

print("Classification Dikau...")

# ===========================================================================
# FIGURE 1 — MNT + courbures + classification
# ===========================================================================

fig1, axes1 = plt.subplots(1, 4, figsize=(22, 6))
fig1.suptitle(f"Dikau — {titre}", fontsize=13, fontweight='bold')

# MNT
_, axes1[0], im0 = vis.afficher_2D(X, Y, Z, ax=axes1[0],
                                    title="MNT",
                                    cmap='gist_earth',
                                    niveaux=True, n_levels=10,
                                    cotes=False, colorbar=True,
                                    Zname="z [m]",
                                    hillshade=(TERRAIN == "reel"),
                                    vert_exag=4, blend_mode='soft')

# Courbure kV
vd.afficher_courbures(X, Y, kv, kh,
                      ax_kv=axes1[1], ax_kh=axes1[2])

# Classification Dikau
vd.afficher_dikau(X, Y, mat_dikau, ax=axes1[3],
                  title="Classification Dikau",
                  colorbar=True)

plt.tight_layout()
plt.savefig(f"dikau_{titre}.png", dpi=130, bbox_inches='tight')
print(f"\n-> dikau_{titre}.png")

# ===========================================================================
# FIGURE 2 — Distribution des classes
# ===========================================================================

fig2, ax2 = plt.subplots(figsize=(9, 6))
fig2.suptitle(f"Distribution des classes — {titre}", fontweight='bold')
vd.afficher_distribution_dikau(mat_dikau, ax=ax2)

plt.tight_layout()
plt.savefig(f"dikau_distribution_{titre}.png", dpi=130, bbox_inches='tight')
print(f"-> dikau_distribution_{titre}.png")

plt.show()
