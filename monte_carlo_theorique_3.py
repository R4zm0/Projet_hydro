"""
monte_carlo_theorique.py
========================
Monte Carlo — impact du bruit blanc σ variable sur pente et exposition.
Compare les 3 méthodes : TPP, FCN, Evans.

Changer : TERRAIN = "plan" | "plateau" | "sinc_card" | "double_sin"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import CenteredNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pente
import fonction_theoriques as ft

# ─────────────────────────────────────────────
TERRAIN = "plateau"
N_MC    = 200
SIGMAS  = [0.0, 0.05, 0.1, 0.2, 0.5]
# ─────────────────────────────────────────────

N = 101; dx = 1.0
x = np.arange(N, dtype=float); y = np.arange(N, dtype=float)
X, Y = np.meshgrid(x, y)

fn_z, fn_grad = {
    "plan":       (ft.plan,       ft.grad_plan),
    "plateau":    (ft.plateau,    ft.grad_plateau),
    "sinc_card":  (ft.sinc_card,  ft.grad_sinc),
    "double_sin": (ft.double_sin, ft.grad_double_sin),
}[TERRAIN]

Z    = fn_z(X, Y)
G_th = fn_grad(X, Y)

sl = slice(1, -1)
Zr = Z[sl, sl];  Xr = X[sl, sl];  Yr = Y[sl, sl]
G_th_r = G_th[sl, sl]
slope_th, aspect_th = pente.slope_aspect(G_th_r[..., 0], G_th_r[..., 1])
ext = (Xr.min(), Xr.max(), Yr.min(), Yr.max())

def diff_angle(a, b):
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi

# Style clair
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.edgecolor':   '#333333',
    'axes.labelcolor':  '#222222',
    'xtick.color':      '#333333',
    'ytick.color':      '#333333',
    'text.color':       '#111111',
    'grid.color':       '#cccccc',
    'grid.alpha':       0.6,
    'legend.facecolor': 'white',
    'legend.edgecolor': '#aaaaaa',
    'font.size':        9,
})

COULEURS = {"TPP": "#E07030", "FCN": "#2E8B57", "Evans": "#1A5EA8"}

def cb(fig, im, ax, label):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    c = fig.colorbar(im, cax=cax, label=label)
    return c

# ===========================================================================
# MONTE CARLO — 3 méthodes
# ===========================================================================

# Stockage : dict méthode → liste sur SIGMAS
resultats = {
    "TPP":   {"slope_rmse": [], "aspect_rmse": [], "slope_bias": [], "aspect_bias": []},
    "FCN":   {"slope_rmse": [], "aspect_rmse": [], "slope_bias": [], "aspect_bias": []},
    "Evans": {"slope_rmse": [], "aspect_rmse": [], "slope_bias": [], "aspect_bias": []},
}

for sigma in SIGMAS:
    print(f"\nσ = {sigma:.2f}  ({N_MC} tirages)...")

    # Tableaux de tirages pour les 3 méthodes
    mc = {m: {"slopes": np.zeros((N_MC, Zr.shape[0], Zr.shape[1])),
              "aspects": np.zeros((N_MC, Zr.shape[0], Zr.shape[1]))}
          for m in ["TPP", "FCN", "Evans"]}

    for k in range(N_MC):
        bruit  = np.random.normal(0, sigma, Z.shape)
        Zb     = Z + bruit

        G_tpp_b         = pente.gradient_tpp(Zb, dx=dx, dy=dx)[sl, sl]
        G_fcn_b         = pente.gradient_fcn(Zb, dx=dx, dy=dx)[sl, sl]
        G_ev_b, _       = pente.gradient_evans(Zb, s=dx)
        G_ev_b          = G_ev_b[sl, sl]

        for nom, G in [("TPP", G_tpp_b), ("FCN", G_fcn_b), ("Evans", G_ev_b)]:
            sl_b, asp_b = pente.slope_aspect(G[..., 0], G[..., 1])
            mc[nom]["slopes"] [k] = sl_b
            mc[nom]["aspects"][k] = asp_b

    for nom in ["TPP", "FCN", "Evans"]:
        slopes_mc  = mc[nom]["slopes"]
        aspects_mc = mc[nom]["aspects"]

        resultats[nom]["slope_rmse"].append(
            np.sqrt(np.mean((slopes_mc - slope_th)**2, axis=0)))
        resultats[nom]["aspect_rmse"].append(
            np.sqrt(np.mean(diff_angle(aspects_mc, aspect_th)**2, axis=0)))
        resultats[nom]["slope_bias"].append(
            np.mean(slopes_mc, axis=0) - slope_th)
        resultats[nom]["aspect_bias"].append(
            diff_angle(np.mean(aspects_mc, axis=0), aspect_th))

        rmse_sl  = resultats[nom]["slope_rmse"][-1].mean()
        rmse_asp = np.degrees(resultats[nom]["aspect_rmse"][-1].mean())
        print(f"  {nom:5s}  RMSE_pente={rmse_sl:.4f}  RMSE_aspect={rmse_asp:.2f}°")

# Convertir en arrays
for nom in resultats:
    for k in resultats[nom]:
        resultats[nom][k] = np.array(resultats[nom][k])

# ===========================================================================
# FIGURE 1 — Cartes RMSE pente pour les 3 méthodes × 5 niveaux de bruit
# ===========================================================================

print("\nFigure 1 : cartes RMSE pente...")
fig1, axes1 = plt.subplots(3, len(SIGMAS), figsize=(4 * len(SIGMAS), 11))
fig1.suptitle(f"RMSE pente vs analytique — {TERRAIN}  ({N_MC} tirages)\n"
              "Une ligne par méthode, une colonne par niveau de bruit",
              fontsize=12, fontweight='bold')

for row, nom in enumerate(["TPP", "FCN", "Evans"]):
    vmax = resultats[nom]["slope_rmse"][-1].max()   # échelle sur sigma max
    for col, sigma in enumerate(SIGMAS):
        ax = axes1[row, col]
        im = ax.imshow(resultats[nom]["slope_rmse"][col],
                       origin='lower', extent=ext,
                       cmap='hot', vmin=0, vmax=vmax)
        rmse_moy = resultats[nom]["slope_rmse"][col].mean()
        if row == 0:
            ax.set_title(f"σ_bruit = {sigma:.2f}", fontweight='bold', fontsize=9)
        ax.set_xlabel("x [m]")
        if col == 0:
            ax.set_ylabel(f"{nom}\nRMSE={rmse_moy:.4f}", color=COULEURS[nom],
                          fontweight='bold')
        else:
            ax.set_ylabel(f"RMSE={rmse_moy:.4f}", fontsize=8)
        cb(fig1, im, ax, "RMSE [m/m]")
        ax.grid(True)

plt.tight_layout()
plt.savefig(f"MC_cartes_{TERRAIN}.png", dpi=110, bbox_inches='tight')
print(f"  → MC_cartes_{TERRAIN}.png")

# ===========================================================================
# FIGURE 2 — Cartes RMSE exposition pour les 3 méthodes × 5 niveaux de bruit
# ===========================================================================

print("Figure 2 : cartes RMSE exposition...")
fig2, axes2 = plt.subplots(3, len(SIGMAS), figsize=(4 * len(SIGMAS), 11))
fig2.suptitle(f"RMSE exposition vs analytique — {TERRAIN}  ({N_MC} tirages)",
              fontsize=12, fontweight='bold')

for row, nom in enumerate(["TPP", "FCN", "Evans"]):
    vmax = np.degrees(resultats[nom]["aspect_rmse"][-1].max())
    for col, sigma in enumerate(SIGMAS):
        ax = axes2[row, col]
        im = ax.imshow(np.degrees(resultats[nom]["aspect_rmse"][col]),
                       origin='lower', extent=ext,
                       cmap='hot', vmin=0, vmax=vmax)
        rmse_moy = np.degrees(resultats[nom]["aspect_rmse"][col].mean())
        if row == 0:
            ax.set_title(f"σ_bruit = {sigma:.2f}", fontweight='bold', fontsize=9)
        ax.set_xlabel("x [m]")
        if col == 0:
            ax.set_ylabel(f"{nom}\nRMSE={rmse_moy:.2f}°", color=COULEURS[nom],
                          fontweight='bold')
        else:
            ax.set_ylabel(f"RMSE={rmse_moy:.2f}°", fontsize=8)
        cb(fig2, im, ax, "RMSE [°]")
        ax.grid(True)

plt.tight_layout()
plt.savefig(f"MC_cartes_aspect_{TERRAIN}.png", dpi=110, bbox_inches='tight')
print(f"  → MC_cartes_aspect_{TERRAIN}.png")

# ===========================================================================
# FIGURE 3 — Cartes biais pente (σ=0.1 uniquement, les 3 méthodes côte à côte)
# ===========================================================================

print("Figure 3 : cartes biais...")
idx_01 = min(range(len(SIGMAS)), key=lambda i: abs(SIGMAS[i] - 0.1))
sigma_ref = SIGMAS[idx_01]

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle(f"Biais pente (moy tirages − analytique) — {TERRAIN}\n"
              f"σ_bruit = {sigma_ref:.2f}",
              fontsize=12, fontweight='bold')

vabs = max(np.max(np.abs(resultats[nom]["slope_bias"][idx_01]))
           for nom in ["TPP", "FCN", "Evans"])

for col, nom in enumerate(["TPP", "FCN", "Evans"]):
    ax = axes3[col]
    biais_moy = resultats[nom]["slope_bias"][idx_01].mean()
    im = ax.imshow(resultats[nom]["slope_bias"][idx_01],
                   origin='lower', extent=ext,
                   cmap='seismic', norm=CenteredNorm(0))
    ax.set_title(f"{nom}\nbiais moyen = {biais_moy:+.4f}", fontweight='bold',
                 color=COULEURS[nom])
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    cb(fig3, im, ax, "Biais [m/m]")
    ax.grid(True)

plt.tight_layout()
plt.savefig(f"MC_biais_{TERRAIN}.png", dpi=110, bbox_inches='tight')
print(f"  → MC_biais_{TERRAIN}.png")

# ===========================================================================
# FIGURE 4 — Synthèse : RMSE moyen vs σ_bruit, comparaison 3 méthodes
# ===========================================================================

print("Figure 4 : synthèse comparaison...")
fig4, (ax_sl, ax_asp) = plt.subplots(1, 2, figsize=(14, 6))
fig4.suptitle(f"Monte Carlo — Comparaison TPP / FCN / Evans — {TERRAIN}  ({N_MC} tirages)",
              fontsize=12, fontweight='bold')

sigmas_arr = np.array(SIGMAS)

for nom in ["TPP", "FCN", "Evans"]:
    c = COULEURS[nom]

    # Pente
    moy_sl = resultats[nom]["slope_rmse"].mean(axis=(1, 2))
    std_sl = resultats[nom]["slope_rmse"].std (axis=(1, 2))
    ax_sl.plot(sigmas_arr, moy_sl, 'o-', color=c, lw=2, ms=6, label=nom)
    ax_sl.fill_between(sigmas_arr, moy_sl - std_sl, moy_sl + std_sl,
                       color=c, alpha=0.15)

    # Exposition
    moy_asp = np.degrees(resultats[nom]["aspect_rmse"].mean(axis=(1, 2)))
    std_asp = np.degrees(resultats[nom]["aspect_rmse"].std (axis=(1, 2)))
    ax_asp.plot(sigmas_arr, moy_asp, 'o-', color=c, lw=2, ms=6, label=nom)
    ax_asp.fill_between(sigmas_arr, moy_asp - std_asp, moy_asp + std_asp,
                        color=c, alpha=0.15)

ax_sl.set_xlabel("Bruit σ_bruit [unités Z]")
ax_sl.set_ylabel("RMSE pente moyen [m/m]")
ax_sl.set_title("Pente", fontweight='bold')
ax_sl.legend(fontsize=10); ax_sl.grid(True)
ax_sl.text(0.02, 0.97,
    "La méthode la plus basse\nest la plus robuste au bruit",
    transform=ax_sl.transAxes, va='top', fontsize=8,
    bbox=dict(boxstyle='round', facecolor='#EEF4FF', edgecolor='#3A86C8', alpha=0.9))

ax_asp.set_xlabel("Bruit σ_bruit [unités Z]")
ax_asp.set_ylabel("RMSE exposition moyen [°]")
ax_asp.set_title("Exposition", fontweight='bold')
ax_asp.legend(fontsize=10); ax_asp.grid(True)

# Axe secondaire en radians
ax_rad = ax_asp.twinx()
ax_rad.set_ylim(np.radians(ax_asp.get_ylim()[0]),
                np.radians(ax_asp.get_ylim()[1]))
ax_rad.set_ylabel("RMSE exposition [rad]", color='#555555')
ax_rad.tick_params(colors='#555555')

plt.tight_layout()
plt.savefig(f"MC_synthese_{TERRAIN}.png", dpi=120, bbox_inches='tight')
print(f"  → MC_synthese_{TERRAIN}.png")

print("\nTerminé.")
plt.show()
