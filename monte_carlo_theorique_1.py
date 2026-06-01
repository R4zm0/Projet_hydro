"""
monte_carlo_theorique.py
========================
Monte Carlo — impact du bruit blanc σ variable sur pente et exposition.
Terrains artificiels uniquement (gradient analytique disponible).

Procédure (sujet §4.2, méthode Lucieer) :
  Pour chaque σ_bruit :
    Répéter N_MC fois :
      1. Z_bruit = Z + b,  b ~ N(0, σ²)
      2. Calculer Evans sur Z_bruit
      3. Enregistrer pente et exposition
    → écart-type des N_MC réalisations = incertitude estimée

Figures :
  MC_cartes_TERRAIN.png   — cartes spatiales d'incertitude par niveau de bruit
  MC_synthese_TERRAIN.png — courbes incertitude vs σ_bruit

Changer : TERRAIN = "plan" | "plateau" | "sinc_card" | "double_sin"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

import pente
import fonction_theoriques as ft

# ─────────────────────────────────────────────
TERRAIN = "sinc_card"
N_MC    = 200
SIGMAS  = [0.0, 0.05, 0.1, 0.2, 0.5]   # niveaux de bruit (unités de Z)
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
Zr = Z[sl, sl]; Xr = X[sl, sl]; Yr = Y[sl, sl]
G_th_r = G_th[sl, sl]
slope_th, aspect_th = pente.slope_aspect(G_th_r[..., 0], G_th_r[..., 1])
ext = (Xr.min(), Xr.max(), Yr.min(), Yr.max())

def diff_angle(a, b):
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi

# Style
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e', 'axes.facecolor': '#0d0d1a',
    'axes.edgecolor': '#444444', 'axes.labelcolor': '#aaaaaa',
    'xtick.color': '#aaaaaa', 'ytick.color': '#aaaaaa',
    'text.color': 'white', 'grid.color': 'white', 'grid.alpha': 0.12,
    'legend.facecolor': '#1a1a2e', 'legend.edgecolor': '#444444',
})

def cb(fig, im, ax, label):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    c = fig.colorbar(im, cax=cax, label=label)
    c.ax.yaxis.set_tick_params(color='#aaaaaa')
    return c

# ===========================================================================
# MONTE CARLO
# ===========================================================================

mc_slope_std   = []   # écart-type des N_MC pentes  → incertitude
mc_aspect_std  = []   # écart-type des N_MC aspects → incertitude
mc_slope_bias  = []   # biais moyen vs analytique
mc_aspect_bias = []

for sigma in SIGMAS:
    print(f"σ = {sigma:.2f}  ({N_MC} tirages)...", end=" ", flush=True)
    slopes_mc  = np.zeros((N_MC, Zr.shape[0], Zr.shape[1]))
    aspects_mc = np.zeros((N_MC, Zr.shape[0], Zr.shape[1]))

    for k in range(N_MC):
        bruit   = np.random.normal(0, sigma, Z.shape)
        G_b, _  = pente.gradient_evans(Z + bruit, s=dx)
        G_b     = G_b[sl, sl]
        sl_b, asp_b = pente.slope_aspect(G_b[..., 0], G_b[..., 1])
        slopes_mc [k] = sl_b
        aspects_mc[k] = asp_b

    # RMSE pixel par pixel vs valeur analytique exacte
    # RMSE² = variance + biais²  → plus complet que std seul
    mc_slope_std .append(np.sqrt(np.mean((slopes_mc - slope_th)**2,  axis=0)))
    mc_aspect_std.append(np.sqrt(np.mean(
        diff_angle(aspects_mc, aspect_th)**2, axis=0)))

    # Biais séparé pour analyse
    mc_slope_bias .append(np.mean(slopes_mc,  axis=0) - slope_th)
    mc_aspect_bias.append(diff_angle(np.mean(aspects_mc, axis=0), aspect_th))

    print(f"RMSE_pente={mc_slope_std[-1].mean():.4f}  "
          f"RMSE_aspect={np.degrees(mc_aspect_std[-1].mean()):.2f}°")

mc_slope_std   = np.array(mc_slope_std)    # RMSE vs analytique (pas std seul)
mc_aspect_std  = np.array(mc_aspect_std)
mc_slope_bias  = np.array(mc_slope_bias)
mc_aspect_bias = np.array(mc_aspect_bias)

# ===========================================================================
# FIGURE 1 — Cartes d'incertitude et de biais
# ===========================================================================

print("\nFigure 1 : cartes...")
fig1 = plt.figure(figsize=(4.5 * len(SIGMAS), 14))
fig1.suptitle(f"Monte Carlo — {TERRAIN}  ({N_MC} tirages)\n"
              "Ligne 1 = RMSE pente vs analytique  ·  "
              "Ligne 2 = RMSE exposition vs analytique  ·  "
              "Ligne 3 = biais pente (mean − analytique)",
              fontsize=12, fontweight='bold')
gs1 = gridspec.GridSpec(3, len(SIGMAS), figure=fig1, hspace=0.5, wspace=0.05)

vmax_sl  = mc_slope_std.max()
vmax_asp = np.degrees(mc_aspect_std.max())
vabs_b   = np.max(np.abs(mc_slope_bias))

for col, sigma in enumerate(SIGMAS):

    # Ligne 0 : incertitude pente
    ax = fig1.add_subplot(gs1[0, col])
    im = ax.imshow(mc_slope_std[col], origin='lower', extent=ext,
                   cmap='hot', vmin=0, vmax=vmax_sl)
    ax.set_title(f"σ_bruit={sigma:.2f}\n"
                 f"RMSE_pente={mc_slope_std[col].mean():.4f}",
                 fontsize=9, fontweight='bold')
    ax.set_xlabel("x [m]")
    if col == 0:
        ax.set_ylabel("① RMSE pente [m/m]",
                      color='#FF8C42', fontweight='bold')
    cb(fig1, im, ax, "σ pente"); ax.grid(True)

    # Ligne 1 : incertitude exposition
    ax = fig1.add_subplot(gs1[1, col])
    im = ax.imshow(np.degrees(mc_aspect_std[col]), origin='lower', extent=ext,
                   cmap='hot', vmin=0, vmax=vmax_asp)
    ax.set_title(f"RMSE_aspect={np.degrees(mc_aspect_std[col].mean()):.2f}°",
                 fontsize=9)
    ax.set_xlabel("x [m]")
    if col == 0:
        ax.set_ylabel("② RMSE exposition [°]",
                      color='#6BCB77', fontweight='bold')
    cb(fig1, im, ax, "σ aspect [°]"); ax.grid(True)

    # Ligne 2 : biais pente
    from matplotlib.colors import CenteredNorm
    ax = fig1.add_subplot(gs1[2, col])
    im = ax.imshow(mc_slope_bias[col], origin='lower', extent=ext,
                   cmap='seismic', norm=CenteredNorm(0))
    ax.set_title(f"Biais={mc_slope_bias[col].mean():+.4f}", fontsize=9)
    ax.set_xlabel("x [m]")
    if col == 0:
        ax.set_ylabel("③ Biais pente\n(num − analytique)",
                      color='#4D9DE0', fontweight='bold')
    cb(fig1, im, ax, "Biais [m/m]"); ax.grid(True)

plt.savefig(f"MC_cartes_{TERRAIN}.png", dpi=110, bbox_inches='tight',
            facecolor=fig1.get_facecolor())
print(f"  → MC_cartes_{TERRAIN}.png")

# ===========================================================================
# FIGURE 2 — Courbes de synthèse
# ===========================================================================

print("Figure 2 : synthèse...")
fig2, (ax_sl, ax_asp) = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle(f"Monte Carlo — Synthèse — {TERRAIN}  ({N_MC} tirages)",
              fontsize=12, fontweight='bold')

sigmas_arr = np.array(SIGMAS)

# Pente
moy_sl = mc_slope_std.mean(axis=(1, 2))
std_sl = mc_slope_std.std (axis=(1, 2))
ax_sl.plot(sigmas_arr, moy_sl, 'o-', color='#FF8C42', lw=2, ms=7,
           label="RMSE moyen (vs analytique)")
ax_sl.fill_between(sigmas_arr, moy_sl - std_sl, moy_sl + std_sl,
                   color='#FF8C42', alpha=0.25, label="±1σ spatiale")
ax_sl.set_xlabel("Bruit σ_bruit [unités Z]")
ax_sl.set_ylabel("RMSE pente [m/m]")
ax_sl.set_title("Pente", fontweight='bold')
ax_sl.legend(fontsize=9); ax_sl.grid(True)

# Annotation à σ=0.1
if 0.1 in SIGMAS:
    i = SIGMAS.index(0.1)
    ax_sl.annotate(f"σ=0.1 → {moy_sl[i]:.4f} m/m",
                   xy=(0.1, moy_sl[i]),
                   xytext=(0.15, moy_sl[i] * 1.4),
                   fontsize=8, color='#FF8C42',
                   arrowprops=dict(arrowstyle='->', color='#FF8C42', lw=1.2))

# Exposition
moy_asp = mc_aspect_std.mean(axis=(1, 2))
std_asp = mc_aspect_std.std (axis=(1, 2))
ax_asp.plot(sigmas_arr, np.degrees(moy_asp), 'o-', color='#6BCB77', lw=2, ms=7,
            label="RMSE moyen (vs analytique)")
ax_asp.fill_between(sigmas_arr,
                    np.degrees(moy_asp - std_asp),
                    np.degrees(moy_asp + std_asp),
                    color='#6BCB77', alpha=0.25, label="±1σ spatiale")
ax_asp.set_xlabel("Bruit σ_bruit [unités Z]")
ax_asp.set_ylabel("RMSE exposition [°]")
ax_asp.set_title("Exposition", fontweight='bold')
ax_asp.legend(fontsize=9); ax_asp.grid(True)

# Axe secondaire en radians
ax_rad = ax_asp.twinx()
ax_rad.set_ylim(np.radians(ax_asp.get_ylim()[0]),
                np.radians(ax_asp.get_ylim()[1]))
ax_rad.set_ylabel("Incertitude [rad]", color='#aaaaaa')
ax_rad.tick_params(colors='#aaaaaa')

plt.tight_layout()
plt.savefig(f"MC_synthese_{TERRAIN}.png", dpi=120, bbox_inches='tight',
            facecolor=fig2.get_facecolor())
print(f"  → MC_synthese_{TERRAIN}.png")

print("\nTerminé.")
plt.show()
