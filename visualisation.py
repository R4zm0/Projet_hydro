#cette lib sert à l'affiche.

# les fonctions de la forme Afficher_Qlqchose sont des fonctions d'affichage:
# elles prennent en argument deux mesh grid : X, Y et un np array Z de même dimension que X et Y
# elles returnerons des subplots à afficher, on EVITERA les plt.show() dans ces fonctions, et les plt.Qlqchose() tout court !
# on fera les plt.show () après avoir appeler les fonctions d'affichage pour afficher les subplots côte à côte !
#


# Regle simple :  -----------------------------------------------

#   Dans les fonction :  seulement ax.*, QUE DES ax.Qlqchose() PAS DE PLT.Qlqchose() 

#   Fin du script ailleur dans un autre fichier  : plt.* comme plt.show() ou plt.tight_layout() 
# -----------------------------------------------  ----------------------------------------------- 



# prend exemple sur afficher 2D pour la forme attendu d'une fonction d'affichage

import matplotlib
from matplotlib import scale
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource, Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

def afficher_2D(X, Y, Z, ax=None, title="Z", niveaux=True, colorbar=False, n_levels=10,
                cotes=True, cmap="viridis", Zname="Z", vmin=None, vmax=None, norm=None,
                hillshade=False, vert_exag=1, blend_mode='overlay'):
    """
    Paramètres
    ----------
    X, Y : np.ndarray   meshgrid (2D)
    Z    : np.ndarray   valeurs à afficher (même shape que X, Y)

    ax       : Axes     si fourni, on dessine dessus ; sinon on crée un nouveau subplot
    title    : str      titre du graphique
    Zname    : str      nom de la variable Z pour la colorbar

    niveaux  : bool     afficher les courbes de niveaux
    n_levels : int      nombre de lignes de niveau
    cotes    : bool     afficher les cotes sur les courbes de niveaux
    colorbar : bool     afficher une colorbar

    cmap     : str      colormap matplotlib
    vmin     : float    valeur min de la colormap (None = auto depuis Z)
    vmax     : float    valeur max de la colormap (None = auto depuis Z)
    norm     : permet de changer la methode de normalisation des couleurs, par exemple CenteredNorm pour centrer sur une valeur particulière (ex: 0 pour la bathymétrie), ou LogNorm pour une échelle logarithmique, par defaut Normalize pour une échelle linéaire classique entre vmin et vmax
    hillshade  : bool   activer l'effet d'ombrage (estompage)
    vert_exag  : float  exagération verticale pour le hillshade
                        1  = échelle réelle, ombres douces
                        4  = relief × 4, ombres marquées (bon défaut pour terrains peu accidentés)
                        10 = très dramatique, moindre variation très visible
    blend_mode : str    mode de fusion ombre/couleur pour le hillshade
                        'overlay' : contraste modéré, bon défaut général
                        'soft'    : ombres douces, rendu naturel (recommandé pour bathymétrie)
                        'hsv'     : préserve mieux la teinte de la cmap

    Retourne
    --------
    fig, ax, im : Figure, Axes, et objet image (ScalarMappable si hillshade, AxesImage sinon)
                  im sert principalement à créer une colorbar externe :
                  fig.colorbar(im, ax=[ax1, ax2], label="Altitude [m]")
    """
   
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    Z = np.ma.masked_invalid(Z)

    # bornes auto si non fournies
    _vmin = vmin if vmin is not None else Z.min()
    _vmax = vmax if vmax is not None else Z.max()
    norm = norm if norm is not None else Normalize(vmin=_vmin, vmax=_vmax)
    if hillshade:
        ls = LightSource(azdeg=315, altdeg=45) # direction de la lumière : 315° = nord-ouest, 45° d'altitude c'est apparament la convention en barymétrie
        rgb = ls.shade(Z, cmap=plt.get_cmap(cmap), norm=norm,
                       vert_exag=vert_exag,   # exagération verticale
                       blend_mode=blend_mode)  # mode de fusion ombre/couleur
        ax.imshow(rgb, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower')
        # ScalarMappable : objet "factice" pour que la colorbar sache
        # quelle échelle afficher (ls.shade retourne du RGBA, pas un AxesImage classique)
        im = ScalarMappable(cmap=cmap, norm=norm) # vuq eu im sert globalement à rien à part des color bar, on s'enbête pas c'est pas un vrai im mais ça fait le job pour la colorbar
    
    else:
        im = ax.imshow(Z, extent=(X.min(), X.max(), Y.min(), Y.max()),
                       origin='lower', cmap=cmap, norm=norm)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.get_figure().colorbar(im, cax=cax, label=Zname)


    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if niveaux:
        contours = ax.contour(X, Y, Z, levels=n_levels, colors="black", linewidths=0.5)
        if cotes:
            ax.clabel(contours, inline=True, fontsize=8)

    return fig, ax, im

def afficher_gradient(X, Y, G, ax=None, step=10, color="red", scale=None,
                      ratio_width=0.0002, Pointe_vers_max=False):

    """afficher les gradient sur une figure déjà existante, G doit être de shape (X.shape[0], X.shape[1], 2)
    et correspondre au gradient vectoriel de Z, c'est à dire G[:,:,0] = dZ/dx et G[:,:,1] = dZ/dy"""

    """scale=None   # automatique (width fixe à 0.003 dans ce cas)
    scale=1      # très longues
    scale=10     # moyennes
    scale=100    # courtes

    ratio_width : float   width = ratio_width * scale, garde les proportions quand on change scale
                          0.0005 est un bon point de départ, à ajuster une fois et oublier

    Pointe_vers_max = True : les flèches pointent vers la direction de la pente la plus forte (max de Z)
    Pointe_vers_max = False : les flèches pointent vers la direction de la pente la plus faible (min de Z)
    """
    if not Pointe_vers_max:
        G = -G

    if ax is None:
        _, ax = plt.subplots()

    sl = (slice(None, None, step), slice(None, None, step))
    Gx = G[..., 0]
    Gy = G[..., 1]

    if scale is not None:
        width = ratio_width * scale
    else:
        width = 0.005  # scale auto : width fixe raisonnable

    ax.quiver(X[sl], Y[sl], Gx[sl], Gy[sl],
              color=color, scale=scale,
              width=width,
              headwidth=4, headlength=5)

    return ax


def afficher_histogramme(Z, ax=None, title="Histogramme des profondeurs", Zname="Profondeur [m]",
                          bins=100, density=False, color="steelblue", edgecolor="white", alpha=0.8,
                          show_moyenne=True, show_mediane=True, show_std=True, show_min=True, show_max=True):
    
    """
    Affiche l'histogramme des valeurs d'un MNT.

    Paramètres
    ----------
    Z        : np.ndarray   tableau 2D (ou 1D) des valeurs bathymétriques
    ax       : Axes         si fourni, on dessine dessus ; sinon on crée un nouveau subplot
    title    : str          titre du graphique
    Zname    : str          label de l'axe X (nom + unité de la variable)
    bins     : int          nombre de classes de l'histogramme
    density  : bool         True = densité de probabilité (aire = 1), False = fréquences absolues
    color    : str          couleur de remplissage des barres
    edgecolor: str          couleur des bordures des barres
    alpha    : float        transparence des barres

    Retourne
    --------
    fig, ax : Figure et Axes matplotlib
    """
        
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    valeurs = Z.flatten()
    valeurs = valeurs[np.isfinite(valeurs)]

    moyenne    = valeurs.mean()
    ecart_type = valeurs.std()
    mediane    = np.median(valeurs)
    vmin       = valeurs.min()
    vmax       = valeurs.max()

    p1, p99 = np.percentile(valeurs, [1, 99])
    margin  = (p99 - p1) * 0.1

    n, bins_edges, patches = ax.hist(valeurs, bins=bins, range=(p1 - margin, p99 + margin),
                                     density=density, color=color, edgecolor=edgecolor, alpha=alpha)

    if show_moyenne:
        ax.axvline(moyenne,  color="red",    linestyle="--", linewidth=1.2, label=f"Moyenne : {moyenne:.2f}")
    if show_mediane:
        ax.axvline(mediane,  color="purple", linestyle="--", linewidth=1.2, label=f"Médiane : {mediane:.2f}")
    if show_std:
        ax.axvline(moyenne - ecart_type, color="orange", linestyle=":", linewidth=1.0, label=f"±1σ : {ecart_type:.2f}")
        ax.axvline(moyenne + ecart_type, color="orange", linestyle=":", linewidth=1.0)
    if show_min:
        ax.axvline(vmin, color="gray", linestyle=":", linewidth=1.0, label=f"Min : {vmin:.2f}")
    if show_max:
        ax.axvline(vmax, color="gray", linestyle=":", linewidth=1.0, label=f"Max : {vmax:.2f}")

    ax.set_title(title)
    ax.set_xlabel(Zname)
    ax.set_ylabel("Densité de probabilité" if density else "Fréquence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax