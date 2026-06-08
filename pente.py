import numpy as np
from scipy.ndimage import correlate, convolve, generic_filter, gaussian_filter

 

echelle = 1.0  # échelle spatiale (m/pixel) pour les gradients

# =========================
# Utils # tous corrects et testés, à garder tels quels
# =========================

def _safe_gradient_norm(Gx, Gy):
    return np.sqrt(Gx**2 + Gy**2)

def _aspect(Gx, Gy):
    " CONVENTION : DANS LE SENS DE LA DESCENTE : 0 = NORD , pi/2 = EST, -pi/2 = OUEST, ±pi = SUD "
    " CONVENTION : DANS LE SENS DE LA MONTÉ : 0 = SUD , -pi/2 = EST, +pi/2 = OUEST, ±pi = NORD "

    return np.arctan2(-Gx, -Gy)


# =========================
# 1. GRADIENTS # tous corrects et testés, à garder tels quels
# =========================

def gradient_tpp(Z, dx=1.0, dy=1.0):
    """Three Points Plane (TPP)"""
    Z_pad = np.pad(Z, 1, mode='edge')
    Gx = (Z_pad[1:-1, 2:] - Z_pad[1:-1, 1:-1]) / dx
    Gy = (Z_pad[2:, 1:-1] - Z_pad[1:-1, 1:-1]) / dy
    return np.stack([Gx, Gy], axis=-1)

def gradient_fcn(Z, dx=1.0, dy=1.0):
    """Four Closest Neighbours (FCN)"""
    Z_pad = np.pad(Z, 1, mode='edge')
    Gx = (Z_pad[1:-1, 2:] - Z_pad[1:-1, :-2]) / (2 * dx)
    Gy = (Z_pad[2:, 1:-1] - Z_pad[:-2, 1:-1]) / (2 * dy)
    return np.stack([Gx, Gy], axis=-1)



def gradient_evans(Z, s=1.0):
    """Evans polynomial fit"""

    def make_coeff(idx):
        def compute(window):
            z1,z2,z3,z4,z5,z6,z7,z8,z9 = window
            A = (z1+z3+z4+z6+z7+z9)/(6*s**2) - (z2+z5+z8)/(3*s**2)
            B = (z1+z2+z3+z7+z8+z9)/(6*s**2) - (z4+z5+z6)/(3*s**2)
            C = (z3+z7-z1-z9)/(4*s**2)
            D = (z3+z6+z9-z1-z4-z7)/(6*s**2)
            E = (z1+z2+z3-z7-z8-z9)/(6*s**2)
            return [A, B, C, D, E][idx]
        return compute

    A = generic_filter(Z, make_coeff(0), size=(3,3), mode='nearest')
    B = generic_filter(Z, make_coeff(1), size=(3,3), mode='nearest')
    C = generic_filter(Z, make_coeff(2), size=(3,3), mode='nearest')
    Gx = generic_filter(Z, make_coeff(3), size=(3,3), mode='nearest')
    Gy = -generic_filter(Z, make_coeff(4), size=(3,3), mode='nearest')

    coeffs = np.stack([A, B, C, Gx, Gy], axis=-1)  # (N, M, 5)
    return np.stack([Gx, Gy], axis=-1), coeffs

def gradient_evans_methode2(Z, n=1):        #ici on se met sur un voisinage de 3x3, d'où n = 1 (vf np.arange), mais si la courbure est trop importante on peut prendre un n plus grand sinon la méthode des moindres carrées part un peu en n'importe quoi
    size = 2*n+1

    x = np.arange(-n, n+1, 1)
    y = np.arange(-n, n+1, 1)
    X, Y = np.meshgrid(x, y)

    A = np.column_stack([                        #matrice de résolution pour la méthode des moindres carrées (identique pour tous)
        X.ravel()**2,
        Y.ravel()**2,
        X.ravel() * Y.ravel(),
        X.ravel(),
        Y.ravel(),
        np.ones(size**2)
    ])

    def moindre_carre(window):
        coeffs, _, _, _ = np.linalg.lstsq(A, window, rcond=None)
        return coeffs  # [a, b, c, d, e, f]      "coeff du repère local"

    # Un appel par coefficient
    def make_fit(idx):
        def func(window):
            coeffs, _, _, _ = np.linalg.lstsq(A, window, rcond=None)
            return coeffs[idx]
        return func
    
    a = generic_filter(Z, make_fit(0), size=size, mode='nearest')  # coeff x²
    b = generic_filter(Z, make_fit(1), size=size, mode='nearest')  # coeff y²
    c = generic_filter(Z, make_fit(2), size=size, mode='nearest')  # coeff xy
    Gx = generic_filter(Z, make_fit(3), size=size, mode='nearest') # pente x
    Gy = generic_filter(Z, make_fit(4), size=size, mode='nearest') # pente y
    
    coeffs = np.stack([a, b, c, Gx, Gy], axis=-1)
    return np.stack([Gx, Gy], axis=-1), coeffs











    


# =============================================================================
# Helpers interne
# =============================================================================

def _bpi_from_footprint(z: np.ndarray, footprint: np.ndarray,
                        mode: str = 'nearest') -> np.ndarray:
    """
    Calcule le BPI à partir d'un footprint (centre = 0).

    BPI = z_centre - mean(z_voisins)

    La moyenne est calculée via deux corrélations pour gérer les NaN :
      - sum_z  = correlate(z_sans_nan, w)   → somme des valeurs valides
      - sum_v  = correlate(masque_valide, w) → nombre de voisins valides
      - mean   = sum_z / sum_v

    Parameters
    ----------
    z         : MNT 2D (N, M), float. Peut contenir des NaN.
    footprint : kernel 2D, centre = 0. Les valeurs non-nulles désignent
                les voisins à inclure dans la moyenne.
    mode      : gestion des bords (voir scipy.ndimage.correlate).

    Returns
    -------
    bpi : ndarray float, même shape que z.
          NaN là où z est NaN ou où tous les voisins sont NaN.
    """
    z = np.asarray(z, dtype=float)
    w = np.asarray(footprint, dtype=float)

    n_neighbors = w.sum()
    if n_neighbors == 0:
        raise ValueError(
            "Le footprint est vide : aucun voisin inclus dans la moyenne. "
            "Vérifier les paramètres (radius trop petit, secteur trop étroit…)."
        )

    nan_mask = np.isnan(z)

    if nan_mask.any():
        z_fill = np.where(nan_mask, 0.0, z)


        valid  = (~nan_mask).astype(float)

        sum_z = correlate(z_fill, w, mode=mode)  
        sum_v = correlate(valid,  w, mode=mode)  

        with np.errstate(invalid='ignore'):
            mean_nb = np.where(sum_v > 0, sum_z / sum_v, np.nan)

        return np.where(nan_mask, np.nan, z - mean_nb)
    else:
        mean_nb = correlate(z, w / n_neighbors, mode=mode)  # ← corrigé
        return z - mean_nb

 
# =============================================================================
# BPI — voisinages fixes
# =============================================================================
 
def bpi_rectangle(z: np.ndarray, size_y: int = 3, size_x: int = 3,
                  mode: str = 'nearest') -> np.ndarray:
    """
    BPI avec voisinage rectangulaire (centre exclu).
 
    Parameters
    ----------
    z              : MNT 2D.
    size_y, size_x : dimensions de la fenêtre. Valeurs impaires recommandées.
    mode           : gestion des bords (scipy.ndimage.convolve).
    """
    if size_y < 1 or size_x < 1:
        raise ValueError("size_y et size_x doivent être >= 1.")
 
    footprint = np.ones((size_y, size_x), dtype=float)
    footprint[size_y // 2, size_x // 2] = 0.0          # exclure le centre
    return _bpi_from_footprint(z, footprint, mode)
 
 
def bpi_disque(z: np.ndarray, radius: float = 3,
               mode: str = 'nearest') -> np.ndarray:
    """
    BPI avec voisinage circulaire (disque, centre exclu).
 
    Parameters
    ----------
    z      : MNT 2D.
    radius : rayon en pixels.
    mode   : gestion des bords.
    """
    r = int(radius)
    if r < 1:
        raise ValueError("radius doit être >= 1.")
 
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    footprint = ((x**2 + y**2) <= r**2).astype(float)
    footprint[r, r] = 0.0                               # exclure le centre
    return _bpi_from_footprint(z, footprint, mode)
 
 
def bpi_anneau(z: np.ndarray, r_inner: float = 2, r_outer: float = 5,
               mode: str = 'nearest') -> np.ndarray:
    """
    BPI avec voisinage en anneau (couronne circulaire, centre exclu).
 
    Pixels à distance euclidienne dans [r_inner, r_outer].
    Avec r_inner = 0, équivalent à bpi_disque.
 
    Parameters
    ----------
    z              : MNT 2D.
    r_inner        : rayon interne (pixels, >= 0).
    r_outer        : rayon externe (pixels, > r_inner).
    mode           : gestion des bords.
    """
    if r_inner < 0:
        raise ValueError("r_inner doit être >= 0.")
    if r_outer <= 0:
        raise ValueError("r_outer doit être > 0.")
    if r_inner >= r_outer:
        raise ValueError(
            f"r_inner ({r_inner}) doit être strictement inférieur à r_outer ({r_outer})."
        )
 
    ro = int(r_outer)
    y, x = np.ogrid[-ro:ro + 1, -ro:ro + 1]
    dist2 = x**2 + y**2
 
    footprint = ((dist2 >= r_inner**2) & (dist2 <= ro**2)).astype(float)
    footprint[ro, ro] = 0.0                             # exclure le centre (dist=0)
    return _bpi_from_footprint(z, footprint, mode)
 
 
# =============================================================================
# BPI — secteur angulaire
# =============================================================================
 
def _sector_footprint(radius: int, angle_center: float,
                      angle_width: float) -> np.ndarray:
    """
    Construit un footprint en secteur angulaire (centre = 0).
 
    Convention des angles : trigonométrique (0 = Est, π/2 = Nord),
    avec y orienté vers le BAS (convention image/raster standard).
    L'inversion -y dans arctan2 ramène à la convention cartographique.
    """
    r = radius
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    dist2 = x**2 + y**2
 
    # arctan2(-y, x) : convention image (y↓) → convention trigonométrique (y↑)
    angles = np.arctan2(y, x)
    half   = angle_width / 2.0
    diff   = (angles - angle_center + np.pi) % (2 * np.pi) - np.pi
 
    in_sector         = (dist2 > 0) & (dist2 <= r**2) & (np.abs(diff) <= half)
    footprint         = in_sector.astype(float)
    footprint[r, r]   = 0.0                            # centre toujours exclu
    return footprint
 
 
def bpi_sector(z: np.ndarray, radius: float = 5, angle_center: float = 0.0,
               angle_width: float = np.pi / 2,
               mode: str = 'nearest') -> np.ndarray:
    """
    BPI avec voisinage en secteur angulaire fixe.
 
    Parameters
    ----------
    z            : MNT 2D.
    radius       : rayon du secteur (pixels).
    angle_center : direction centrale du secteur en radians.
                   Convention trigonométrique : 0 = Est, π/2 = Nord.
                   (y du raster supposé orienté vers le bas.)
    angle_width  : ouverture angulaire totale du secteur (radians, dans (0, 2π]).
    mode         : gestion des bords.
    """
    r = int(radius)
    if r < 1:
        raise ValueError("radius doit être >= 1.")
    if not (0 < angle_width <= 2 * np.pi):
        raise ValueError("angle_width doit être dans (0, 2π].")
 
    footprint = _sector_footprint(r, angle_center, angle_width)
 
    if footprint.sum() == 0:
        raise ValueError(
            f"Secteur vide : radius={r}, "
            f"angle_width={np.degrees(angle_width):.1f}°. "
            "Augmenter radius ou angle_width."
        )
    return _bpi_from_footprint(z, footprint, mode)
 
 
# =============================================================================
# BPI adaptatif (secteur orienté selon l'aspect local)
# =============================================================================
 
def bpi_sector_adaptive(z: np.ndarray, aspect: np.ndarray,
                        radius: float = 5,
                        angle_width: float = np.pi / 2,
                        mode: str = 'nearest',
                        n_bins: int = 72,
                        aspect_convention: str = 'bearing_rad') -> np.ndarray:
    """
    BPI en secteur orienté pixel par pixel selon l'aspect local.
 
    Stratégie : discrétiser l'aspect en n_bins directions, calculer
    un BPI par convolution pour chaque direction, puis assembler.
    Chaque passe est une convolution C-optimisée → beaucoup plus rapide
    que n_bins passes de generic_filter.
 
    Parameters
    ----------
    z                  : MNT 2D (N, M), float.
    aspect             : carte d'aspect, même shape que z.
    radius             : rayon du secteur (pixels).
    angle_width        : ouverture angulaire totale (radians).
    mode               : gestion des bords.
    n_bins             : nombre de directions discrètes (défaut 72 → pas 5°).
                         Valeurs plus élevées = moins d'erreur angulaire,
                         mais plus de convolutions.
        
    en gros on peut pas calculer un footprint pour chaque angle possible
    on dit on découpe alosrs 360° en nbins dans quel intervalle tombe l'exposition du pixel pour lui appliquer le footprint correspondant !
                         

    aspect_convention  : format de l'aspect fourni :
    
    'bearing_rad' → radians  (-π, π],  Nord=0, sens horaire  ← _aspect() de pente.py
    'geo'         → degrés   [0, 360], Nord=0, sens horaire  (GDAL, GRASS, RichDEM…)
    'geo_rad'     → radians  [0, 2π],  Nord=0, sens horaire
    'trig'        → radians  (-π, π],  Est=0,  sens antihoraire (numpy brut)
 
    Returns
    -------
    bpi_map : ndarray float, même shape que z.
              NaN là où z est NaN ou aspect est NaN.
 
    Notes
    -----
    Erreur angulaire maximale = ±(180 / n_bins)°.
    Avec n_bins=72 → ±2.5°, avec n_bins=36 → ±5°.
    """
    z      = np.asarray(z,      dtype=float)
    aspect = np.asarray(aspect, dtype=float)
 
    if z.shape != aspect.shape:
        raise ValueError(
            f"z et aspect doivent avoir la même shape "
            f"(z: {z.shape}, aspect: {aspect.shape})."
        )
 
    # --- Conversion vers la convention trigonométrique interne [-π, π] ---

    if aspect_convention == 'bearing_rad':
        # Relèvement en radians (-π, π], Nord=0, sens horaire  ← ce que _aspect produit A GARDER PAR DEFAUT
        # Conversion → convention trig interne (Est=0, CCW)
        asp = np.pi / 2.0 - aspect
        asp = (asp + np.pi) % (2 * np.pi) - np.pi

    elif aspect_convention == 'geo':
        # Degrés, Nord=0, horaire → radians, Est=0, antihoraire
        asp = np.deg2rad(aspect)
        asp = np.pi / 2.0 - asp
        asp = (asp + np.pi) % (2 * np.pi) - np.pi
 
    elif aspect_convention == 'geo_rad':
        # Radians, Nord=0, horaire → radians, Est=0, antihoraire
        asp = np.pi / 2.0 - aspect
        asp = (asp + np.pi) % (2 * np.pi) - np.pi
 
    elif aspect_convention == 'trig':
        asp = aspect.copy()
 
    else:
        raise ValueError(
            f"aspect_convention inconnu : '{aspect_convention}'. "
            "Valeurs valides : 'geo', 'geo_rad', 'trig'."
        )
 
    # --- Discrétisation angulaire ---
    bin_edges   = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
 
    valid   = ~np.isnan(asp)
    indices = np.full(asp.shape, -1, dtype=int)          # -1 → NaN d'aspect
    indices[valid] = np.clip(
        np.digitize(asp[valid], bin_edges) - 1, 0, n_bins - 1
    )
 
    # --- BPI par direction (convolution) ---
    result = np.full(z.shape, np.nan, dtype=float)
 
    for i, angle in enumerate(bin_centers):
        mask = (indices == i)
        if not np.any(mask):
            continue
 
        footprint = _sector_footprint(int(radius), angle, angle_width)
        if footprint.sum() == 0:
            continue                                      # secteur vide → passer
 
        bpi_map        = _bpi_from_footprint(z, footprint, mode)
        result[mask]   = bpi_map[mask]
 
    return result

# =========================
# 4. RUGOSITÉ # Pas testé, à modifier / verifier / compléter
# =========================

def roughness_std(z, size=3):
    return generic_filter(z, np.std, size=size)


def roughness_tpi(z, sigma=1.0):
    smooth = gaussian_filter(z, sigma=sigma)
    return np.std(z - smooth)


def roughness_normals(slope, aspect, size=3):
    theta = slope
    phi = aspect

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    def func(window):
        n = len(window)//3
        xs = window[:n]
        ys = window[n:2*n]
        zs = window[2*n:]

        r = np.sqrt(np.sum(xs)**2 + np.sum(ys)**2 + np.sum(zs)**2)
        return 1 - r / n

    stacked = np.concatenate([x.flatten(), y.flatten(), z.flatten()])
    return func(stacked)


# =========================
# 5. COURBURES # Pas testé, à modifier / verifier / compléter/ c'est de la merde d' IA
# =========================

def calculer_courbures(Z, dx=1.0):
    """
    Calcule les 4 courbures d'Evans en chaque pixel.

    Parametres
    ----------
    Z  : np.ndarray (N, M)   MNT
    dx : float               pas spatial (m/pixel)

    Retourne
    --------
    kv    : courbure verticale   (sujet eq.4)
    kh    : courbure horizontale (sujet eq.5)
    kmin  : courbure principale minimale (sujet eq.6) — zones plates
    kmax  : courbure principale maximale (sujet eq.7) — zones plates
    slope : pente ||grad z||
    G     : gradient Evans (N, M, 2)
    """

    Z = np.where(np.isfinite(Z), Z, np.nan)  # Assurer que les NaN sont bien des NaN flottants
    G, coeffs = gradient_evans(Z, s=dx)

    A  = coeffs[..., 0]
    B  = coeffs[..., 1]
    C  = coeffs[..., 2]
    fx = coeffs[..., 3]
    fy = coeffs[..., 4]

    fxx = 2 * A       # d2z/dx2
    fyy = 2 * B       # d2z/dy2
    fxy = C           # d2z/dxdy
    
    p = np.maximum(fx**2 + fy**2, 1e-10)   # norme2 gradient (protege /0)
    q = p + 1

    # Courbure verticale — sujet eq.4
    kv = -(fxx*fx**2 + 2*fxy*fx*fy + fyy*fy**2) / (p * np.sqrt(q**3))

    # Courbure horizontale — sujet eq.5
    kh = -(fxx*fy**2 - 2*fxy*fx*fy + fyy*fx**2) / (p * np.sqrt(q))

    # Courbures principales — sujet eq.6 et 7 (utilisees quand pente = 0)
    kmin = -A - B - np.sqrt((A - B)**2 + C**2)
    kmax = -A - B + np.sqrt((A - B)**2 + C**2)

    slope, _ = _aspect(fx, fy)

    return kv, kh, kmin, kmax, slope, G


def calculer_courbures_evans2(Z, n=1):
    """
    Calcule les 4 courbures d'Evans avec la méthode des moindres carrés en chaque pixel.

    Parametres
    ----------
    Z  : np.ndarray (N, M)   MNT
    dx : float               pas spatial (m/pixel)

    Retourne
    --------
    kv    : courbure verticale   (sujet eq.4)
    kh    : courbure horizontale (sujet eq.5)
    kmin  : courbure principale minimale (sujet eq.6) — zones plates
    kmax  : courbure principale maximale (sujet eq.7) — zones plates
    slope : pente ||grad z||
    G     : gradient Evans (N, M, 2)
    """
    G, coeffs = gradient_evans_methode2(Z, n=n)

    A  = coeffs[..., 0]
    B  = coeffs[..., 1]
    C  = coeffs[..., 2]
    fx = coeffs[..., 3]
    fy = coeffs[..., 4]

    fxx = 2 * A       # d2z/dx2
    fyy = 2 * B       # d2z/dy2
    fxy = C           # d2z/dxdy

    p = np.maximum(fx**2 + fy**2, 1e-10)   # norme2 gradient (protege /0)
    q = p + 1

    # Courbure verticale — sujet eq.4
    kv = -(fxx*fx**2 + 2*fxy*fx*fy + fyy*fy**2) / (p * np.sqrt(q**3))

    # Courbure horizontale — sujet eq.5
    kh = -(fxx*fy**2 - 2*fxy*fx*fy + fyy*fx**2) / (p * np.sqrt(q))

    # Courbures principales — sujet eq.6 et 7 (utilisees quand pente = 0)
    kmin = -A - B - np.sqrt((A - B)**2 + C**2)
    kmax = -A - B + np.sqrt((A - B)**2 + C**2)

    slope, _ = _aspect(fx, fy)

    return kv, kh, kmin, kmax, slope, G


def principal_curvatures(coeffs):
    A = coeffs[...,0]
    B = coeffs[...,1]
    C = coeffs[...,2]

    kmin = -A - B - np.sqrt((A - B)**2 + C**2)
    kmax = -A - B + np.sqrt((A - B)**2 + C**2)

    return kmin, kmax

