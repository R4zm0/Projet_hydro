import numpy as np
from scipy.ndimage import convolve, generic_filter, gaussian_filter

# =========================
# Utils
# =========================

def _safe_gradient_norm(Gx, Gy):
    return np.sqrt(Gx**2 + Gy**2)

def _aspect(Gx, Gy):
    return np.arctan2(-Gx, -Gy)


# =========================
# 1. GRADIENTS
# =========================

def gradient_tpp(Z, dx=1.0, dy=1.0):
    """Three Points Plane (TPP)"""
    Z_pad = np.pad(Z, 1, mode='edge')
    Gx = (np.roll(Z_pad, -1, axis=1) - Z_pad) / dx
    Gy = (np.roll(Z_pad, -1, axis=0) - Z_pad) / dy
    return np.stack([Gx[1:-1, 1:-1], Gy[1:-1, 1:-1]], axis=-1)


def gradient_fcn(Z, dx=1.0, dy=1.0):
    """Four Closest Neighbours (FCN)"""
    Z_pad = np.pad(Z, 1, mode='edge')          # (N+2, M+2)
    print("shape de roll-1 : ", np.roll(Z_pad, -1, axis=1).shape)
    print("shape de roll+1 : ", np.roll(Z_pad, 1, axis=1).shape)
    Gx = (np.roll(Z_pad, -1, axis=1) - np.roll(Z_pad, 1, axis=1)) / (2 * dx)
    Gy = (np.roll(Z_pad, -1, axis=0) - np.roll(Z_pad, 1, axis=0)) / (2 * dy)

    return np.stack([Gx[1:-1, 1:-1], Gy[1:-1, 1:-1]], axis=-1)  # → (N, M, 2)

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




# =========================
# 3. BPI
# =========================
# =========================
# Utils BPI
# =========================
 
def _center_idx_in_footprint(footprint):
    """
    Retourne l'index du pixel central dans la fenêtre compressée passée
    par generic_filter (seuls les pixels True du footprint sont fournis).
    """
    cy, cx = np.array(footprint.shape) // 2
    flat_true = np.flatnonzero(footprint)
    center_flat = np.ravel_multi_index((cy, cx), footprint.shape)
    return int(np.searchsorted(flat_true, center_flat))
 
 
def _bpi_func(center_idx):
    """Générateur de fonction BPI : centre - moyenne des voisins."""
    def func(window):
        center = window[center_idx]
        neighbors = np.delete(window, center_idx)
        return center - np.mean(neighbors) if len(neighbors) > 0 else 0.0
    return func
 
 
# =========================
# 3. BPI calcul 
# =========================
 
def bpi_rectangle(z, size_y=3, size_x=3, mode='nearest'):
    """
    BPI avec voisinage rectangulaire.
    size_y, size_x : dimensions de la fenêtre (impairs recommandés).
    """
    footprint = np.ones((size_y, size_x), dtype=bool)
    cidx = _center_idx_in_footprint(footprint)
    return generic_filter(z, _bpi_func(cidx), footprint=footprint, mode=mode)
 
 
def bpi_disk(z, radius=3, mode='nearest'):
    """
    BPI avec voisinage circulaire (disque).
    Tous les pixels à distance <= radius du centre.
    """
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    footprint = (x**2 + y**2) <= r**2
    cidx = _center_idx_in_footprint(footprint)
    return generic_filter(z, _bpi_func(cidx), footprint=footprint, mode=mode)
 
 
def bpi_annulus(z, r_inner=2, r_outer=5, mode='nearest'):
    """
    BPI avec voisinage en anneau (couronne circulaire).
    Pixels à distance dans [r_inner, r_outer].
    Le centre lui-même est exclu du voisinage.
 
    Notes
    -----
    Avec r_inner=0 → équivalent disque.
    Classiquement utilisé pour capturer la position à grande échelle
    en ignorant le voisinage immédiat.
    """
    ro = int(r_outer)
    y, x = np.ogrid[-ro:ro + 1, -ro:ro + 1]
    dist2 = x**2 + y**2
    ring = (dist2 >= r_inner**2) & (dist2 <= ro**2)
 
    # Le centre doit être dans le footprint pour que generic_filter
    # le passe dans la fenêtre — on le retire ensuite dans func.
    cy, cx = ro, ro
    ring[cy, cx] = True
    footprint = ring
 
    cidx = _center_idx_in_footprint(footprint)
    return generic_filter(z, _bpi_func(cidx), footprint=footprint, mode=mode)
 
 
def bpi_sector(z, radius=5, angle_center=0.0, angle_width=np.pi / 2,
               mode='nearest'):
    """
    BPI avec voisinage en secteur angulaire.
 
    Parameters
    ----------
    radius       : rayon du secteur (pixels)
    angle_center : direction centrale (radians, convention trigonométrique
                   0 = Est, π/2 = Nord)
    angle_width  : ouverture angulaire totale du secteur (radians)
    mode         : gestion des bords (voir scipy.ndimage)
 
    Notes
    -----
    Pour orienter le secteur vers l'aspect d'un versant, calculer
    angle_center pixel par pixel et appeler bpi_sector_adaptive().
    """
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    dist2 = x**2 + y**2
 
    # Angles dans la convention image (y vers le bas → inversion)
    angles = np.arctan2(-y, x)
 
    # Différence angulaire ramenée dans [-π, π]
    half = angle_width / 2.0
    diff = (angles - angle_center + np.pi) % (2 * np.pi) - np.pi
 
    in_sector = (dist2 > 0) & (dist2 <= r**2) & (np.abs(diff) <= half)
 
    # Inclure le centre
    cy, cx = r, r
    in_sector[cy, cx] = True
    footprint = in_sector
 
    cidx = _center_idx_in_footprint(footprint)
    return generic_filter(z, _bpi_func(cidx), footprint=footprint, mode=mode)
 
 
def bpi_sector_adaptive(z, aspect, radius=5, angle_width=np.pi / 2,
                        mode='nearest'):
    """
    BPI en secteur orienté pixel par pixel selon l'aspect local.
    Revient à calculer bpi_sector pour chaque angle unique présent
    dans la carte d'aspect (discrétisée).
 
    Parameters
    ----------
    z            : MNT (N, M)
    aspect       : carte d'aspect en radians (même shape que z)
    radius, angle_width, mode : idem bpi_sector
    """
    n_bins = 36  # résolution angulaire : pas de 10°
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    indices = np.digitize(aspect, bin_edges) - 1
    indices = np.clip(indices, 0, n_bins - 1)
 
    result = np.zeros_like(z, dtype=float)
    for i, angle in enumerate(bin_centers):
        mask = (indices == i)
        if not np.any(mask):
            continue
        bpi_map = bpi_sector(z, radius=radius, angle_center=angle,
                             angle_width=angle_width, mode=mode)
        result[mask] = bpi_map[mask]
 
    return result

# =========================
# 4. RUGOSITÉ
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
# 5. COURBURES
# =========================

def curvatures(fx, fy, coeffs):
    A = coeffs[...,0]
    B = coeffs[...,1]
    C = coeffs[...,2]

    fxx = 2*A
    fyy = 2*B
    fxy = C

    p = fx**2 + fy**2
    q = p + 1

    eps = 1e-8
    p = np.maximum(p, eps)

    kv = -(fxx*fx**2 + 2*fxy*fx*fy + fyy*fy**2) / (p * np.sqrt(q**3))
    kh = -(fxx*fy**2 - 2*fxy*fx*fy + fyy*fx**2) / (p * np.sqrt(q))

    return kv, kh


def principal_curvatures(coeffs):
    A = coeffs[...,0]
    B = coeffs[...,1]
    C = coeffs[...,2]

    kmin = -A - B - np.sqrt((A - B)**2 + C**2)
    kmax = -A - B + np.sqrt((A - B)**2 + C**2)

    return kmin, kmax