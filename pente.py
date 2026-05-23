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

def bpi(z, size=3):
    def func(window):
        center = window[len(window)//2]
        return center - np.mean(window)

    return generic_filter(z, func, size=size)


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