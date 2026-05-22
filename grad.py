import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def plan(x, y):
    return 0.07*(x - 50) + 0.1*(y - 50)

def plateau(x, y):
    return 5 * np.tanh((x - 40) / 5)

def sinc_card(x, y):
    d = np.sqrt((x - 40)**2 + (y - 50)**2)
    if d == 0:
        return 10.0
    return 10 * np.sin(0.1*d) / (0.1*d)

def double_sin(x, y):
    return 5*np.sin(x/10 + 3*np.sin(y/20)) + 2*np.sin(y/5)

def grad_plan(x, y):
    return np.array([0.07, 0.1])

def grad_plateau(x, y):
    u = (x - 40) / 5
    return np.array([1 - np.tanh(u)**2, 0.0])

def grad_sinc(x, y):
    d = np.sqrt((x - 40)**2 + (y - 50)**2)
    if d == 0:
        return np.array([0.0, 0.0])
    coeff = 10 / d * (np.cos(0.1*d) / d - np.sin(0.1*d) / (0.1 * d**2))
    return coeff * np.array([x - 40, y - 50])

def grad_double_sin(x, y):
    u = x/10 + 3*np.sin(y/20)
    dx = 0.5 * np.cos(u)
    dy = 0.75 * np.cos(u) * np.cos(y/20) + 0.4 * np.cos(y/5)
    return np.array([dx, dy])


fname = 'sin_card.txt'

mnt = np.loadtxt(fname)
x = np.arange(0, 101)
y = np.arange(0, 101)
X, Y = np.meshgrid(x, y)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
col = plt.get_cmap('gist_earth')(mnt)

ax.plot_surface(X, Y, mnt, cmap = 'magma')


plt.show()

#max, min etc

flat4 = [val for sublist in data4 for val in sublist]
min4 = np.amin(flat4)
max4 = np.amax(flat4)
m4=np.mean(flat4)
ec4=np.std(flat4)
print(min4,max4,m4,ec4)

#histogramme

res = plt.hist(flat4, range = (0, 5), bins = 100)

#calcul de pente et d'expositon

import numpy as np
from scipy.ndimage import convolve, generic_filter, gaussian_filter

# =========================
# Utils
# =========================

def pente(fx, fy):
    return np.sqrt(fx**2 + fy**2)

def exposition(fx, fy):
    return np.arctan2(-fx, -fy)

# 1. différentes méthodes pour les pentes


def tpp(z, dx=1.0, dy=1.0):
    """Three Points Plane (TPP)"""
    fx = (np.roll(z, -1, axis=1) - z) / dx
    fy = (np.roll(z, -1, axis=0) - z) / dy
    return fx[:-1, :-1], fy[:-1, :-1]


def fcn(z, dx=1.0, dy=1.0):
    """Four Closest Neighbours (FCN)"""
    fx = (np.roll(z, -1, axis=1) - np.roll(z, 1, axis=1)) / (2 * dx)
    fy = (np.roll(z, -1, axis=0) - np.roll(z, 1, axis=0)) / (2 * dy)
    return fx[1:-1, 1:-1], fy[1:-1, 1:-1]


def evans(z, s=1.0):
    """Evans polynomial fit (returns fx, fy + coeffs)"""

    def compute_coeffs(window):
        z1,z2,z3,z4,z5,z6,z7,z8,z9 = window

        A = (z1+z3+z4+z6+z7+z9)/(6*s**2) - (z2+z5+z8)/(3*s**2)
        B = (z1+z2+z3+z7+z8+z9)/(6*s**2) - (z4+z5+z6)/(3*s**2)
        C = (z3+z7-z1-z9)/(4*s**2)
        D = (z3+z6+z9-z1-z4-z7)/(6*s**2)
        E = (z1+z2+z3-z7-z8-z9)/(6*s**2)

        return np.array([A,B,C,D,E])

    coeffs = generic_filter(z, compute_coeffs, size=(3,3), mode='nearest')
    coeffs = coeffs.reshape(z.shape + (5,))

    D = coeffs[...,3]
    E = coeffs[...,4]

    return D[1:-1,1:-1], E[1:-1,1:-1], coeffs


# =========================
# 2. PENTE + EXPOSITION
# =========================

def pente_exposition(fx, fy):
    pe = pente(fx, fy)
    e = exposition(fx, fy)
    return pe, e


def ecart_tpp(z):
    tpp = tpp(z)
    sin_card = sin_card(x, y)
    f1_diff, f2_diff = deriv_sin_card - tpp


#norme(racine de fx 2 + fy 2) de la différence entre fonction théorique et ce quon trouve avec les différentes normes