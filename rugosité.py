import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generic_filter, gaussian_filter
import pente as p

taille_fenetre = [3, 5, 7, 9, 11, 13, 15]

z = np.loadtxt('txt/reels/bertheaume_z.txt')
print(z.shape)

def rugosite_std(z, size):
    ecarttype = generic_filter(z, np.nanstd, size=size) 
    return ecarttype

fig1, axes1 = plt.subplots(2, 4, figsize=(16, 8))
axes1 = axes1.flatten()

for index, size in enumerate(taille_fenetre):
    rugosite = rugosite_std(z, size=size)
    ax = axes1[index]
    ax.imshow(rugosite, cmap='viridis')
    ax.set_title(f'Rugosité (écart-type local) avec une fenêtre de {size}x{size}')

axes1[-1].axis('off')
plt.tight_layout()
plt.show()


gradients = p.gradient_fcn(z)
pente = p.safe_gradient_norm(gradients[..., 0], gradients[..., 1])
exposition = p.aspect(gradients[..., 0], gradients[..., 1])

def rugosite_normale(pente, exposition):
    theta = pente
    phi = exposition
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    sum_x = convolve(x, mode='constant', cval=0.0)
    sum_y = convolve(y, mode='constant', cval=0.0)
    sum_z = convolve(z, mode='constant', cval=0.0)
    r = np.sqrt(np.sum(sum_x)**2 + np.sum(sum_y)**2 + np.sum(sum_z)**2)
    return 1 - r / n

fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
axes2 = axes2.flatten()

for index, size in enumerate(taille_fenetre):
    rugosite = rugosite_normale(pente, exposition, size=size)
    ax = axes2[index]
    ax.imshow(rugosite, cmap='viridis')
    ax.set_title(f'Rugosité (dispersion des normales locales) avec une fenêtre de {size}x{size}')    

axes2[-1].axis('off')
plt.tight_layout()
plt.show()