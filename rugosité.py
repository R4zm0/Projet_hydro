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
    ax.set_title(f'Rugosité (écart-type local) \n avec une fenêtre de {size}x{size}')

axes1[-1].axis('off')
plt.colorbar(axes1[0].images[0], ax=axes1, orientation='vertical', fraction=0.02, pad=0.01)



gradients = p.gradient_fcn(z)
pente = p._safe_gradient_norm(gradients[..., 0], gradients[..., 1])
exposition = p._aspect(gradients[..., 0], gradients[..., 1])

def rugosite_normale(pente, exposition, size):
    theta = pente
    phi = exposition
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z_comp = np.cos(theta)
    noyau=np.ones((size,size))
    sum_x = convolve(x,noyau, mode='reflect')
    sum_y = convolve(y, noyau, mode='reflect')
    sum_z = convolve(z_comp, noyau, mode='reflect')
    n=size*size
    r = np.sqrt(sum_x**2 + sum_y**2 + sum_z**2)
    return 1 - r / n

fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
axes2 = axes2.flatten()

for index, size in enumerate(taille_fenetre):
    rugosite = rugosite_normale(pente, exposition, size=size)
    ax = axes2[index]
    ax.imshow(rugosite, cmap='viridis')
    ax.set_title(f'Rugosité (dispersion des\n normales locales) \n fenêtre de {size}x{size}')    

axes2[-1].axis('off')
plt.colorbar(axes2[0].images[0], ax=axes2, orientation='vertical', fraction=0.02, pad=0.01)

#différence de ruentre le MNT et sa version lissée
def rugosite_lisse(z,size):
    z_lisse = gaussian_filter(z,  sigma=size/2)
    return np.std(z - z_lisse)

fig3, axes3 = plt.subplots(2, 4, figsize=(16, 8))
axes3 = axes3.flatten()
for  index, size in enumerate(taille_fenetre):
    rugosite = rugosite_lisse(z, size=size)
    ax = axes3[index]
    ax.imshow(rugosite, cmap='viridis')
    ax.set_title(f'Rugosité (différence entre\n MNT et MNT lissé) \n fenêtre de {size}x{size}')

axes3[-1].axis('off')
plt.colorbar(axes3[0].images[0], ax=axes3, orientation='vertical', fraction=0.02, pad=0.01)


plt.show()


