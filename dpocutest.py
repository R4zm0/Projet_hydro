from scipy.ndimage import gaussian_filter, generic_filter
import numpy as np
random = np.random.rand(3,3)
print(random)
print(gaussian_filter(random, sigma=5))


def gaussian_footprint(arr, sigma=1.0):
    shape = arr.shape
    axes = [np.arange(s) - (s - 1) / 2.0 for s in shape]
    grids = np.meshgrid(*axes, indexing='ij')
    r2 = sum(g**2 for g in grids)
    kernel = np.exp(-r2 / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def gaussian_weighted_mean(values, sigma=1.0):
    shape = (int(np.round(len(values)**(1/2))),) * 2  # ex: 25 -> (5,5)
    weights = gaussian_footprint(np.ones(shape), sigma=sigma)
    return np.sum(values * weights.ravel())



arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
result = generic_filter(arr, gaussian_weighted_mean, size=3)