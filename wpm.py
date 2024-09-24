import numpy as np
from diffractio.utils_math import get_k
from diffractio.scalar_fields_XY import WPM_schmidt_kernel


def wpm_2d(x, y, u_field,
        dz: float, wavelength: float,
        z: float, obstacle=False,):
    
    """
    An original implementation of 2D WPM by Vojtech Votruba
    """

    k0 = 2 * np.pi / wavelength
    
    kx = get_k(x, flavour='+')
    ky = get_k(y, flavour='+')

    KX, KY = np.meshgrid(kx, ky)
    k_perp2 = KX**2 + KY**2

    X,Y = np.meshgrid(x, y)

    
    if obstacle:
        ...
    else:
        n_field = np.ones_like(X)


    width_edge = 0.95*(x[-1] - x[0]) / 2
    x_center = (x[-1] + x[0]) / 2
    y_center = (y[-1] + y[0]) / 2

    filter_x = np.exp(-(np.abs(X[:,:]-x_center) / width_edge)**80)
    filter_y = np.exp(-(np.abs(Y[:,:]-y_center) / width_edge)**80)
    filter_function = filter_x * filter_y


    num_steps = int(np.round(z/dz)) + 1

    for i in range(num_steps):
        u_field[:,:] += WPM_schmidt_kernel(u_field[:, :], n_field[:, :], k0, k_perp2,
                dz) * filter_function

        print(f"{i}/{num_steps-1}", sep='\r', end='\r')

    return u_field

if __name__ == "__main__":
    print("idk")
    