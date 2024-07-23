import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import jv

with open("config.json", encoding="utf-8") as f:
    CONSTANTS = json.load(f)

WAVELENGTH = CONSTANTS["wavelength"] / 1000
ANGLE = CONSTANTS["axicon"]["angle"] / 180 * np.pi
Z_MAX = CONSTANTS["nozzle"]["z_size"] + CONSTANTS["nozzle"]["dist_z"]
REGION_SIZE = CONSTANTS["region"]["size"]/1000
XY_PROFILES = [20,60,120]


def intensity(z_: float, rho_: float) -> float:
    """A function for calculating intensity of a plane wave which passed through an axicon.
    The variables are 
        rho, a radius in cylindrical coordinates
        z, the z coordinate in polar coordinates.
    Source: https://ora.ox.ac.uk/objects/uuid:a31ce72c-3d5e-4ca2-afa3-a38afbb93833
    """
    k = 2*np.pi / WAVELENGTH
    I0 = 1 # Intensity of the incoming plane wave
    return 2*np.pi * k * z_ * I0 * ANGLE**2 * jv(0, k * ANGLE * rho_)**2

z = np.linspace(0, Z_MAX, 30)
rho = np.linspace(-REGION_SIZE/2, REGION_SIZE/2, 300)

# XZ Profile
I_field = np.zeros((z.size, rho.size))
for i in range(z.size):
    for j in range(rho.size):
        I_field[i,j] = intensity(z[i], rho[j])

fig,ax = plt.subplots(nrows=2,ncols=2)
ax[0,0].imshow(I_field.T, cmap="hot",aspect="auto",extent=(0,Z_MAX,-REGION_SIZE/2,REGION_SIZE/2))
print("XZ profile done!")

ax[0,0].set_title("xz profile")
ax[0,0].set_xlabel("z (mm)")
ax[0,0].set_ylabel("x (mm)")

# XY profiles
for seq,z0 in enumerate(XY_PROFILES):
    I_field = np.zeros((rho.size, rho.size))
    for i in range(rho.size):
        for j in range(rho.size):
            I_field[i,j] = intensity(XY_PROFILES[seq], np.sqrt(rho[i]**2 + rho[j]**2))
    ax[(seq + 1) // 2, (seq + 1) % 2].imshow(I_field.T,
                                             cmap="hot",
                                             aspect="equal",
                                             extent=(-REGION_SIZE/2,REGION_SIZE/2,-REGION_SIZE/2,REGION_SIZE/2))
    
    ax[(seq + 1) // 2, (seq + 1) % 2].set_title(f"xy profile in {z0} mm")
    ax[(seq + 1) // 2, (seq + 1) % 2].set_xlabel("y (mm)")
    ax[(seq + 1) // 2, (seq + 1) % 2].set_ylabel("x (mm)")

    print(f"XY profile no. {seq+1} done!")

fig.tight_layout(pad=1.0)
plt.style.use("ggplot")
plt.show()
