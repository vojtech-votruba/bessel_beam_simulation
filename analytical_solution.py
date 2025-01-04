import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import jv

with open("config.json", encoding="utf-8") as f:
    CONSTANTS = json.load(f)

""" 
    All constants for this simulation are stored in the config.json file,
    the length dimensions used are inputed in millimeters,
    the energy is inputed in joules, length of the pulse is in seconds,
    wavelength is inputed in um. 

    The quantities below need to be calculated from these constants
"""

W0 = CONSTANTS["laser"]["radius"]
TOTAL_ENERGY = CONSTANTS["laser"]["total_energy"]
TOTAL_SURFACE = CONSTANTS["laser"]["total_surface"]
AT = CONSTANTS["laser"]["pulse_length"]
WAVELENGTH = CONSTANTS["laser"]["wavelength"] / 1000
ANGLE = CONSTANTS["axicon"]["angle"] / 180 * np.pi
REGION_SIZE = CONSTANTS["region"]["size"]
XY_PROFILES = [1,2,3] # The distances in mm of the profiles that we want to calculate
# 3 profiles are needed for plotting purposes

ENERGY = TOTAL_ENERGY / TOTAL_SURFACE * np.pi * W0**2
POWER = 2*np.sqrt(np.log(2)/np.pi) * ENERGY/AT
I0 = POWER / (np.pi*W0**2) * 100 # W / cm^2

def intensity(z_: float, rho_) -> float:
    """
        A formula for calculating the intensity of a plane wave which passed through an axicon.
        The variables are , TOTAL_ENERGY, TOTAL_SURFACE,
            rho_; radius variable in cylindrical coordinates
            z_; the z coordinate in cylindrical coordinates.
        Axicons have axial symmetry, therefore the intensity doesn't depend on φ.
        Source: https://ora.ox.ac.uk/objects/uuid:aa7a03d0-2d64-423f-be42-40e01479d312
    """
    k = 2*np.pi / WAVELENGTH
    return 2*np.pi * k * z_ * I0 * ANGLE**2 * jv(0, k * ANGLE * rho_)**2


Z_MAX = W0 / np.tan(ANGLE)
z = np.linspace(0, Z_MAX, 30)
rho = np.linspace(-REGION_SIZE/2, REGION_SIZE/2, 4000)

# XZ Profile
I_field = np.zeros((z.size, rho.size))
for i in range(z.size):
    for j in range(rho.size):
        I_field[i,j] = intensity(z[i], rho[j])

fig,ax = plt.subplots(nrows=2,ncols=2)
pos = ax[0,0].imshow(I_field.T, cmap="hot",aspect="auto",
                     extent=(0,Z_MAX,-REGION_SIZE/2,REGION_SIZE/2))
print("XZ profile done!\n")

fig.colorbar(pos, ax=ax[0,0], label="I (W/cm^2)")
ax[0,0].set_title("xz profile")
ax[0,0].set_xlabel("z (mm)")
ax[0,0].set_ylabel("x (mm)")

# XY profiles
for seq,z0 in enumerate(XY_PROFILES):
    I_field = np.zeros((rho.size, rho.size))
    for i in range(rho.size):
        for j in range(rho.size):
            I_field[i,j] = intensity(XY_PROFILES[seq], np.sqrt(rho[i]**2 + rho[j]**2))

    pos = ax[(seq + 1) // 2, (seq + 1) % 2].imshow(I_field.T,
                                            cmap="hot",
                                            aspect="equal",
                                            extent=(-REGION_SIZE/2,REGION_SIZE/2,-REGION_SIZE/2,REGION_SIZE/2),)
    
    fig.colorbar(pos, ax=ax[(seq + 1) // 2, (seq + 1) % 2], label="I (W/cm^2)")
    ax[(seq + 1) // 2, (seq + 1) % 2].set_title(f"xy profile in {z0} mm")
    ax[(seq + 1) // 2, (seq + 1) % 2].set_xlabel("y (mm)")
    ax[(seq + 1) // 2, (seq + 1) % 2].set_ylabel("x (mm)")

    print(f"XY profile no. {seq+1} done!")

fig.tight_layout(pad=1.0)
plt.style.use("ggplot")

print(f"The plots are generated. However, procede with caution, as the analytical solution assumes infinite energy.")
plt.show()
