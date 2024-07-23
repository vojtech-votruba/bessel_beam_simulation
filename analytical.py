import numpy as np
from matplotlib import pyplot as plt
from scipy.special import jv

WAVELENGTH = 0.8
ANGLE = 6 * np.pi/180
C = 1
Z_MAX = 135
REGION_SIZE = 60
XY_PROFILES = [20,60,120]

def intensity(z, rho) -> float:
    k = 2*np.pi / WAVELENGTH
    I0 = 1
    return 2*np.pi * k * z * I0 * ANGLE**2 * jv(0, k*ANGLE*rho)**2 

"""xz profile graph"""
z = np.linspace(0, Z_MAX, 15)
rho = np.linspace(-REGION_SIZE/2, REGION_SIZE/2, 1500)
I_field = np.zeros((z.size, rho.size))

for i in range(z.size):
    for j in range(rho.size):
        I_field[i,j] = intensity(z[i], rho[j])

fig,ax = plt.subplots(nrows=2,ncols=2)
ax[0,0].imshow(I_field.T, cmap="hot",aspect="auto") # xz profile

print("xz profile done")

"""xy profiles"""
for seq,z0 in enumerate(XY_PROFILES):
    I_field = np.zeros((rho.size, rho.size))
    for i in range(rho.size):
        for j in range(rho.size):
            I_field[i,j] = intensity(XY_PROFILES[seq], np.sqrt(rho[i]**2 + rho[j]**2))
    ax[(seq + 1) // 2, (seq + 1) % 2].imshow(I_field.T, cmap="hot", aspect="equal")
    print(f"xy profile no. {seq+1} done")

plt.style.use("classic")
plt.show()
