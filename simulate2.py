import json
import argparse
from diffractio import np, plt, um, mm, degrees
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ

parser = argparse.ArgumentParser()
parser.add_argument("--obstacle", action=argparse.BooleanOptionalAction,
                    default="True", help="Do you want to add the obstacle to the simulation?")
args = parser.parse_args()

"""
All of the constants for this simulation are stored in the config.json file.

The length dimensions used are inputed in millimeters,
the energy is inputed in joules, length of the pulse is in seconds,
wavelength is in um.
"""

with open("config.json", encoding="utf-8") as f:
    CONSTANTS = json.load(f)

W0 = CONSTANTS["laser"]["radius"]
TOTAL_ENERGY = CONSTANTS["laser"]["total_energy"]
TOTAL_SURFACE = CONSTANTS["laser"]["total_surface"]
ENERGY = TOTAL_ENERGY / TOTAL_SURFACE * np.pi * W0**2
AT = CONSTANTS["laser"]["pulse_length"]
POWER = 2*np.sqrt(np.log(2)/np.pi) * ENERGY/AT
I0 = POWER / (np.pi*W0**2)*100 # Intensity of the incoming plane wave in W/cm^2; I = 1/2 ϵ c E^2
E0 = np.sqrt(2*I0 / (3e8 * 8.9e-12)) # Amplitude of the electric field in V/cm

WAVELENGTH = CONSTANTS["laser"]["wavelength"] * um
SCALE = 1/50
REGION_SIZE = CONSTANTS["region"]["size"] * SCALE

Nx = int(1000*REGION_SIZE)
Ny = int(1000*REGION_SIZE)

print(f"In the x axis using {Nx/REGION_SIZE/1000} px/um with total of {Nx} px")
print(f"In the y axis using {Ny/REGION_SIZE/1000} px/um with total of {Ny} px")

x = np.linspace(-REGION_SIZE/2*mm, REGION_SIZE/2*mm, Nx)
y = np.linspace(-REGION_SIZE/2*mm, REGION_SIZE/2*mm, Ny)
z = np.linspace(0, 105*mm, 105)

# Initiation of the field
u0 = Scalar_source_XY(x, y, WAVELENGTH)
u0.plane_wave(A=E0, theta=0*degrees)

t0 = Scalar_mask_XY(x, y, WAVELENGTH)
t0.circle(r0=(0*mm, 0*mm),
          radius=REGION_SIZE/2*mm - 5*SCALE*mm,
          angle=0)

t1 = Scalar_mask_XY(x, y, WAVELENGTH)
t1.axicon(r0=(0*mm, 0*mm),
        radius=REGION_SIZE/2*mm - 2*SCALE*mm,
        angle=CONSTANTS["axicon"]["angle"]*degrees,
        refraction_index=1.51,
        reflective=True)

u1 = u0 * t0 * t1
uxyz = Scalar_mask_XYZ(x, y, z, WAVELENGTH)

uxyz.incident_field(u1)
uxyz.clear_field()
uxyz.WPM(verbose=True, has_edges=True)

uxyz.draw_XZ()

XY_PROFILES = [1, 2, 3]

for z in XY_PROFILES:
    XY_cut = uxyz.cut_resample([-10,10], [-10,10], num_points=(128,128,128), new_field=True)
    XY_cut.draw_XY(z0=z*mm, kind='intensity',
                title=f"XY profile in {z} mm",
                normalize=False,
                has_colorbar=True,) # Doesn't work for some stupid reason

plt.show()
