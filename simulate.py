import json
import argparse
from diffractio import np, plt, um, mm, degrees
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from XYZ_masks.scalar_masks_XYZ import Scalar_mask_XYZ
from wpm import wpm_2d 


parser = argparse.ArgumentParser()
parser.add_argument("--obstacle", action=argparse.BooleanOptionalAction,
                    default="True", help="Do you want to add the obstacle to the simulation?")
args = parser.parse_args()

"""
All constants for this simulation are stored in the config.json file.

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
I0 = POWER / (np.pi*W0**2)*100 # intensity of the incoming plane wave in W/cm^2; I = 1/2 ϵ c E^2
E0 = np.sqrt(2*I0 / (3e8 * 8.9e-12)) # amplitude of the electric field in V/cm
E0 = 1

WAVELENGTH = CONSTANTS["laser"]["wavelength"] * um
REGION_SIZE = CONSTANTS["region"]["size"]/100

Nx = int(1000*REGION_SIZE)
Ny = int(1000*REGION_SIZE)

print(f"In the x axis using {Nx/REGION_SIZE/1000} px/um with total of {Nx} px")
print(f"In the y axis using {Ny/REGION_SIZE/1000} px/um with total of {Ny} px")

x = np.linspace(-REGION_SIZE/2*mm, REGION_SIZE/2*mm, Nx)
y = np.linspace(-REGION_SIZE/2*mm, REGION_SIZE/2*mm, Ny)

u0 = Scalar_source_XY(x, y, WAVELENGTH)
u0.plane_wave(A=E0, theta=0*degrees)

t0 = Scalar_mask_XY(x, y, WAVELENGTH)
t0.circle(r0=(0*mm, 0*mm),
          radius=REGION_SIZE/2*mm - 5/50*mm,
          angle=0)

t1 = Scalar_mask_XY(x, y, WAVELENGTH)
t1.axicon(r0=(0*mm, 0*mm),
        radius=REGION_SIZE/2*mm - 2/50*mm,
        angle=CONSTANTS["axicon"]["angle"]*degrees,
        refraction_index=1.51,
        reflective=True)

u1 = u0 * t0 * t1

# Field before the obstacle
u2 = u1.RS(z=(CONSTANTS["nozzle"]["dist_z"]-5)* mm, verbose=True)
u2.draw(kind="intensity")

# Propagation through the obstacle
u3 = wpm_2d(x, y, u2.u, 10*mm, wavelength=WAVELENGTH, z=40*mm)
fig,ax = plt.subplots()

ax.imshow(abs(u3)**2,
        cmap="hot",
        aspect="equal",
        extent=(-REGION_SIZE/2,REGION_SIZE/2,-REGION_SIZE/2,REGION_SIZE/2),)

plt.show()
breakpoint()




Z_SIZE = CONSTANTS["nozzle"]["z_size"] + 5
Nz = int(np.round(Z_SIZE))
print(f"In the z axis using {Nz/Z_SIZE/1000} px/um with total of {Nz} px")
z = np.linspace((CONSTANTS["nozzle"]["dist_z"]-5)*mm, (CONSTANTS["nozzle"]["dist_z"]+Z_SIZE)*mm, Nz)




uxyz = Scalar_mask_XYZ(x, y, z, WAVELENGTH)

if args.obstacle:
    # Upper, declined part of the nozzle
    uxyz.prism(r0=(0,(CONSTANTS["nozzle"]["dist_x"] - CONSTANTS["nozzle"]["slope_height"])*mm,
                            (CONSTANTS["nozzle"]["z_size"]/2 + CONSTANTS["nozzle"]["dist_z"])*mm),
                            length = CONSTANTS["nozzle"]["z_size"]*mm,
                            refractive_index=1.2+7j, # Approximately the refractive index of aluminum
                            height=CONSTANTS["nozzle"]["slope_height"]*mm,
                            upper_base=CONSTANTS["nozzle"]["slope_upper"]*mm,
                            lower_base=CONSTANTS["nozzle"]["slope_lower"]*mm)

    # Lower part of the nozzle
    uxyz.square(r0=(0,(-(REGION_SIZE/4) + CONSTANTS["nozzle"]["dist_x"] - CONSTANTS["nozzle"]["slope_height"])*mm,
                (CONSTANTS["nozzle"]["z_size"]/2 + CONSTANTS["nozzle"]["dist_z"])*mm), # the X and Y coordinates are inverted for some reason.
                length=(CONSTANTS["nozzle"]["y_size"]*mm,
                        (REGION_SIZE/2 + CONSTANTS["nozzle"]["dist_x"])*mm + 5*mm,
                        CONSTANTS["nozzle"]["z_size"]*mm),
                refractive_index=1.2+7j, # Approximately the refractive index of aluminum
                angles=(0*degrees, 0*degrees, 0*degrees),
                rotation_point=0)

uxyz.incident_field(u2)
uxyz.clear_field()
uxyz.WPM(verbose=True, has_edges=True)

uxyz.draw_XZ(y0=0, kind='intensity',
            normalize=False,
            colorbar_kind='vertical',
            draw_borders = True,)

XY_PROFILES = [50, 130]

for z in XY_PROFILES:
    #XY_cut = uxyz.cut_resample([-10,10], [-10,10], num_points=(128,128,128), new_field=True)
    uxyz.draw_XY(z0=z*mm, kind='intensity',
                title=f"XY profile in {z} mm",
                normalize=False,
                has_colorbar=True,) # Doesn't work for some stupid reason

plt.show()
