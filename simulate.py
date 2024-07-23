import json
import argparse
from XYZ_masks.scalar_masks_XYZ import Scalar_mask_XYZ
from diffractio import np, plt, um, mm, degrees
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY

parser = argparse.ArgumentParser()
parser.add_argument("--obstacle", action=argparse.BooleanOptionalAction, 
                    default="True", help="Do you want to add the obstacle to the simulation?")
args = parser.parse_args()

"""
All constants for this simulation are stored in a .json file
"""
with open("config.json", encoding="utf-8") as f:
    CONSTANTS = json.load(f)

x = np.linspace(-CONSTANTS["region"]["size"]/2*um, CONSTANTS["region"]["size"]/2*um, 1000)
y = np.linspace(-CONSTANTS["region"]["size"]/2*um, CONSTANTS["region"]["size"]/2*um, 1000)
z = np.linspace(0*um, (CONSTANTS["nozzle"]["z_size"]+CONSTANTS["nozzle"]["dist_z"]+5)*um, 15)
WAVELENGTH = CONSTANTS["wavelength"] * um
XY_PROFILES = [20,60,120]

u0 = Scalar_source_XY(x, y, WAVELENGTH)
u0.plane_wave(A=1, theta=0*degrees)

t0 = Scalar_mask_XY(x, y, WAVELENGTH)
t0.axicon(r0=(0*um, 0*um),
          radius=CONSTANTS["region"]["size"]/2*um,
          angle=CONSTANTS["axicon"]["angle"]*degrees,
          refraction_index=1.51,
          reflective=True)

uxyz = Scalar_mask_XYZ(x, y, z, WAVELENGTH)

if args.obstacle:
    # Upper, declined part of the nozzle
    uxyz.prism(r0=((CONSTANTS["nozzle"]["dist_x"] - CONSTANTS["nozzle"]["slope_height"])*um,
                            0,
                            (CONSTANTS["nozzle"]["z_size"]/2 + CONSTANTS["nozzle"]["dist_z"])*um),
                            length = CONSTANTS["nozzle"]["z_size"]*um,
                            refractive_index=1.3+7j,
                            height=CONSTANTS["nozzle"]["slope_height"]*um,
                            upper_base=CONSTANTS["nozzle"]["slope_upper"]*um,
                            lower_base=CONSTANTS["nozzle"]["slope_lower"]*um)

    # Lower part of the nozzle
    uxyz.square(r0=((-(CONSTANTS["region"]["size"]/4) + CONSTANTS["nozzle"]["dist_x"] - CONSTANTS["nozzle"]["slope_height"])*um, 0*um,
                (CONSTANTS["nozzle"]["z_size"]/2 + CONSTANTS["nozzle"]["dist_z"])*um), # the X and Y coordinates are inverted for some reason.
                length=((CONSTANTS["region"]["size"]/2 + CONSTANTS["nozzle"]["dist_x"])*um + 5*um, 
                        CONSTANTS["nozzle"]["y_size"]*um,
                        CONSTANTS["nozzle"]["z_size"]*um),
                refractive_index=1.3+7j, # Approximately refractive index of aluminum
                angles=(0*degrees, 0*degrees, 0*degrees),
                rotation_point=0)

uxyz.incident_field(u0*t0)
uxyz.clear_field()
uxyz.WPM(verbose=True, has_edges=True)

uxyz.draw_XZ(y0=0, kind='intensity',
             logarithm=1e1, 
             normalize=None, 
             colorbar_kind='vertical', 
             draw_borders = True,)

for z in XY_PROFILES:
    uxyz.draw_XY(z0=z*um, kind='intensity',
                 logarithm=1e1, 
                 normalize=None,) 
plt.show()
