import json
import argparse
from diffractio import np, plt, um, mm, degrees
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.utils_math import get_k
from diffractio.scalar_fields_XY import WPM_schmidt_kernel

parser = argparse.ArgumentParser()
parser.add_argument("--obstacle", action=argparse.BooleanOptionalAction,
                    default="True", help="Do you want to add the obstacle to the simulation?")
args = parser.parse_args()


def wpm_2d(x, y, u_field,
        num_points: int, wavelength: float, z0: float,
        z_final: float, obstacle,):

    """A modified algorithm from diffractio.py for propagating only with the XY scheme"""
    
    dz = (z_final - z0) / num_points
    k0 = 2 * np.pi / wavelength
    
    kx = get_k(x, flavour='+')
    ky = get_k(y, flavour='+')

    KX, KY = np.meshgrid(kx, ky)
    k_perp2 = KX**2 + KY**2

    X,Y = np.meshgrid(x, y)

    width_edge = 0.95*(x[-1] - x[0]) / 2
    x_center = (x[-1] + x[0]) / 2
    y_center = (y[-1] + y[0]) / 2

    filter_x = np.exp(-(np.abs(X[:,:] - x_center) / width_edge)**20)
    filter_y = np.exp(-(np.abs(Y[:,:] - y_center) / width_edge)**20)
    filter_function = filter_x * filter_y

    z = z0
    for i in range(num_points):
        if obstacle:
            if CONSTANTS["nozzle"]["dist_z"]*mm <= z <= CONSTANTS["nozzle"]["dist_z"]*mm + CONSTANTS["nozzle"]["z_size"]*mm:
                n_field = np.ones_like(X, dtype=complex)
                square = (abs(X) <= CONSTANTS["nozzle"]["x_size"]/2*mm) * (Y >= - CONSTANTS["nozzle"]["dist_y"]*mm)
                n_field[square] = 1.2 + 7j # The refractive index of aluminum

                triangle1 = (Y <= X + (-CONSTANTS["nozzle"]["dist_y"] - CONSTANTS["nozzle"]["slope_upper"]/2)*mm)
                triangle2 = (Y <= -X + (-CONSTANTS["nozzle"]["dist_y"] - CONSTANTS["nozzle"]["slope_upper"]/2)*mm)
                n_field[triangle1] = 1.0 + 0j
                n_field[triangle2] = 1.0 + 0j

                plt.imshow(abs(n_field)**2,
                        aspect="equal",
                        extent=(-REGION_SIZE/2, REGION_SIZE/2, -REGION_SIZE/2, REGION_SIZE/2),)

                plt.show()

            else:
                n_field = np.ones_like(X, dtype=complex)
        else:
            n_field = np.ones_like(X, dtype=complex)

        u_field[:,:] += WPM_schmidt_kernel(u_field[:, :], n_field[:, :], k0, k_perp2,
                dz) * filter_function
        print(f"{i+1}/{num_points}", sep='\r', end='\r')
        
        z += dz
        
    return u_field

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
I0 = POWER / (np.pi*W0**2)*100 # Intensity of the incoming plane wave in W/cm^2; I ~ E^2
E0 = np.sqrt(I0) # Amplitude of the electric field in V/cm

WAVELENGTH = CONSTANTS["laser"]["wavelength"] * um
SCALE = 1
REGION_SIZE = CONSTANTS["region"]["size"] * SCALE

Nx = int(100*REGION_SIZE)
Ny = int(100*REGION_SIZE)

print(f"In the x axis using {Nx/REGION_SIZE/1000} px/um with total of {Nx} px")
print(f"In the y axis using {Ny/REGION_SIZE/1000} px/um with total of {Ny} px")

x = np.linspace(-REGION_SIZE/2*mm, REGION_SIZE/2*mm, Nx)
y = np.linspace(-REGION_SIZE/2*mm, REGION_SIZE/2*mm, Ny)

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

# Field before the obstacle using RS
"""u_new = u1.RS(z=(CONSTANTS["nozzle"]["dist_z"]-5)* mm, verbose=True)
u_cut = u_new#.cut_resample([-10,10], [-10,10], num_points=(128,128), new_field=True)
u_cut.draw(kind="intensity", title=f"xy profile in {CONSTANTS['nozzle']['dist_z']-5} mm using Rayleigh-Sommerfeld method")
u_new = u_new.u"""

u_new = u1.u

# Propagation through the obstacle
PROFILE_LOCATIONS = [50*mm*SCALE, 100*mm*SCALE]

for seq,location in enumerate(PROFILE_LOCATIONS):
    print(f"Calulating the {seq+1}/{len(PROFILE_LOCATIONS)} profile with WPM")

    if seq == 0:
        #distance = location - (CONSTANTS["nozzle"]["dist_z"]-5)* mm
        distance = location
    else:
        distance = location - PROFILE_LOCATIONS[seq-1]

    Nz = int(distance/1000)
    print(f"In the z axis using {Nz/distance} px/um with total of {Nz} px")

    if seq == 0:
        #u_new = wpm_2d(x, y, u_new, Nz, WAVELENGTH, (CONSTANTS["nozzle"]["dist_z"]-5)*mm, location, args.obstacle)
        u_new = wpm_2d(x, y, u_new, Nz, WAVELENGTH, 0*mm, location, args.obstacle)

    else:
        u_new = wpm_2d(x, y, u_new, Nz, WAVELENGTH, PROFILE_LOCATIONS[seq-1], location, args.obstacle)
    
    plt.figure()
    #plt.colorbar(plt.cm.ScalarMappable(cmap="hot", norm=None) , label="I (W/cm^2)")
    plt.title(f"xy profile in {location/1000} mm")
    plt.xlabel("y (mm)")
    plt.ylabel("x (mm)")

    plt.imshow(abs(u_new)**2,
        cmap="hot",
        aspect="equal",
        extent=(-REGION_SIZE/2, REGION_SIZE/2, -REGION_SIZE/2, REGION_SIZE/2),)

plt.show()

"""
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

plt.show()"""
