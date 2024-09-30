import psutil
import json
import argparse
from diffractio import np, plt, um, mm, degrees
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.utils_math import get_k
from scipy.fftpack import fft2, fftshift, ifft2
from numpy.lib.scimath import sqrt as csqrt


parser = argparse.ArgumentParser()
parser.add_argument("--obstacle", action=argparse.BooleanOptionalAction,
                    default="True", help="Do you want to add the obstacle to the simulation?")
args = parser.parse_args()

def PWD_kernel(u, n, k0, k_perp2, dz):
    """
    Step for scalar(TE) Plane wave decomposition(PWD) algorithm.

    Parameters:
        u(np.array): field
        n(np.array): refraction index
        k0(float): wavenumber
        k_perp(np.array): transversal k
        dz(float): increment in distances

    Returns:
        (numpy.array): Field at at distance dz from the incident field

    References:
        1. Schmidt, S. et al. Wave - optical modeling beyond the thin - element - approximation. Opt. Express 24, 30188 (2016).

    """
    absorption = 0.00

    Ek = fftshift(fft2(u))
    # H = np.exp(1j * dz * csqrt(n**2 * k0**2 - k_perp2.transpose()) - absorption)
    H = np.exp(1j * dz * csqrt(n**2 * k0**2 - k_perp2) - absorption)

    result = (ifft2(fftshift(H * Ek)))
    return result

def WPM_schmidt_kernel(u, n, k0, k_perp2, dz, z, z_min, z_max):
    """
    Kernel for fast propagation of WPM method

    Parameters:
        u (np.array): fields
        n (np.array): refraction index
        k0 (float): wavenumber
        k_perp2 (np.array): transversal k**2
        dz (float): increment in distances

    References:

        1. M. W. Fertig and K.-H. Brenner, “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.

        2. S. Schmidt et al., “Wave-optical modeling beyond the thin-element-approximation,” Opt. Express, vol. 24, no. 26, p. 30188, 2016.
    """
    if z_min <= z <= z_max: # Stupid hardcoded optimizer
        refractive_indexes = [np.complex64(1.0 + 0j), np.complex64(1.5 + 7j)]
    else:
        refractive_indexes = [np.complex64(1.0 + 0j)]

    u_final = np.zeros_like(u, dtype=np.complex64)
    for m, n_m in enumerate(refractive_indexes):
        # print (m, n_m)
        u_temp = PWD_kernel(u, n_m, k0, k_perp2, dz)
        Imz = (n == n_m)
        u_final = u_final + Imz * u_temp

    return u_final

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
                n_field = np.ones_like(X, dtype=np.complex64)
                square = (abs(X) <= CONSTANTS["nozzle"]["x_size"]/2*mm) * (Y >= - CONSTANTS["nozzle"]["dist_y"]*mm)
                n_field[square] = np.complex64(1.2 + 7j) # The refractive index of aluminum

                triangle1 = (Y <= X + (-CONSTANTS["nozzle"]["dist_y"] - CONSTANTS["nozzle"]["slope_upper"]/2)*mm)
                triangle2 = (Y <= -X + (-CONSTANTS["nozzle"]["dist_y"] - CONSTANTS["nozzle"]["slope_upper"]/2)*mm)
                n_field[triangle1] = np.complex64(1.0 + 0j)
                n_field[triangle2] = np.complex64(1.0 + 0j)

                plt.imshow(abs(n_field)**2,
                        aspect="equal",
                        extent=(-REGION_SIZE/2, REGION_SIZE/2, -REGION_SIZE/2, REGION_SIZE/2),)

                plt.show()

            else:
                n_field = np.ones_like(X, dtype=np.complex64)
        else:
            n_field = np.ones_like(X, dtype=np.complex64)

        u_field[:,:] += WPM_schmidt_kernel(u_field[:, :], n_field[:, :], k0, k_perp2,
                dz, z) * filter_function
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

Nx = int(1000*REGION_SIZE)
Ny = int(1000*REGION_SIZE)

print(f"In the x axis using {Nx/REGION_SIZE/1000} px/um with total of {Nx} px")
print(f"In the y axis using {Ny/REGION_SIZE/1000} px/um with total of {Ny} px")

x = np.linspace(-REGION_SIZE/2*mm, REGION_SIZE/2*mm, Nx, dtype=np.float16)
y = np.linspace(-REGION_SIZE/2*mm, REGION_SIZE/2*mm, Ny, dtype=np.float16)
print(f"Available memory: {psutil.virtual_memory().available/1000000000} GB\n")

# Initiation of the field
def initialization(x_, y_, wavelength):
    u0 = Scalar_source_XY(x_, y_, wavelength)
    u0.plane_wave(A=E0, theta=0*degrees)

    t0 = Scalar_mask_XY(x_, y_, wavelength)
    t0.circle(r0=(0*mm, 0*mm),
            radius=REGION_SIZE/2*mm - 5*SCALE*mm,
            angle=0)

    t1 = Scalar_mask_XY(x_, y_, wavelength)
    t1.axicon(r0=(0*mm, 0*mm),
            radius=REGION_SIZE/2*mm - 2*SCALE*mm,
            angle=CONSTANTS["axicon"]["angle"]*degrees,
            refraction_index=1.51,
            reflective=True)

    return np.complex64((u0 * t0 * t1).u)

u_new = initialization(x, y, WAVELENGTH)
print("The field is initialized")
print(f"Available memory: {psutil.virtual_memory().available/1000000000} GB\n")

""" Field before the obstacle using RS
u_new = u1.RS(z=(CONSTANTS["nozzle"]["dist_z"]-5)* mm, verbose=True)
u_cut = u_new#.cut_resample([-10,10], [-10,10], num_points=(128,128), new_field=True)
u_cut.draw(kind="intensity", title=f"xy profile in {CONSTANTS['nozzle']['dist_z']-5} mm using Rayleigh-Sommerfeld method")
u_new = u_new.u"""

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
