import os
import time
import psutil
import json
import argparse
import numexpr as ne
from numpy import angle
from diffractio import np, plt, um, mm, degrees
from diffractio.scalar_sources_XY import Scalar_field_XY
from diffractio.utils_math import get_k
from numpy.lib.scimath import sqrt as csqrt
import pyfftw
from pyfftw.interfaces.scipy_fft import fft2, fftshift, ifft2
from multiprocessing import cpu_count
from matplotlib.colors import LogNorm
from multiprocessing.pool import ThreadPool as Pool

parser = argparse.ArgumentParser()
parser.add_argument("--obstacle", action=argparse.BooleanOptionalAction,
                    default="True", help="Do you want to add the obstacle to the simulation?")
args = parser.parse_args()

nthreads = cpu_count()
pyfftw.interfaces.cache.enable()

with open("config.json", encoding="utf-8") as f:
    CONSTANTS = json.load(f)

def adaptive_mesh(start: float, stop: float, w0: float, fn, delta: float, alpha=0.1):
    """
    Generates an adaptive mesh with quadratic scaling: large distances at the boundaries,
    decreasing toward the center, and switching to constant delta spacing inside the w0 radius.
    
    Parameters:
        start (float): The starting coordinate (left boundary, can be negative)
        stop (float): The ending coordinate (right boundary)
        w0 (float): Radius around the midpoint where constant spacing is used
        delta (float): Base distance for constant spacing inside w0
        alpha (float): Quadratic scaling coefficient for adaptive spacing outside w0
        fn: The functions which scales the grid 
    """
    
    midpoint = (start + stop) / 2
    result_left = [start]
    result_right = [stop]
    x_left, x_right = start, stop
    
    while True:
        dist_left = delta * (1 + alpha * fn(abs(x_left - midpoint) / w0))
        dist_right = delta * (1 + alpha * fn(abs(x_right - midpoint) / w0))

        if (midpoint - x_left <= w0) or (x_right - midpoint <= w0):
            break

        x_left += dist_left
        x_right -= dist_right
        result_left.append(x_left)
        result_right.append(x_right)

    while x_left < midpoint - delta:
        x_left += delta
        x_right -= delta
        result_left.append(x_left)
        result_right.append(x_right)

    if x_left < midpoint:
        result_left.append(midpoint)

    result = result_left + result_right[::-1]
    return np.array(result, dtype=np.float64)

class Scalar_source_XY(Scalar_field_XY):
    """Class for XY scalar sources.

    Parameters:
        x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
        y (numpy.array): linear array wit equidistant positions for y values
        wavelength (float): wavelength of the incident field
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
        self.y (numpy.array): linear array wit equidistant positions for y values
        self.wavelength (float): wavelength of the incident field.
        self.u (numpy.array): (x,z) complex field
        self.info (str): String with info about the simulation
    """

    def __init__(self, x=None, y=None, wavelength=None, info=""):
        super().__init__(x, y, wavelength, info)
        self.type = 'Scalar_source_XY'

    #@profile
    def plane_wave(self, A=1, theta=0 * degrees, phi=0 * degrees, z0=0 * um):
        """Plane wave. self.u = A * exp(1j * k *
                         (self.X * sin(theta) * cos(phi) +
                          self.Y * sin(theta) * sin(phi) + z0 * cos(theta)))

        According to https://en.wikipedia.org/wiki/Spherical_coordinate_system: physics (ISO 80000-2:2019 convention)

        Parameters:
            A (float): maximum amplitude
            theta (float): angle in radians
            phi (float): angle in radians
            z0 (float): constant value for phase shift
        """
        k = 2 * np.pi / self.wavelength
        X = self.X
        Y = self.Y
        self.u = np.complex64(ne.evaluate("A * exp(1j * k *(X * sin(theta) * cos(phi) + Y * sin(theta) * sin(phi) + z0 * cos(theta)))"))


class Scalar_mask_XY(Scalar_field_XY):
    """Class for working with XY scalar masks.

    Parameters:
        x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n`
        y (numpy.array): linear array with equidistant positions for y values
        wavelength (float): wavelength of the incident field
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
        self.y (numpy.array): linear array wit equidistant positions for y values
        self.wavelength (float): wavelength of the incident field.
        self.u (numpy.array): (x,z) complex field
        self.info (str): String with info about the simulation
    """

    def __init__(self, x=None, y=None, wavelength=None, info=""):
        super().__init__(x, y, wavelength, info)
        self.type = 'Scalar_mask_XY'

    def set_amplitude(self, q=1, positive=0, amp_min=0, amp_max=1):
        """makes that the mask has only amplitude.

        Parameters:
            q (int): 0 - amplitude as it is and phase is removed. 1 - take phase and convert to amplitude

            positive (int): 0 - value may be positive or negative. 1 - value is only positive
        """

        amplitude = ne.evaluate("abs(self.u)")
        phase = angle(self.u)

        if q == 0:
            if positive == 0:
                self.u = ne.evaluate("amp_min + (amp_max - amp_min) * amplitude * sign(phase)")
            if positive == 1:
                self.u = ne.evaluate("amp_min + (amp_max - amp_min) * amplitude")
        else:
            if positive == 0:
                self.u = ne.evaluate("amp_min + (amp_max - amp_min) * phase")
            if positive == 1:
                self.u = ne.evaluate("amp_min + (amp_max - amp_min) * (phase)")
    
    def axicon(self,
               r0,
               refractive_index,
               angle,
               radius=0,
               off_axis_angle=0 * degrees,
               reflective=False):
        """Axicon,

        Parameters:
            r0 (float, float): (x0,y0) - center of lens
            refractive_index (float): refraction index
            angle (float): angle of the axicon
            radius (float): radius of lens mask
            off_axis_angle (float) angle when it works off-axis
            reflective (bool): True if the axicon works in reflective mode.

        """

        k = 2 * np.pi / self.wavelength
        x0, y0 = r0

        # distance de la generatriz al eje del cono
        X = self.X
        Y = self.Y
        r = ne.evaluate("sqrt((X - x0)**2 + (Y - y0)**2)")

        # Region de transmitancia
        u_mask = np.zeros(X.shape)
        ipasa = ne.evaluate("r < radius")
        u_mask[ipasa] = 1

        if off_axis_angle == 0 * degrees:
            t_off_axis = 1
        else:
            t_off_axis = np.complex64(ne.evaluate("exp(-1j * k * X * sin(off_axis_angle))"))

        if reflective is True:
            self.u = np.complex64(ne.evaluate("u_mask * exp(-2j * k * r * tan(angle)) * t_off_axis"))

        else:
            self.u = u_mask * \
                np.complex64(ne.evaluate("exp(-1j * k * (refractive_index - 1) * r * tan(angle)) * t_off_axis"))

#@profile
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

    def propagate(field, n=n, k0=k0, k_perp2 = k_perp2, dz=dz):
        Ek = fft2(u, workers=nthreads, overwrite_x=True)
        Ek = fftshift(Ek)
        H = ne.evaluate("Ek*exp(1j * dz * sqrt(n**2 * k0**2 - k_perp2))")

        return fftshift(H)

    HEk = propagate(u)
    result = ifft2(HEk, workers=nthreads, overwrite_x=True)
    
    return result

#@profile
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
    if (z_min <= z <= z_max) and args.obstacle: # Stupid hardcoded optimizer
        refractive_indexes = [np.complex64(1.0 + 0j), np.complex64(1.5 + 7j)]
    else:
        refractive_indexes = [np.complex64(1.0 + 0j)]

    u_final = np.zeros(u.shape, dtype=np.complex64)
    for m, n_m in enumerate(refractive_indexes):
        # print (m, n_m)
        u_temp = PWD_kernel(u, n_m, k0, k_perp2, dz)
        Imz = ne.evaluate("n == n_m")
        u_final += ne.evaluate("Imz * u_temp")

    return u_final

#@profile
def wpm_2d(x, y, u_field,
        num_points: int, wavelength: float, z0: float,
        z_final: float, obstacle,):

    """A modified algorithm from diffractio.py for propagating only with the XY scheme"""
    
    dz = (z_final - z0) / num_points
    k0 = 2 * np.pi / wavelength
    
    kx = get_k(x, flavour='+')
    ky = get_k(y, flavour='+')

    def k_perp(kx=kx, ky=ky):
        KX, KY = np.meshgrid(kx, ky)
        return ne.evaluate("KX**2 + KY**2")

    k_perp2 = k_perp()
    X,Y = np.meshgrid(x, y)

    width_edge = 0.95*(x[-1] - x[0]) / 2
    x_center = (x[-1] + x[0]) / 2
    y_center = (y[-1] + y[0]) / 2

    def filter(X=X, x_center=x_center, width_edge=width_edge, Y=Y, y_center=y_center):
        filter_x = ne.evaluate("exp(-(abs(X - x_center) / width_edge) ** 20)")
        filter_y = ne.evaluate("exp(-(abs(Y - y_center) / width_edge) ** 20)")
        return ne.evaluate("filter_x * filter_y")
    
    filter_function = filter()

    z = z0
    if obstacle: 
        n_field_normal = np.ones(X.shape, dtype=np.complex64)
        n_field_bs = np.ones(X.shape, dtype=np.complex64)
        square = (abs(X) <= CONSTANTS['nozzle']['x_size']/2*mm) * (Y >= - CONSTANTS['nozzle']['dist_y']*mm)
        n_field_bs[square] = np.complex64(1.2 + 7j) # The refractive index of aluminum

        triangle1 = Y <= X + (-CONSTANTS['nozzle']['dist_y'] - CONSTANTS['nozzle']['slope_upper']/2)*mm
        triangle2 = Y <= -X + (-CONSTANTS['nozzle']['dist_y'] - CONSTANTS['nozzle']['slope_upper']/2)*mm
        n_field_bs[triangle1] = np.complex64(1.0 + 0j)
        n_field_bs[triangle2] = np.complex64(1.0 + 0j)
    
    else:
        n_field = np.ones(X.shape, dtype=np.complex64)

    for i in range(num_points):
        if obstacle:
            if CONSTANTS["nozzle"]["dist_z"]*mm <= z <= CONSTANTS["nozzle"]["dist_z"]*mm + CONSTANTS["nozzle"]["z_size"]*mm:
                n_field = n_field_bs
            else:
                n_field = n_field_normal

        kernel = WPM_schmidt_kernel(u_field, n_field, k0, k_perp2,
                dz, z, CONSTANTS["nozzle"]["dist_z"]*mm, CONSTANTS["nozzle"]["dist_z"]*mm + CONSTANTS["nozzle"]["z_size"]*mm) * filter_function
        u_field = ne.evaluate("u_field + kernel")
        print(f"{i+1}/{num_points}", sep='\r', end='\r')
        
        z += dz
        
    return u_field


def main():
    try:
        """
        All of the constants for this simulation are stored in the config.json file.

        The length dimensions used are inputed in millimeters,
        the energy is inputed in joules, length of the pulse is in seconds,
        wavelength is in um.
        """

        W0 = CONSTANTS["laser"]["radius"]
        TOTAL_ENERGY = CONSTANTS["laser"]["total_energy"]
        TOTAL_SURFACE = CONSTANTS["laser"]["total_surface"]
        ENERGY = TOTAL_ENERGY / TOTAL_SURFACE * np.pi * W0**2
        AT = CONSTANTS["laser"]["pulse_length"]
        POWER = 2*np.sqrt(np.log(2)/np.pi) * ENERGY/AT
        I0 = POWER / (np.pi*W0**2)*100 # Intensity of the incoming plane wave in W/cm^2; I ~ E^2
        E0 = np.sqrt(I0) # Amplitude of the electric field in V/cm

        WAVELENGTH = CONSTANTS["laser"]["wavelength"] * um
        SCALE = 1/2
        REGION_SIZE = CONSTANTS["region"]["size"] * SCALE

        Nx = 2 ** int(np.round(np.log2(1000*REGION_SIZE)) - 3)
        Ny = 2 ** int(np.round(np.log2(1000*REGION_SIZE)) - 3)

        print(f"In the x axis using {Nx/REGION_SIZE/1000} px/um with total of {Nx} px")
        print(f"In the y axis using {Ny/REGION_SIZE/1000} px/um with total of {Ny} px")

        x = np.linspace(-REGION_SIZE/2*mm, REGION_SIZE/2*mm, Nx,)
        y = np.linspace(-REGION_SIZE/2*mm, REGION_SIZE/2*mm, Ny,)

        #x = adaptive_mesh(-REGION_SIZE/2*mm, REGION_SIZE/2*mm, fn=np.square, w0=0.5*mm, delta=1*um,) 
        #y = adaptive_mesh(-REGION_SIZE/2*mm, REGION_SIZE/2*mm, fn=np.square, w0=0.5*mm, delta=1*um,) 

        print(f"Available memory: {psutil.virtual_memory().available/1000000000} GB")
        print(f"The number of threads is {nthreads}\n")


        # Initiation of the field
        def initialization(x_, y_, wavelength):
            u0 = Scalar_source_XY(x_, y_, wavelength)
            u0.plane_wave(A=E0, theta=0*degrees)
            print(f"The first plane wave is initialized, we now have: {psutil.virtual_memory().available/1000000000} GB.")

            t0 = Scalar_mask_XY(x_, y_, wavelength)
            t0.axicon(r0=(0*mm, 0*mm),
                    radius=REGION_SIZE/2*mm - 2*SCALE*mm,
                    angle=CONSTANTS["axicon"]["angle"]*degrees,
                    refractive_index=1.51,
                    reflective=True)
            print(f"The axicon is initialized, we now have: {psutil.virtual_memory().available/1000000000} GB.")

            u1 = u0.u
            t1 = t0.u

            return ne.evaluate("u1 * t1")

        start = time.time()
        u_new = initialization(x, y, WAVELENGTH)
        length = time.time() - start

        print(f"The field is initialized, it took {length} sec.")
        print(f"Available memory after initialization: {psutil.virtual_memory().available/1000000000} GB\n")

        """ Field before the obstacle using RS
        u_new = u1.RS(z=(CONSTANTS["nozzle"]["dist_z"]-5)* mm, verbose=True)
        u_cut = u_new#.cut_resample([-10,10], [-10,10], num_points=(128,128), new_field=True)
        u_cut.draw(kind="intensity", title=f"xy profile in {CONSTANTS['nozzle']['dist_z']-5} mm using Rayleigh-Sommerfeld method")
        u_new = u_new.u"""

        start = time.time()
        # Propagation through the obstacle
        PROFILE_LOCATIONS = [(CONSTANTS["nozzle"]["dist_z"]-2)*mm*SCALE, (CONSTANTS["nozzle"]["dist_z"]+2)*mm*SCALE, 50*mm*SCALE, 100*mm*SCALE]


        for seq,location in enumerate(PROFILE_LOCATIONS):
            print(f"Calulating the {seq+1}/{len(PROFILE_LOCATIONS)} profile in the location {location} mm with WPM")

            if seq == 0:
                #distance = location - (CONSTANTS["nozzle"]["dist_z"]-5)* mm
                distance = location
            else:
                distance = location - PROFILE_LOCATIONS[seq-1]

            if ((seq == 0 and (CONSTANTS["nozzle"]["dist_z"]*mm < location < mm*CONSTANTS["nozzle"]["dist_z"] + mm*CONSTANTS["nozzle"]["z_size"]))\
                or ((PROFILE_LOCATIONS[seq-1] < mm*CONSTANTS["nozzle"]["dist_z"]) and (mm*CONSTANTS["nozzle"]["dist_z"] < location))) and args.obstacle:
                Nz = max(1,int(distance/1000))
            else:
                Nz = 1

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

        length = time.time() - start
        print(f"\nCalculating profiles took {length} sec")

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

    except KeyboardInterrupt:
        plt.show()

if __name__ == "__main__":
    main()