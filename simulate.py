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
from pyfftw.interfaces.scipy_fftpack import fft2, fftshift, ifft2, convolve
from multiprocessing import Pool, cpu_count

num_threads = cpu_count()
pyfftw.config.NUM_THREADS = num_threads
pyfftw.interfaces.cache.enable()

def fft_per_row(row):
    return fft2(row, overwrite_x=True)

def ifft_per_row(row):
    return ifft2(row, overwrite_x=True)

def fftshift_per_row(row):
    return fftshift(row)

def fft_parallel_2d(matrix):
    with Pool(processes=num_threads) as pool:
        fft_rows = pool.map(fft_per_row, matrix)

    return np.array(fft_rows)

def ifft_parallel_2d(matrix):
    with Pool(processes=num_threads) as pool:
        ifft_rows = pool.map(ifft_per_row, matrix)

    return np.array(ifft_rows)

def fftshift_parallel_2d(matrix):
    with Pool(processes=num_threads) as pool:
        shift_rows = pool.map(fftshift_per_row, matrix)

    return np.array(shift_rows)

parser = argparse.ArgumentParser()
parser.add_argument("--obstacle", action=argparse.BooleanOptionalAction,
                    default="True", help="Do you want to add the obstacle to the simulation?")
args = parser.parse_args()

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

    @profile
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
        self.u = A * ne.evaluate("exp(1j * k *(X * sin(theta) * cos(phi) + Y * sin(theta) * sin(phi) + z0 * cos(theta)))")


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

    def set_phase(self, q=1, phase_min=0, phase_max=np.pi):
        """Makes the mask as phase,
            q=0: Pass amplitude to 1.
            q=1: amplitude pass to phase
            """

        amplitude = np.abs(self.u)
        phase = angle(self.u)

        if q == 0:
            self.u = ne.evaluate("exp(1j * phase)")
        if q == 1:
            self.u = ne.evaluate("exp(1j * (phase_min + (phase_max - phase_min) * amplitude))")

    def area(self, percentage):
        """Computes area where mask is not 0

        Parameters:
            percentage_maximum (float): percentage from maximum intensity to compute

        Returns:
            float: area (in um**2)

        Example:
            area(percentage=0.001)
        """

        intensity = np.abs(self.u)**2
        max_intensity = intensity.max()
        num_pixels_1 = ne.eval("sum(sum(intensity > max_intensity * percentage))")
        num_pixels = len(self.x) * len(self.y)
        delta_x = self.x[1] - self.x[0]
        delta_y = self.y[1] - self.y[0]

        return (num_pixels_1 / num_pixels) * (delta_x * delta_y)

    def inverse_amplitude(self, new_field=False):
        """Inverts the amplitude of the mask, phase is equal as initial
        
        Parameters:
            new_field (bool): If True it returns a Scalar_mask_XY object, else, it modifies the existing object
            
            
        Returns:
            Scalar_mask_XY:  If new_field is True, it returns a Scalar_mask_XY object.
        """

        amplitude = np.abs(self.u)
        phase = angle(self.u)

        new_amplitude = ne.eval("(1 - amplitude) * exp(1j * phase)")

        if new_field is False:
            self.u = new_amplitude
        else:
            new = Scalar_mask_XY(self.x, self.y, self.wavelength)
            new.u = new_amplitude
            return new

    def inverse_phase(self, new_field=False):
        """Inverts the phase of the mask, amplitude is equal as initial
        
        Parameters:
            new_field (bool): If True it returns a Scalar_mask_XY object, else, it modifies the existing object
            
            
        Returns:
            Scalar_mask_XY:  If new_field is True, it returns a Scalar_mask_XY object.
        """

        amplitude = np.abs(self.u)
        phase = angle(self.u)

        new_amplitude = amplitude * exp(-1j * phase)

        if new_field is False:
            self.u = new_amplitude
        else:
            new = Scalar_mask_XY(self.x, self.y, self.wavelength)
            new.u = new_amplitude
            return new

    def filter(self, mask, new_field=True, binarize=False, normalize=False):
        """Widens a field using a mask

        Parameters:
            mask (diffractio.Scalar_mask_XY): filter
            new_field (bool): If True, develope new Field
            binarize (bool, float): If False nothing, else binarize in level
            normalize (bool): If True divides the mask by sum.
        """

        f1 = ne.evaluate("abs(mask.u)")

        if normalize is True:
            f1 = f1 / f1.sum()

        covolved_image = convolve(f1, np.abs(self.u))
        if binarize is not False:
            covolved_image[covolved_image > binarize] = 1
            covolved_image[covolved_image <= binarize] = 0

        if new_field is True:
            new = Scalar_field_XY(self.x, self.y, self.wavelength)
            new.u = covolved_image
            return new
        else:
            self.u = covolved_image

    def widen(self, radius, new_field=True, binarize=True):
        """Widens a mask using a convolution of a certain radius

        Parameters:
            radius (float): radius of convolution
            new_field (bool): returns a new XY field
            binarize (bool): binarizes result.
        """

        filter = Scalar_mask_XY(self.x, self.y, self.wavelength)
        filter.circle(r0=(0 * um, 0 * um), radius=radius, angle=0 * degrees)

        image = np.abs(self.u)
        filtrado = np.abs(filter.u) / np.abs(filter.u.sum())

        covolved_image = fft_convolution2d(image, filtrado)
        minimum = 0.01 * covolved_image.max()

        if binarize is True:
            covolved_image[covolved_image > minimum] = 1
            covolved_image[covolved_image <= minimum] = 0
        else:
            covolved_image = covolved_image / covolved_image.max()

        if new_field is True:
            filter.u = covolved_image
            return filter
        else:
            self.u = covolved_image
    
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
        u_mask = np.zeros_like(X)
        ipasa = ne.evaluate("r < radius")
        u_mask[ipasa] = 1

        if off_axis_angle == 0 * degrees:
            t_off_axis = 1
        else:
            t_off_axis = ne.evaluate("exp(-1j * k * X * sin(off_axis_angle))")

        if reflective is True:
            self.u = ne.evaluate("u_mask * exp(-2j * k * r * tan(angle)) * t_off_axis")

        else:
            self.u = u_mask * \
                ne.evaluate("exp(-1j * k * (refractive_index - 1) * r * tan(angle)) * t_off_axis")
            
    def circle(self, r0, radius, angle=0 * degrees):
        """Creates a circle or an ellipse.

        Parameters:
            r0 (float, float): center of circle/ellipse
            radius (float, float) or (float): radius of circle/ellipse
            angle (float): angle of rotation in radians

        Example:

            circle(r0=(0 * um, 0 * um), radius=(250 * um, 125 * um), angle=0 * degrees)
        """
        x0, y0 = r0

        if isinstance(radius, (float, int, complex)):
            radiusx, radiusy = (radius, radius)
        else:
            radiusx, radiusy = radius

        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        u = np.zeros(np.shape(self.X))
        ipasa = ne.evaluate("Xrot**2 / radiusx**2 + Yrot**2 / radiusy**2 < 1")
        u[ipasa] = 1
        self.u = u

@profile
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

    Ek = fftshift_parallel_2d(fft_parallel_2d(u))
    H = ne.evaluate("exp(1j * dz * sqrt(n**2 * k0**2 - k_perp2) - absorption)")
    HEk = ne.evaluate("H * Ek")
    result = ifft_parallel_2d(fftshift_parallel_2d(HEk))
    
    return result

@profile
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
        Imz = ne.evaluate("n == n_m")
        u_final = ne.evaluate("u_final + Imz * u_temp")

    return u_final

@profile
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

    filter_x = ne.evaluate("exp(-(abs(X - x_center) / width_edge) ** 20)")
    filter_x = ne.evaluate("exp(-(abs(X - x_center) / width_edge) ** 20)")
    filter_y = ne.evaluate("exp(-(abs(Y - y_center) / width_edge) ** 20)")
    filter_function = ne.evaluate("filter_x * filter_y")

    z = z0
    for i in range(num_points):
        if obstacle:
            if CONSTANTS["nozzle"]["dist_z"]*mm <= z <= CONSTANTS["nozzle"]["dist_z"]*mm + CONSTANTS["nozzle"]["z_size"]*mm:
                n_field = np.ones_like(X, dtype=np.complex64)
                square = ne.evaluate("(abs(X) <= CONSTANTS['nozzle']['x_size']/2*mm) * (Y >= - CONSTANTS['nozzle']['dist_y']*mm)")
                n_field[square] = np.complex64(1.2 + 7j) # The refractive index of aluminum

                triangle1 = ne.evaluate("(Y <= X + (-CONSTANTS['nozzle']['dist_y'] - CONSTANTS['nozzle']['slope_upper']/2)*mm)")
                triangle2 = ne.evaluate("(Y <= -X + (-CONSTANTS['nozzle']['dist_y'] - CONSTANTS['nozzle']['slope_upper']/2)*mm)")
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
                dz, z, CONSTANTS["nozzle"]["dist_z"]*mm, CONSTANTS["nozzle"]["dist_z"]*mm + CONSTANTS["nozzle"]["z_size"]*mm) * filter_function
        print(f"{i+1}/{num_points}", sep='\r', end='\r')
        
        z += dz
        
    return u_field


if __name__ == "__main__":
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
    SCALE = 1/5
    REGION_SIZE = CONSTANTS["region"]["size"] * SCALE

    Nx = int(1000*REGION_SIZE)
    Ny = int(1000*REGION_SIZE)

    print(f"In the x axis using {Nx/REGION_SIZE/1000} px/um with total of {Nx} px")
    print(f"In the y axis using {Ny/REGION_SIZE/1000} px/um with total of {Ny} px")

    x = np.linspace(-REGION_SIZE/2*mm, REGION_SIZE/2*mm, Nx,)
    y = np.linspace(-REGION_SIZE/2*mm, REGION_SIZE/2*mm, Ny,)
    print(f"Available memory: {psutil.virtual_memory().available/1000000000} GB")
    print(f"The number of threads is {num_threads}\n")


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
                refractive_index=1.51,
                reflective=True)

        return np.complex64((u0 * t0 * t1).u)

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
