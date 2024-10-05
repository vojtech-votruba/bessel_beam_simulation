from diffractio import sp, nm, plt, np, mm, degrees, um
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_fields_XY import Scalar_field_XY
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ
#from cv2 import cv2
import matplotlib.cm as cm
from numpy import pi,sqrt,exp, linspace, array, concatenate, meshgrid, zeros, flipud
from scipy.fftpack import fft2, ifft2, fftshift


def kernelRS(X, Y, wavelength, z, n, kind='z'):
    """Kernel for RS propagation. 

    Parameters:
        X(numpy.array): positions x
        Y(numpy.array): positions y
        wavelength(float): wavelength of incident fields
        z(float): distance for propagation
        n(float): refraction index of background
        kind(str): 'z', 'x', '0': for simplifying vector propagation

    Returns:
        complex np.array: kernel
    """
    k = 2 * pi * n / wavelength
    R = sqrt(X**2 + Y**2 + z**2)
    if kind == 'z':
        return 1 / (2 * pi) * exp(1.j * k * R) * z / R**2 * (1 / R - 1.j * k)
    elif kind == 'x':
        return 1 / (2 * pi) * exp(1.j * k * R) * X / R**2 * (1 / R - 1.j * k)
    elif kind == 'y':
        return 1 / (2 * pi) * exp(1.j * k * R) * Y / R**2 * (1 / R - 1.j * k)
    elif kind == '0':
        return 1 / (2 * pi) * exp(1.j * k * R) / R * (1 / R - 1.j * k)

def _RS_(self,
             z,
             n,
             new_field=True,
             out_matrix=False,
             kind='z',
             xout=None,
             yout=None,
             verbose=False):
        """Fast-Fourier-Transform  method for numerical integration of diffraction Rayleigh-Sommerfeld formula. `Thin Element Approximation` is considered for determining the field just after the mask: :math:`\mathbf{E}_{0}(\zeta,\eta)=t(\zeta,\eta)\mathbf{E}_{inc}(\zeta,\eta)` Is we have a field of size N*M, the result of propagation is also a field N*M. Nevertheless, there is a parameter `amplification` which allows us to determine the field in greater observation planes (jN)x(jM).

        Parameters:
            z (float): distance to observation plane.
                if z<0 inverse propagation is executed
            n (float): refraction index
            new_field (bool): if False the computation goes to self.u
                              if True a new instance is produced

            xout (float), init point for amplification at x
            yout (float), init point for amplification at y
            verbose (bool): if True it writes to shell

        Returns:
            if New_field is True: Scalar_field_X
            else None

        Note:
            One adventage of this approach is that it returns a quality parameter: if self.quality>1, propagation is right.


        References:
            F. Shen and A. Wang, “Fast-Fourier-transform based numerical integration method for the Rayleigh-Sommerfeld diffraction formula,” Appl. Opt., vol. 45, no. 6, pp. 1102–1110, 2006.

        """

        if xout is None:
            xout = self.x[0]
        if yout is None:
            yout = self.y[0]

        xout = self.x + xout - self.x[0]
        yout = self.y + yout - self.y[0]

        nx = len(xout)
        ny = len(yout)
        dx = xout[1] - xout[0]
        dy = yout[1] - yout[0]

        dr_real = sqrt(dx**2 + dy**2)
        rmax = sqrt((xout**2).max() + (yout**2).max())
        dr_ideal = sqrt((self.wavelength / n)**2 + rmax**2 + 2 *
                        (self.wavelength / n) * sqrt(rmax**2 + z**2)) - rmax
        self.quality = dr_ideal / dr_real

        """if verbose is True:
            if (self.quality.min() >= 0.99):
                print('Good result: factor {:2.2f}'.format(self.quality),
                      end='\r')
            else:
                print('- Needs denser sampling: factor {:2.2f}\n'.format(
                    self.quality))"""
        precise = 0
        if precise:
            a = [4, 2]
            num_repx = int(round((nx) / 2) - 1)
            num_repy = int(round((ny) / 2) - 1)
            bx = array(a * num_repx)
            by = array(a * num_repy)
            cx = concatenate(((1, ), bx, (2, 1))) / 3.
            cy = concatenate(((1, ), by, (2, 1))) / 3.

            if float(nx) / 2 == round(nx / 2):  # es par
                i_centralx = num_repx + 1
                cx = concatenate((cx[:i_centralx], cx[i_centralx + 1:]))
            if float(ny) / 2 == round(ny / 2):  # es par
                i_centraly = num_repy + 1
                cy = concatenate((cy[:i_centraly], cy[i_centraly + 1:]))

            W = (cx[:, np.newaxis] * cy[np.newaxis, :]).T

        else:
            W = 1

        U = zeros((2 * ny - 1, 2 * nx - 1), dtype=complex)
        U[0:ny, 0:nx] = array(W * self.u)

        xext = self.x[0] - xout[::-1]
        xext = xext[0:-1]
        xext = concatenate((xext, self.x - xout[0]))

        yext = self.y[0] - yout[::-1]
        yext = yext[0:-1]
        yext = concatenate((yext, self.y - yout[0]))

        Xext, Yext = meshgrid(xext, yext)

        # permite calcula la propagacion y la propagacion inverse, cuando z<0.
        if z > 0:
            H = kernelRS(Xext, Yext, self.wavelength, z, n, kind=kind)
        else:
            H = kernelRSinverse(Xext, Yext, self.wavelength, z, n, kind=kind)

        # calculo de la transformada de Fourier
        S = ifft2(fft2(U) * fft2(H)) * dx * dy
        # transpose cambiado porque daba problemas para matrices no cuadradas
        Usalida = S[ny - 1:, nx - 1:]  # hasta el final
        # los calculos se pueden dejar en la instancia o crear un new field

        # Usalida = Usalida / z  210131

        if out_matrix is True:
            return Usalida

        if new_field is True:
            field_output = Scalar_field_XY(self.x, self.y, self.wavelength)
            field_output.u = Usalida
            field_output.quality = self.quality
            return field_output
        else:
            self.u = Usalida

def RS(self,
        z,
        n,
        amplification=(1, 1),
        new_field=True,
        matrix=False,
        xout=None,
        yout=None,
        kind='z',
        verbose=False):
    """Fast-Fourier-Transform  method for numerical integration of diffraction Rayleigh-Sommerfeld formula. Is we have a field of size N*M, the result of propagation is also a field N*M. Nevertheless, there is a parameter `amplification` which allows us to determine the field in greater observation planes (jN)x(jM).

    Parameters:
        amplification (int, int): number of frames in x and y direction
        z (float): distance to observation plane. if z<0 inverse propagation is executed
        n (float): refraction index
        new_field (bool): if False the computation goes to self.u, if True a new instance is produced.
        matrix(Bool): if True returns a matrix, else a Scalar_field_XY
        xout (float): If not None, the sampling area is moved. This is the left position
        yout (float): If not None, the sampling area y moved. This is the lower position.
        kind (str):
        verbose (bool): if True it writes to shell

    Returns:
        if New_field is True: Scalar_field_X, else None.

    Note:
        One advantage of this approach is that it returns a quality parameter: if self.quality>1, propagation is right.

    References:
        F. Shen and A. Wang, “Fast-Fourier-transform based numerical integration method for the Rayleigh-Sommerfeld diffraction formula,” Appl. Opt., vol. 45, no. 6, pp. 1102–1110, 2006.
    """

    amplification_x, amplification_y = amplification
    ancho_x = self.x[-1] - self.x[0]
    ancho_y = self.y[-1] - self.y[0]
    num_pixels_x = len(self.x)
    num_pixels_y = len(self.y)

    if amplification_x * amplification_y > 1:

        posiciones_x = -amplification_x * ancho_x / 2 + array(
            list(range(amplification_x))) * ancho_x
        posiciones_y = -amplification_y * ancho_y / 2 + array(
            list(range(amplification_y))) * ancho_y

        X0 = linspace(-amplification_x * ancho_x / 2,
                        amplification_x * ancho_x / 2,
                        num_pixels_x * amplification_x)
        Y0 = linspace(-amplification_y * ancho_y / 2,
                        amplification_y * ancho_y / 2,
                        num_pixels_y * amplification_y)

        U_final = Scalar_field_XY(x=X0, y=Y0, wavelength=self.wavelength)

        for i, xi in zip(list(range(len(posiciones_x))),
                            flipud(posiciones_x)):
            for j, yi in zip(list(range(len(posiciones_y))),
                                flipud(posiciones_y)):
                # num_ventana = j * amplification_x + i + 1
                u3 = _RS_(z=z,
                                n=n,
                                new_field=False,
                                kind=kind,
                                xout=xi,
                                yout=yi,
                                out_matrix=True,
                                verbose=verbose)
                xshape = slice(i * num_pixels_x, (i + 1) * num_pixels_x)
                yshape = slice(j * num_pixels_y, (j + 1) * num_pixels_y)
                U_final.u[yshape, xshape] = u3

        if matrix is True:
            return U_final.u
        else:
            if new_field is True:
                return U_final
            else:
                self.u = U_final.u
                self.x = X0
                self.y = Y0
    else:

        if xout is None:
            u_s = _RS_(z,
                            n,
                            new_field=new_field,
                            out_matrix=True,
                            kind=kind,
                            xout=xout,
                            yout=yout,
                            verbose=verbose)
        else:
            u_s = _RS_(z,
                            n,
                            new_field=new_field,
                            out_matrix=True,
                            kind=kind,
                            xout=-xout + self.x[0] - ancho_x / 2,
                            yout=-yout + self.y[0] - ancho_y / 2,
                            verbose=verbose)

        if matrix is True:
            return u_s

        if new_field is True:
            U_final = Scalar_field_XY(x=self.x,
                                        y=self.y,
                                        wavelength=self.wavelength)
            U_final.u = u_s
            if xout is not None:
                U_final.x = self.x + xout - self.x[0]
                U_final.y = self.y + yout - self.y[0]
                U_final.X, U_final.Y = meshgrid(self.x, self.y)

            return U_final
        else:
            self.u = u_s
            self.x = self.x + xout - self.x[0]
            self.y = self.y + yout - self.y[0]
            self.X, self.Y = meshgrid(self.x, self.y)


x0 = np.linspace(-0.5 * mm, 0.5 * mm, 1000)
y0 = np.linspace(-0.5 * mm, 0.5 * mm, 1000)
wavelength = 0.8 * um

X,Y = np.meshgrid(x0,y0)

u0 = Scalar_source_XY(x0, y0, wavelength)
u0.plane_wave(A=1, theta=0 * degrees)

t1 = Scalar_mask_XY(x0, y0, wavelength)
t1.axicon(r0=(0 * mm, 0 * mm),
          radius=0.49 * mm,
          angle=2*degrees,
          refraction_index=1.5,
          reflective=False)

t2 = Scalar_mask_XY(x0, y0, wavelength)
t2.circle(r0=(0, 0 * mm),
          radius=0.485 * mm,
          angle=0 * degrees)

u1 = u0*t2*t1 # Bessel beam created with axicon

# nozzle.normalize()"""

u2 = u1.RS(5*mm, verbose=True)
u2.draw(kind='intensity',
          logarithm=1e1,
          normalize=None,)

z0 = np.linspace(5 * mm, 7 * mm, 4)
nozzle = Scalar_mask_XYZ(x0, y0, z0, wavelength, n_background=1.0, info='')
nozzle.square(r0=(-0.525* mm, 0 * mm, 9 * mm),
              length=(1 * mm, 0.4 * mm, 10 * mm),
              refraction_index=1.3+7j,
              angles=(0 * degrees, 0 * degrees, 0 * degrees),
              rotation_point=0)

nozzle.incident_field(u1)

nozzle.clear_field()
nozzle.WPM(verbose=True)

n0 = np.ones_like(u2.u)
condition1 = (-0.2 < X) * (X < 0.2)
condition2 = Y < -0.525
n0[condition1 * condition2] = refraction_index=1.3+7j
print(n0)

n0 = np.ones_like(u2.u)

u3 = nozzle.to_Scalar_field_XY(1, 7*mm)
u4 = u3.RS(10*mm, n=n0, verbose=True)
u4.draw(kind='intensity',
          logarithm=1e1,
          normalize=None,)


plt.show()
