from diffractio import sp, nm, plt, np, mm, degrees, um
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY
from XYZ_masks.scalar_masks_XYZ import Scalar_mask_XYZ
#from cv2 import cv2
import matplotlib.cm as cm

x0 = np.linspace(-0.5 * mm, 0.5 * mm, 1000)
y0 = np.linspace(-0.5 * mm, 0.5 * mm, 1000)
z0 = np.linspace(0 * mm, 20 * mm, 10)
wavelength = 0.8 * um

u0 = Scalar_source_XY(x0, y0, wavelength)
u0.plane_wave(A=1, theta=0 * degrees)

t1 = Scalar_mask_XY(x0, y0, wavelength)
t1.axicon(r0=(0 * mm, 0 * mm),
          radius=0.49 * mm,
          angle=5*degrees,
          refraction_index=1.5,
          reflective=False)

t2 = Scalar_mask_XY(x0, y0, wavelength)
t2.circle(r0=(0, 0 * mm),
          radius=0.485 * mm,
          angle=0 * degrees)

u1 = u0*t2*t1

nozzle = Scalar_mask_XYZ(x0, y0, z0, wavelength, n_background=1.0, info='')
"""nozzle.square(r0=(-0.7* mm, 0 * mm, 6.5 * mm),
              length=(1.2 * mm, 0.15 * mm, 13 * mm),
              refraction_index=1.3+7j,
              angles=(0 * degrees, 0 * degrees, 0 * degrees),
              rotation_point=0)

nozzle.draw_refraction_index(draw_borders=True,
                             min_incr=0.01,
                             colormap_kind=cm.Blues,
                             colorbar_kind='vertical')
plt.show()
"""
nozzle.incident_field(u1)

"""4000
nozzle.smooth_refraction_index(type_filter=2, pixels_filtering=25)

nozzle.incident_field(u0)
"""
nozzle.clear_field()
nozzle.WPM(verbose=True)
# nozzle.normalize()

nozzle.draw_XZ(y0=0, kind='intensity',
          logarithm=1e1,
          normalize=None,
          colorbar_kind='vertical')

nozzle.draw_XY(z0=4.9*mm, kind='intensity',
          logarithm=1e1,
          normalize=None)
plt.show()



