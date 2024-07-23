#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:08:58 2024

@author: jirisisma
"""

from diffractio import sp, nm, plt, np, mm, degrees, um
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_fields_XY import Scalar_field_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_masks_X import Scalar_mask_X
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ
#from cv2 import cv2
import matplotlib.cm as cm

x0 = np.linspace(-12.5 * mm, 12.5 * mm, 120)
y0 = np.linspace(-12.5 * mm, 12.5 * mm, 120)
z0 = np.linspace(0 * mm, 150 * mm, 1024)
wavelength = 0.8 * um

u0 = Scalar_source_XY(x0, y0, wavelength)
u0.plane_wave(A=1, theta=0 * degrees)

t1 = Scalar_mask_XY(x0, y0, wavelength)
t1.axicon(r0=(0 * mm, 0 * mm),
          radius=10 * mm,
          angle=160*degrees,
          refraction_index=1.5,
          reflective=False)
"""
t1.circle(r0=(0, 0 * mm),
          radius=10 * mm,
          angle=0 * degrees)
"""
u1 = u0*t1

nozzle = Scalar_mask_XYZ(x0, y0, z0, wavelength, n_background=1.0, info='')
nozzle.square(r0=(-7 * mm, 0 * mm, 50 * mm),
              length=(12 * mm, 2 * mm, 100 * mm),
              refraction_index=1+2j,
              angles=(0 * degrees,0 * degrees,0 * degrees),
              rotation_point=0)
"""
nozzle.draw_refraction_index(draw_borders=True,
                             min_incr=0.01,
                             colormap_kind=cm.Blues,
                             colorbar_kind='vertical')
plt.show()
"""
nozzle.incident_field(u1)

"""
nozzle.smooth_refraction_index(type_filter=2, pixels_filtering=25)

nozzle.incident_field(u0)
"""
nozzle.clear_field()
nozzle.WPM(verbose=False)
# nozzle.normalize()

nozzle.draw_XZ(y0=0, kind='intensity',
          logarithm=1e1,
          normalize=None,
          colorbar_kind='vertical')

nozzle.draw_XY(z0=120*mm, kind='intensity',
          logarithm=1e1,
          normalize=None)
plt.show()



