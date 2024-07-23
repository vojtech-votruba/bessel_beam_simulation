from diffractio import np, plt, um, mm, degrees
from diffractio.scalar_masks_XY import Scalar_mask_XY
from XYZ_masks.scalar_masks_XYZ import Scalar_mask_XYZ
from diffractio.scalar_sources_XY import Scalar_source_XY

constants = {
    "nozzle": {"dist_z": 30, # mm
        "slope_height": 3.4, # mm
        "slope_base": 10, # mm
        "dist_x": -5, # mm
        "y_size": 10, # mm
        "z_size": 100,}, # mm
    "region": {"size": 55}, # mm
    "axicon": {"angle": 3}, # dg, approach angle must be 6 dg.
}

x = np.linspace(-(constants["region"]["size"]+5)/2*um, (constants["region"]["size"]+5)/2*um, 1500) 
y = np.linspace(-(constants["region"]["size"]+5)/2*um, (constants["region"]["size"]+5)/2*um, 1500)
z = np.linspace(0*um, (constants["nozzle"]["z_size"]+constants["nozzle"]["dist_z"]+5)*um, 15)
wavelength = 0.8 * um

u0 = Scalar_source_XY(x, y, wavelength)
u0.plane_wave(A=1, theta=0*degrees)

t0 = Scalar_mask_XY(x, y, wavelength)
t0.axicon(r0=(0*um, 0*um),
          radius=constants["region"]["size"]/2*um,
          angle=constants["axicon"]["angle"]*degrees,
          refraction_index=1.51,
          reflective=True)

uxyz = Scalar_mask_XYZ(x, y, z, wavelength)
# Upper, declined part of the nozzle
"""uxyz.triangle_prism(r0=((5)*um,
                        0,
                        (constants["nozzle"]["z_size"]/2 + constants["nozzle"]["dist_z"])*um),
                        length = constants["nozzle"]["z_size"],
                        refractive_index=1.3+7j,
                        triangle_height=constants["nozzle"]["slope_height"]*um,
                        triangle_base=constants["nozzle"]["slope_base"]*um)"""

# Lower part of the nozzle
"""uxyz.square(r0=((-(constants["region"]["size"]/2 + constants["nozzle"]["dist_x"])/2 + constants["nozzle"]["dist_x"])*um - 5*um - constants["nozzle"]["slope_height"]*um, 
                  0*um, 
                  (constants["nozzle"]["z_size"]/2 + constants["nozzle"]["dist_z"])*um), # the X and Y coordinates are inverted.
              length=((constants["region"]["size"]/2 + constants["nozzle"]["dist_x"])*um + 5*um, 
                      constants["nozzle"]["y_size"]*um, 
                      constants["nozzle"]["z_size"]*um),
              refractive_index=1.3+7j,
              angles=(0*degrees, 0*degrees, 0*degrees),
              rotation_point=0)"""

uxyz.incident_field(u0*t0)

uxyz.clear_field()
uxyz.WPM(verbose=True, has_edges=True)

uxyz.draw_XZ(y0=0, kind='intensity',
          logarithm=1e1,
          normalize=None,
          colorbar_kind='vertical',
          draw_borders = True,)

uxyz.draw_XY(z0=20*um, kind='intensity',
          logarithm=1e1,
          normalize=None,)

uxyz.draw_XY(z0=60*um, kind='intensity',
          logarithm=1e1,
          normalize=None,)   

uxyz.draw_XY(z0=120*um, kind='intensity',
          logarithm=1e1,
          normalize=None,)
  
"""For comparison with analytical solution using bessel profile without axicon:"""


plt.show()
