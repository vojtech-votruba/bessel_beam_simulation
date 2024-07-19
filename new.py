from diffractio import np, plt, um, mm, degrees
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ

constants = {
    "nozzle": {"dist_z": 30, # mm
        "dist_x": -5, # mm
        "y_size": 10, # mm
        "z_size": 100,}, # mm
    "wave": {"size": 55}, # mm
    "axicon": {"angle": 3}, # dg
}

x = np.linspace(-(constants["wave"]["size"]+5)/2*mm, (constants["wave"]["size"]+5)/2*mm, 1000) 
y = np.linspace(-(constants["wave"]["size"]+5)/2*mm, (constants["wave"]["size"]+5)/2*mm, 1000)
z = np.linspace(0*mm, (constants["nozzle"]["z_size"] + constants["nozzle"]["dist_z"]+5)*mm, 15)
wavelength = 0.8 * um

t0 = Scalar_mask_XY(x, y, wavelength)
t0.axicon(r0=(0*mm, 0*mm),
          radius=constants["wave"]["size"]/2*mm,
          angle=constants["axicon"]["angle"]*degrees,
          refraction_index=1.5,
          reflective=True)


uxyz = Scalar_mask_XYZ(x, y, z, wavelength)
uxyz.square(r0=((-(constants["wave"]["size"]/2 + constants["nozzle"]["dist_x"])/2 + constants["nozzle"]["dist_x"])*mm - 5*mm, 
                  0*mm, 
                  (constants["nozzle"]["z_size"]/2 + constants["nozzle"]["dist_z"])*mm), # the X and Y coordinates are inverted.
              length=((constants["wave"]["size"]/2 + constants["nozzle"]["dist_x"])*mm + 5*mm, 
                      constants["nozzle"]["y_size"]*mm, 
                      constants["nozzle"]["z_size"]*mm),
              refraction_index=1.3+7j,
              angles=(0*degrees, 0*degrees, 0*degrees),
              rotation_point=0)

uxyz.incident_field(u0=t0)

uxyz.clear_field()
uxyz.WPM(verbose=True, has_edges=True)

uxyz.draw_XZ(y0=0, kind='intensity',
          logarithm=1e1,
          normalize=None,
          colorbar_kind='vertical',
          draw_borders = True,)

uxyz.draw_XY(z0=0*mm, kind='intensity',
          logarithm=1e1,
          normalize=None,)

uxyz.draw_XY(z0=20*mm, kind='intensity',
          logarithm=1e1,
          normalize=None,)

uxyz.draw_XY(z0=60*mm, kind='intensity',
          logarithm=1e1,
          normalize=None,)   

uxyz.draw_XY(z0=120*mm, kind='intensity',
          logarithm=1e1,
          normalize=None,)
  
plt.show()
