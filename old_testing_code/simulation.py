from diffractio import plt, np, mm, degrees, um
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ

"""
Problematic - at least for now. The shrinking of the grid shrinks
not only the computational domain, but also the optical elements used in the model
(axicone, nozzle). Might try something with it in the future.

def squared_space(start: float, stop: float, num: int):
    relative = np.linspace(0, 1, int(num/2)) ** 0.5
    return np.array(list(relative * (stop-start)/2 + start) + list(relative * (start-stop)/2 + stop)[::-1])

"""

constants = {
    "nozzle": {"dist_z": 30, # mm
        "dist_x": -15, # mm
        "y_size": 10, # mm
        "z_size": 100,}, # mm
    "wave": {"size": 55}, # mm
    "axicon": {"angle": 3}, # dg
}

x0 = np.linspace(-(constants["wave"]["size"]+5)/2*mm, (constants["wave"]["size"]+5)/2*mm, 1000) 
y0 = np.linspace(-(constants["wave"]["size"]+5)/2*mm, (constants["wave"]["size"]+5)/2*mm, 1000)
z0 = np.linspace(0*mm, (constants["nozzle"]["z_size"] + constants["nozzle"]["dist_z"]+5)*mm, 15)
wavelength = 0.8*um

u0 = Scalar_source_XY(x0, y0, wavelength)
u0.plane_wave(A=1, theta=0 * degrees)

t1 = Scalar_mask_XY(x0, y0, wavelength)
t1.square(r0=(0*mm, 0*mm),
          size=constants["wave"]["size"]*mm,
          angle=0*degrees)

t2 = Scalar_mask_XY(x0, y0, wavelength)
t2.axicon(r0=(0*mm, 0*mm),
          radius=constants["wave"]["size"]/2*mm,
          angle=constants["axicon"]["angle"]*degrees,
          refraction_index=1.5,
          reflective=True)

u1 = u0 * t1* t2

nozzle = Scalar_mask_XYZ(x0, y0, z0, wavelength, n_background=1.0, info='')
nozzle.square(r0=((-(constants["wave"]["size"]/2 + constants["nozzle"]["dist_x"])/2 + constants["nozzle"]["dist_x"])*mm, 
                  0*mm, 
                  (constants["nozzle"]["z_size"]/2 + constants["nozzle"]["dist_z"])*mm), # the X and Y coordinates are inverted.
              length=((constants["wave"]["size"]/2 + constants["nozzle"]["dist_x"])*mm, 
                      constants["nozzle"]["y_size"]*mm, 
                      constants["nozzle"]["z_size"]*mm),
              refraction_index=1.3+7j,
              angles=(0*degrees, 0*degrees, 0*degrees),
              rotation_point=0)

nozzle.incident_field(u1)

nozzle.WPM(verbose=True, has_edges=True)
# nozzle.normalize()

nozzle.draw_XZ(y0=0, kind='intensity',
          logarithm=1e1,
          normalize=None,
          colorbar_kind='vertical',
          draw_borders = True,)

nozzle.draw_XY(z0=0*mm, kind='intensity',
          logarithm=1e1,
          normalize=None,)

nozzle.draw_XY(z0=20*mm, kind='intensity',
          logarithm=1e1,
          normalize=None,)

nozzle.draw_XY(z0=60*mm, kind='intensity',
          logarithm=1e1,
          normalize=None,)   

nozzle.draw_XY(z0=120*mm, kind='intensity',
          logarithm=1e1,
          normalize=None,)       

plt.show()
