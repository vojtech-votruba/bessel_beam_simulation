from diffractio import sp, nm, plt, np, mm, degrees, um
from diffractio.scalar_sources_X import Scalar_source_X
from diffractio.scalar_fields_XZ import Scalar_field_XZ
from diffractio.scalar_masks_XZ import Scalar_mask_XZ
from diffractio.scalar_masks_X import Scalar_mask_X
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ

import matplotlib.cm as cm

x0 = np.linspace(-500 * um, 500 * um, 1024 * 4)
z0 = np.linspace(-0.1 * mm, 1.25 * mm, 1024 * 2)
wavelength = 5 * um

u0 = Scalar_source_X(x0, wavelength)
u0.plane_wave(A=1, theta=0 * degrees)
                

lens = Scalar_mask_XZ(x0, z0, wavelength, n_background=1, info='')
ipasa, conds = lens.aspheric_lens(r0=(0 * mm, 0 * mm),
                                  angle=(0 * degrees, (0 * mm, 0 * mm)),
                                  refraction_index=1.5,
                                  cx=(1 / (1 * mm), -1 / (.25 * mm)),
                                  Qx=(0, 0),
                                  a2=(0, 1e-13),
                                  a3=(0, 0),
                                  a4=(0, 0),
                                  depth=.4 * mm,
                                  size=0.8 * mm)

lens.slit(r0=(0, 100 * um),
          aperture=800 * um,
          depth=75 * um,
          refraction_index=1 + 2j)

lens.draw_refraction_index(draw_borders=True,
                           min_incr=0.01,
                           colormap_kind=cm.Blues,
                           colorbar_kind='vertical')

lens.smooth_refraction_index(type_filter=2, pixels_filtering=25)

lens.incident_field(u0)

lens.clear_field()
lens.BPM(verbose=False)
lens.normalize()

lens.draw(kind='intensity',
          logarithm=1e1,
          normalize=None,
          draw_borders=True,
          colorbar_kind='vertical')

Intensity_BPM = lens.intensity()


x_f_bpm, z_f_bpm = lens.search_focus()

ylim_max = 20 * um
zlim_max = 100 * um

lens.draw(kind='intensity', colorbar_kind='horizontal')

plt.ylim(-ylim_max, ylim_max)
plt.xlim(z_f_bpm - zlim_max, z_f_bpm + zlim_max)


lens.draw(kind='phase', colorbar_kind='horizontal', percentage_intensity=0.05)

plt.ylim(-ylim_max, ylim_max)
plt.xlim(z_f_bpm - zlim_max, z_f_bpm + zlim_max)
plt.show()
