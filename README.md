# Bessel beam simulation
Simulation of a beam propagating through an axicon and diffracting on a plasma nozzle for electron acceleration experiments.
The code is using WPM - Wave Propagation Method.
Main parts of the project are: simulate.py and analytical_solution.py, the former is used for the actual simulation while the latter
is for comparison purposes, and calculates the intensity field analytically for the cases without any obstacles.

# Things to be aware of in diffractio
- Refractive index vs. Refraction index
- r0 = (x,y,z) vs. r0 = (y,x,z)

# References
- L.M. Sanchez Brea, “Diffractio, python module for diffraction and interference optics”, https://pypi.org/project/diffractio/ (2019)
- K.-H. Brenner, W. Singer, “Light propagation through micro lenses: a new simulation method”, Appl. Opt., 32(6) 4984-4988 (1993).
- Shalloo, R. 2018. “Hydrodynamic Optical-Field-Ionized Plasma Waveguides for Laser Plasma Accelerators.” PhD thesis, University of Oxford.