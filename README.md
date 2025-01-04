README.md
# Bessel beam simulation
Simulation of a beam propagating through an axicon and diffracting on a plasma nozzle for electron acceleration experiments.
The code is using WPM - Wave Propagation Method.
Main parts of the project are: simulate.py and analytical_solution.py, the former is used for the actual numerical simulation while the latter
is more for comparison purposes and calculates the intensity field analytically for the cases without any obstacles.

# How to run it
Firstly, you should install the used libraries via
```
pip install -r /path/to/bessel-beam-simulation/requirements.txt
```
I tested the requirements on Python 3.12, and I am not entirely sure if they work on other versions.
I also highly encourage using a virutal environment for this. diffractio as a library isn't particullary stable, and you don't want to mess up your python installation.

After you installed the libraries you should be ready to go. For simulate.py there is an option to use the argument `--obstacle` or `--no-obstacle`
to run the simulation with the plasma nozzle.

All constants and parameters used in the simulation are stored in config.json file.

# Possible improvements
- Implement an propagation algorithm with an adaptive grid, e.g. CZT?
- Find a larger computer with more RAM

# Things to be aware of in diffractio
- Refractive index vs. Refraction index
- r0 = (x,y,z) vs. r0 = (y,x,z)

# References
- L.M. Sanchez Brea, “Diffractio, python module for diffraction and interference optics”, https://pypi.org/project/diffractio/ (2019)
- K.-H. Brenner, W. Singer, “Light propagation through micro lenses: a new simulation method”, Appl. Opt., 32(6) 4984-4988 (1993).
- Shalloo, R. 2018. “Hydrodynamic Optical-Field-Ionized Plasma Waveguides for Laser Plasma Accelerators.” PhD thesis, University of Oxford.
