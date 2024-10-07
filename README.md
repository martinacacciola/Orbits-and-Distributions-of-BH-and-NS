# Orbits and Distributions of BH and NS in the Milky Way
In this project, we will leverage N-body simulations to integrate massive stars in disc-like orbits until explosion. At this point, the compact object is subject to both its orbital velocity and the velocity kick. The remnant will then be studied in its evolution for 3 − 5 Gyr: the aim is to investigate the spatial properties of the population of NSs and BHs, depending on the remnant type, the kick model and the metallicity.
## Astrophysical Introduction
Numerical N-body simulations involve the computation of forces acting on N particles over a time period t. These simulations are particularly important in astrophysics as they model the evolution of celestial structures, which are governed by the dynamical interactions between bodies under gravitational forces. This approach helps to understand large-scale structure problems (e.g. the formation and evolution of galaxies) and to investigate the dynamics of star clusters. Potential estimates are crucial: they provide a measure of the gravitational potential at each point in space, used then for calculating the forces experienced by each particle. An interesting application is the study of the spatial distribution of the remnants of high-mass stars. When massive stars (> 10 $M_{sun}$) die via core-collapse SN explosion, they leave behind a compact object (either a NS or a BH), expected to receive a spatial velocity at its birth because of asymmetric mass ejection. For NSs, we can recon- struct the distribution of kick magnitudes from observa- tions of proper motions of Galactic pulsars, thanks to the radio jet powered by high-velocity rotation and magnetic field. Instead, for BHs the data are scanty and complex to interpret, as their detections come from indirect methods, e.g. gravitational micro-lensing or the observation of BH-star systems. Thus, the constraints on the kick distribution represent still an open question.
## Code Packages
To initialize and evolve a N-body simulation, we use [Fireworks] (https://ca23-fireworks.readthedocs.io/en/latest/), a Python package with a series of tools for collisional/collisionless systems and orbit integration. It contains wrappers to other integrators such as [Pyfalcon] (https://github.com/GalacticDynamics-Oxford/pyfalcon) and [TSUNAMI] (https://www.cambridge.org/core/journals/proceedings-of-the-international-astronomical-union/article/modeling-gravitational-fewbody-problems-with-tsunami-and-okinami/021535C19F650C6C1DA1DAD265B1D7C0). It takes as input an instance of the class Particles with specified initial conditions and a function to estimate the acceleration from the module dynamics (or potentials). Then, we iteratively integrate until reaching the required evolution time. To insert additional physics, we can couple the simulations with the population synthesis code SEVN (Stellar EVolution N-body). It includes stellar evolution via interpolation of precomputed stellar tracks (from PARSEC and MIST), binary evolution, and dif- ferent recipes for core-collapse supernovae. In such a way, while updating the dynamics, we are evolving also the stellar properties in a reasonable amount of computational time. We have to keep in mind that once we include additional physics, the N-body simulations are not scale-free anymore: the N-body units from Fireworks must be converted into physical units for SEVN, and vice-versa. In our case, we set the proper scaling for the MW by setting the N-body units to: $M_scale$ = 1010 $M_{sun}$, $r_{scale}$ = 1000 pc = 1kpc, and $T_{scale}$ = 1000 Myr = 1 Gyr.





