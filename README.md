# ErSE 2010 Seismology - teaching material
Teaching material for ErSE 2010 Seismology course to be held at KAUST during the Fall semester.

## Material

Each folder in this repository contains python codes and Jupiter notebooks used in the practical sessions (labs):

- **PlaneWave**: create and display plane waves in time-space and wavenumber domain.
- **GassmannFluidSub**: implement basic rock physics equations and Gassmann substitution and apply it to the Smehaia well log.
- **SeismicModelling**: perform convolutional and AVO modelling, and apply pre-stack inversion
- **RayTrace**: implement 2D raytracing by solving the associated ODE
- **SeismicTomography**: setup the 2D tomographic matrix and solve the associated inverse problem
- **Obspy**: a mild introduction to Obspy and its usage for epicenter localization of earthquakes
- **ReflectionSeismic**: implement basic NMO processing and learn how to work with SEGY files using *segyio* and the Volve dataset.

## Environment

To ensure reproducibility of the results, we have provided a `environment.yml` file. Ensure to have installed Anaconda or Miniconda on your computer. If you are not familiar with it, we suggesting using the [KAUST Miniconda Install recipe](https://github.com/kaust-rccl/ibex-miniconda-install). This has been tested both on macOS and Unix operative systems.

After that simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the work `Done!` on your terminal you are ready to go!