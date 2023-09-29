![Seismology](https://github.com/DIG-Kaust/Seismology/blob/main/logo.png)

Teaching material for ErSE 210 Seismology course to be held at KAUST during the Fall semester.

## Material

The repository is organized as follows:

- **Slides**: deck of slides summarizing the key concept introduced in each class. Some of the figures in these slides are taken from the reference textbook (Shearer, P., Introduction to Seismology). 
- **Data**: input data used in the practical sessions:
- All of the other folders in this repository contains Python codes and Jupyter Notebooks used in the practical sessions:

   - [**PlaneWave**](https://github.com/DIG-Kaust/Seismology/blob/main/PlaneWave/PlaneWave.ipynb): create and display plane waves in time-space and wavenumber domain.
   - [**GassmannFluidSub**](https://github.com/DIG-Kaust/Seismology/blob/main/GassmannFluidSub/Gassmann.ipynb): implement basic rock physics equations and Gassmann substitution and apply it to the Smehaia well log.
   - [**SeismicModelling**](https://github.com/DIG-Kaust/Seismology/blob/main/SeismicModelling/SeismicModellingInversion.ipynb): perform convolutional and AVO modelling, and apply pre-stack inversion.
   - [**RayTrace**](https://github.com/DIG-Kaust/Seismology/blob/main/RayTrace/RayTrace.ipynb): implement 2D raytracing by solving the associated ODE.
   - [**SeismicTomography**](https://github.com/DIG-Kaust/Seismology/blob/main/SeismicTomography/SeismicTomography.ipynb): create the 2D tomographic matrix and solve the associated inverse problem.
   - [**ReflectionSeismic**](https://github.com/DIG-Kaust/Seismology/blob/main/ReflectionSeismic): implement basic NMO processing and learn how to work with SEGY files using *segyio* and the Volve dataset.
   - [**TimeMigration**](https://github.com/DIG-Kaust/Seismology/blob/main/TimeMigration/TimeMigration.ipynb): implement basic time-domain post-stack Kirchhoff demigration-migration on simple synthetic dataset.
   - [**Dispersion**](https://github.com/DIG-Kaust/Seismology/blob/main/Dispersion/Dispersion.ipynb): create a surface-wave only seismic dataset, compute dispersion panel and perform surface wave dispersion curve inversion.
   - [**Obspy**](https://github.com/DIG-Kaust/Seismology/blob/main/Obspy/ObspyIntro.ipynb): a short introduction to Obspy and its usage for epicenter localization of earthquakes


## Environment

To ensure reproducibility of the results, we have provided an `environment.yml` file. Ensure to have installed Anaconda or Miniconda on your computer. If you are not familiar with it, we suggest using the 
[KAUST Miniconda Install recipe](https://github.com/kaust-rccl/ibex-miniconda-install). This has been tested both on macOS and Unix operative systems.

After that simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the work `Done!` on your terminal you are ready to go!

## Binder

Alternatively, you can work directly on Binder. Simply click this button and access
the material from your web browser without the need for any local installation

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DIG-Kaust/Seismology/HEAD)