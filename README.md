[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fterroso/climaticpark/653f1274f2e3162f3ad5486a442ffd1b9f48df53?urlpath=lab%2Ftree%2Fdemo.ipynb)

# CLIMATICPARK - A Python-based framework for the climatic simulation of vehicular parking losts

# Description

ClimaticPark is a pyhton library to simulate different environement variables of a vehicular parking lot. By means integrating different algorithms from the Deep Learning and thermodynamics fields, it is able to simulate the following parameters of a parking lot:

- The demand behavior of a PL (entry and exit hours of vehicles).
- The movement of the shadows projected by physical roofs on the PL's spaces.
- Cabin temperature of vehicles while they remain parked in the parking lot.
- The fuel consumption required by the air-conditioning system of these vehicles to cool them down based on a predefined comfort temperature

# Class Diagram

![UML class diagram]([files/Users/jzhang/Desktop/Isolated.png](https://github.com/fterroso/climaticpark/blob/main/figs/climaticpark_arch.png))




# Content

- `lib`: Includes the source code of the library.
- `demo.ipynb`: Jupyter Notebook comprising a step by step guideline to use the library.
- `data`: Includes the input data required for the step-by-step demo.
- `environment.yml`: dependencies of the library.
