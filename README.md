[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fterroso/climaticpark/0d818f2dd89655ee5afd73d8594ff7d338f874f0?urlpath=lab%2Ftree%2Fdemo.ipynb)

# CLIMATICPARK - A Python-based framework for the climatic simulation of vehicular parking losts

# Description

ClimaticPark is a pyhton library to simulate different environement variables of a vehicular parking lot. By means integrating different algorithms from the Deep Learning and thermodynamics fields, it is able to simulate the following parameters of a parking lot:

- The demand behavior of a PL (entry and exit hours of vehicles).
- The movement of the shadows projected by physical roofs on the PL's spaces.
- Cabin temperature of vehicles while they remain parked in the parking lot.
- The fuel consumption required by the air-conditioning system of these vehicles to cool them down based on a predefined comfort temperature

# Content

- `lib`: Includes the source code of the library.
- `demo.ipynb`: Jupyter Notebook comprising a step by step guideline to use the library.
- `data`: Includes the input data required for the step-by-step demo.
- `figs`: Includes additional figures that describe the library.
- `environment.yml`: dependencies of the library.

# Class Diagram

The detailed UML class diagram of the library is provided next,

![image-url](https://github.com/fterroso/climaticpark/blob/main/figs/climaticpark_arch.png)
