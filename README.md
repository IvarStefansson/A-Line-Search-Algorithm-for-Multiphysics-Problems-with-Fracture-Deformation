This repository contains run scripts for the paper "A Line Search Algorithm for Multiphysics Problems with Fracture Deformation".
The code is structured as follows:
- Each of the two run scripts contain the definition of the runs for the test cases in either of the section 4.1 and 4.2.
- Material parameters, boundary conditions and geometries are defined in intuitively named files.
- Solution parameters etc. are defined in `common_params.py`.
- The (physical) model setup is collected in `model_setup.py`.
- The solution strategy extensions relative to [PorePy](https://github.com/pmgbergen/porepy) are defined in `line_search.py`.
- Remaining files contain utilities for running simulations, visualising etc.

For version 1.0, please use PorePy commit: 9449a76f42d2615785f23ede575ce16e108c26f and commit: 9322b1a232155b50776b139e2ae442a7693d30c8 for this repository.
