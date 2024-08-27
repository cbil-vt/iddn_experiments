# Simulation studies in iDDN paper

This repository is about the code to perform the simulation and draw the 
figures related to simulations for iDDN.
The main repository of DDN 3.0 is at https://github.com/cbil-vt/iDDN.
The results of the simulation are saved at Zenodo.

## Installation
Install DDN3.0 in develop mode.

Users need to Install R locally, and edit the path to R in `tools_r.py`.

```python
import os
os.environ["R_HOME"] = "C:\\Program Files\\R\\R-4.3.2"
```

Required Python packages: `numpy`, `matplotlib`, `joblib`, `rpy2`, `ddn3`, and their dependencies.

Required R packages: `iDINGO`, `glasso`, `huge`, and their dependencies.
We call the `huge` package to generate synthetic data. `JGL` and `iDINGO` contain peer methods.

Install the `JGL` version which corrects a numerical error.


