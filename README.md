# Simulation in iDDN paper

This repository contains code to perform the simulation and draw the 
figures related to simulations for iDDN.
The main repository of iDDN is at https://github.com/cbil-vt/iDDN.

The results of the simulation, as well as the input data and figures, 
can be downloaded at [Zenodo](https://zenodo.org/records/13381083).

## Installation
Since these simulations involve both Python and R, 
and some third-party packages need to be modified to run, there are a number of things to install.

Before we begin, install `Conda` and make a new environment.
```bash
conda create -n iddn python=3.11
conda activate iddn
```

### Install Python package
We need some functions in `DDN3` package to generate synthetic data.
Download the package [here](https://github.com/cbil-vt/DDN3), unzip, 
go ton the folder, and install in develop mode
```bash
pip install -e ./`
```
If you want to use `DDN3` in develop mode, you need to install it before `iDDN` below.
If there is no need to change code in `DDN3`, installing `iDDN` will download the version from PyPI.

Download the `iDDN` package [here](https://github.com/cbil-vt/iDDN), unzip, 
go ton the folder, and install in develop mode
```bash
pip install -e ./`
```

Most functions needed to generate synthetic data and run simulations are organized 
in `iddn_paper` package.

Download the package [here](https://github.com/cbil-vt/iddn_paper), unzip, 
go to the folder, and install in develop mode
```bash
pip install -e ./`
```

### Install R and set path
Users need to Install R. We use version `4.4.1`. RStudio is also needed.
For instructions, see [here](https://posit.co/download/rstudio-desktop/).
After that, go to the DDN3 folder, in `src/ddn3_extra`, open `tools_r.py`.
Then edit the path to R in `tools_r.py`. For example:

```python
import os
os.environ["R_HOME"] = "C:\\Program Files\\R\\R-4.4.1"
```

### Install R packages

Install the required R packages: `iDINGO`, `glasso`, `huge`, `devtools`.
We call the `huge` package to generate synthetic data.

There are some numerical issues in the `JGL` package on CRAN, so we need a modifed version.
Download it [here](https://github.com/cbil-vt/jgl_modified) and unzip.
Then use RStudio to open the project (`JGL.Rproj`). Then in the `build` tab, click `install`.

We modified `admm_iters.r` by adding the `symmetric=TRUE` argument 
in the `eigen` function to avoid getting imaginary values.

### Download the data and set the path

Download the data at [Zenodo](https://zenodo.org/records/13381083).
We need two zip files, [sim_input](https://zenodo.org/records/13381083/files/sim_input.zip?download=1)
and [sim_output](https://zenodo.org/records/13381083/files/sim_output.zip?download=1).
Make a folder of any name, like `sim_iddn_data`, then unzip these two zip files inside it.

If you wish to generate synthetic data your self, you do not need to download these files.
But you still need to make a folder (of any name), 
and make two subfolders `sim_input` and `sim_output` inside it.
The generated synthetic data will be saved in `sim_input`, 
and the output will be saved in `sim_output`.
These two folders will be used by simulation using R or Python.

We keep the input and output data for the simulation outside the code repository,
so we need to know the path of these files.
In your `HOME` folder, make a file named `ddn_cfg.txt`.
Inside that file, set `ddn` to the folder we just created.
For example:

```text
ddn=C:/work/sim_iddn_data/
```

In Microsoft Windows, the `HOME` folder can be something like `C:\Users\your_user_name\`.

## Generating synthetic data

We use the Jupyter notebook `sim3_s0_gen_ggm_three_layers.ipynb` to generate the synthetic data.

After importing the needed modules, go the section `Three layer v2`, 
adjust the parameters if needed. For example, changing the number of nodes in 
each of the three layers, adjusting the connectivity, the partial correlation strength
and the repeating number. As the network topology will be different for each repeat,
the data and ground truth are all different.

Then run the cells in this section, and the output file will be saved in the
`sim_input` folder specified.

You can also use the data already generated by us. In the paper, we use two files:
- Each regulator connects to 5 mRNAs: `sim3_ggm_three_layer_v2_batch_2024_08_07_16_41_06.hdf5`
- Each regulator connects to 2 mRNAs: `sim3_ggm_three_layer_v2_batch_2024_08_07_22_31_38.hdf5`

## Running simulation for Python based methods
We have one script for each method we need to compare.
- `sim3_s1_run_ddn.py`: applying DDN3.0.
- `sim3_s1_run_iddn.py`: applying iDDN.
- `sim3_s1_run_coexpression.py`: co-expression, which apply different thresholds on the correlation matrix. 
- `sim3_s1_run_iddn_constraints_various.py`: iDDN with different constraint levels.
- `sim3_s1_run_iddn_subset.py`: iDDN with different number of omics types.
- `sim3_s1_run_iddn_speed.py`: speed comparison based on feature number.
- `sim3_s1_run_iddn_speed_sample_size.py`: speed comparison based on sample size.

The usage of these scripts are similar. We will use `sim3_s1_run_iddn.py` as an example.
For each simulation, we can set some running parameters.

```python
import numpy as np
exp_name = "sim3_ggm_three_layer_v2_batch_2024_08_07_16_41_06"
n_rep = 50
n_sample_work = 200  # 100
rho1_rg = np.arange(0.02, 0.81, 0.02)
rho2_rg = np.arange(0.0, 0.16, 0.01)
msk_level_rg = [3, 6, 9]
```

We first need to specify `exp_name`, and do not include the extension `hdf5`.
Then we set how much time we need to repeat the simulation; more repeats make the results more stable.
The sample size for each condition is `n_sample_work`.
The list of $\lambda_1$ is `rho1_rg` and list of $\lambda_2$ is `rho2_rg`.

Besides, we can also adjust the number of cores used in the simulation by adjusting `n_jobs`.

As we generated several constraints matrices, we can choose which one we will simulate.
For reference, usually we generate 10 constraints for each repeat in the simulation:
- `0`: no constraints
- `1`: constraints on layer level. For example, we do not allow edges among miRNAs
- `2` to `10`: we remove 10% to 10% elements in the matrix that are not edges. More means stronger prior knowledge.

Then run the simulation by running the script. The output will be saved in `sim_output` folder.
The output file will contain the name of the input, as well as the method name.

## Running simulation for R based methods

We compare `JGL` and `iDINGO`, which are in R. 
To run the simulations, go to the `iddnSim` folder in this repository, 
open the project using RStudio. 
Make sure the `devtools` package is installed, then use `Ctro`+`Shift`+`L` keys to load the package.

Go to the `experiments` folder, there are the scripts of the simulations.
- `sim_jgl.R`: applying JGL
- `sim_idingo_three_layer_as_two`: applying iDINGO, and treat the three layers in the synthetic data as two.
- `sim_jgl_speed`: speed comparison for JGL about the feature size
- `sim_jgl_speed_sample.R`: speed comparison for JGL about the sample size

We will use `sim_jgl.R` as an example to show the usage.
We need to adjust some parameters before running the simulation.

```R
sim_data_folder <- "C:/work/sim_iddn_data/sim_input/"
sim_out_folder <- "C:/work/sim_iddn_data/sim_output/"
exp_name <- "sim3_ggm_three_layer_v2_batch_2024_08_07_22_31_38"
n_rep <- 32
n_sample_work <- 200
l1_rg <- seq(0.02, 0.8, by = 0.02)
l2_rg <- seq(0.0, 0.16, by = 0.01)
```

Compared to the Python version, the only difference is that we need to manually 
specify the folder containing the synthetic data (`sim_data_folder`), as well as the output folder
`sim_out_folder`. The meaning of all other parameters ar the same as the Python version.
The output format is also (almost) the same; note that a transpose is needed for R version of HDF5.
To adjust the number of cores used, change the number in `makeCluster(16)`

To run the simulation, click `Source` button in RStudio. 
It will take a while for JGL and iDINGO to finish (usually hours).

## Analyzing simulation results

After running the simulation, we can use `sim3_s2_analysis_draw_v2.ipynb` to calculate the errors
and draw the figures. 
We can choose the name of synthetic data, and no other parameter setting is needed to reproduce the 
figures in the main body and the supplementary of the paper.
