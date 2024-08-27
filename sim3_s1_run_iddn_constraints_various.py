# %%
# Different types of constraints
# null: allow all edges
# basic: mRNA depends on mRNA, TFs and miRNAs
# basic_mrna: specify TFs in mRNA layer (node degree > 1)
# basic_regu: limit the choice of targets for each regulator
# basic_mrna_regu: above two types

import os
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import h5py
from joblib import Parallel, delayed
from iddn_paper import tool_scan, sim3_h5op, tool_sys


# %%

exp_name = "sim3_ggm_three_layer_batch_444445"

n_rep = 32
n_sample_work = 200  # 100

rho1_rg = np.arange(0.02, 0.81, 0.02)
rho2_rg = np.arange(0.0, 0.21, 0.02)

prior_name = "basic"

top_folder = tool_sys.get_work_folder() + "/experiment_iddn_paper/"
dat_file = f"{top_folder}/sim_input/{exp_name}.hdf5"

dat1, dat2, con1, con2, comm, diff, _, dep_mat_prior, _, layer_count = (
    sim3_h5op.read_sim_data(dat_file)
)

# %%


# %%
res = Parallel(n_jobs=16)(
    delayed(tool_scan.scan2_iddn)(
        dat1[n],
        dat2[n],
        rho1_rg,
        rho2_rg,
        dep_mat=dep_mat_prior[n],
        n_sample_work=n_sample_work,
        n=n,
    )
    for n in range(n_rep)
)
res_mat = np.array(res)

# %%

res_file = f"{top_folder}/sim_output/{exp_name}_iddn_sample_{n_sample_work}_sigma_0.0_prior_{prior_name}.hdf5"

f = h5py.File(res_file, "w")
f.create_dataset("dep_est", data=res_mat, compression="gzip")
f.close()
