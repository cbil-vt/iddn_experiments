# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import h5py
from joblib import Parallel, delayed
from iddn_paper import tool_scan, sim3_h5op, tool_sys


# %%

exp_name = "sim3_direction_two_layer_batch_428416_lin"

n_rep = 32
n_sample_work = 50  # 100
sigma_add = 0.0  # 2.0
rho1_rg = np.arange(0.02, 0.81, 0.02)
rho2_rg = np.arange(0.0, 0.21, 0.02)

# msk_idx = 0  # Direction prior
msk_idx = 1  # Symmetric prior

top_folder = tool_sys.get_work_folder() + "/experiment_iddn_paper/"
dat_file = f"{top_folder}/sim_input/{exp_name}.hdf5"

(
    dat1,
    dat2,
    con_mat1,
    con_mat2,
    comm_gt,
    diff_gt,
    dep_mat_null,
    dep_mat_prior_precise,
    dep_mat_prior_loose,
    layer_count,
) = sim3_h5op.read_sim_data(dat_file)

# %%

if msk_idx == 0:
    dep_mat_prior = dep_mat_prior_precise
else:
    dep_mat_prior = dep_mat_prior_loose

res = Parallel(n_jobs=16)(
    delayed(tool_scan.scan2_iddn)(
        dat1[n],
        dat2[n],
        rho1_rg,
        rho2_rg,
        dep_mat=dep_mat_prior[n],
        n_sample_work=n_sample_work,
        sigma_add=sigma_add,
        n=n,
    )
    for n in range(n_rep)
)
res_mat = np.array(res)

# %%

res_file = f"{top_folder}/sim_output/{exp_name}_iddn_sample_{n_sample_work}_sigma_{sigma_add}_msk_{msk_idx}.hdf5"

f = h5py.File(res_file, "w")
f.create_dataset("dep_est", data=res_mat, compression="gzip")
f.close()
