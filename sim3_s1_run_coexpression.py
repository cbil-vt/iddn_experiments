# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import h5py
from iddn_paper import sim3_h5op, tool_sys

exp_name = "sim3_ggm_three_layer_v2_batch_2024_08_07_22_31_38"

n_rep = 50
n_sample_work = 200  # 100
sigma_add = 0.0  # 2.0
rho1_rg = np.arange(0.025, 1.01, 0.025)

# %%

top_folder = tool_sys.get_work_folder() + "/experiment_iddn_paper/"
dat_file = f"{top_folder}/sim_input/{exp_name}.hdf5"

(
    dat1,
    dat2,
    con_mat1,
    con_mat2,
    comm_gt,
    diff_gt,
    _,
    _,
    _,
    _,
) = sim3_h5op.read_sim_data(dat_file)

# %%
n_feature = dat1[0].shape[1]
res_mat = np.zeros((n_rep, len(rho1_rg), 1, 2, n_feature, n_feature))

# n_sample, n_feature = dat1[0].shape

for n in range(n_rep):
    print(n)
    # idx1 = np.random.choice(n_sample, n_sample_work, replace=False)
    # idx2 = np.random.choice(n_sample, n_sample_work, replace=False)
    # dat1_sel = dat1[idx1, :]
    # dat2_sel = dat2[idx2, :]
    dat1_sel = dat1[n][:n_sample_work, :]
    dat2_sel = dat2[n][:n_sample_work, :]
    dat1_sel = dat1_sel + np.random.normal(0, sigma_add, dat1_sel.shape)
    dat2_sel = dat2_sel + np.random.normal(0, sigma_add, dat2_sel.shape)
    c1 = np.abs(np.corrcoef(dat1_sel.T))
    c2 = np.abs(np.corrcoef(dat2_sel.T))
    for idx, rho1 in enumerate(rho1_rg):
        c1_bin = 1 * (c1 > rho1)
        c2_bin = 1 * (c2 > rho1)
        res_mat[n, idx, 0, 0] = c1_bin
        res_mat[n, idx, 0, 1] = c2_bin

# %%
res_file = f"{top_folder}/sim_output/{exp_name}_coexpression_sample_{n_sample_work}_sigma_{sigma_add}.hdf5"

f = h5py.File(res_file, "w")
f.create_dataset("dep_est", data=res_mat, compression="gzip")
f.close()
