# %%
# Ablation studies for accuracy and speed
# Accuracy: l2 penalty and constraints
# Speed: parallel and constraints
#
# null: allow all edges
# all: precise prior

import time
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import h5py
from joblib import Parallel, delayed
from iddn_paper import tool_scan, sim3_h5op

# %%

node = 5
if node == 5:
    exp_name = "sim3_ggm_three_layer_v2_batch_2024_08_07_16_41_06"
else:
    exp_name = "sim3_ggm_three_layer_v2_batch_2024_08_07_22_31_38"

n_sample_work = 200  # 100
n_rep = 50

top_folder = "../../iddn_data/"
dat_file = f"{top_folder}/sim_input/{exp_name}.hdf5"

dat1, dat2, con1, con2, comm, diff, _, _, dep_mat_loose, _ = (
    sim3_h5op.read_sim_data(dat_file)
)

# %%
# 0: null
# 1: basic
# 8: a lot

n_cores = 5
prior_idx = 8
rho1_rg = np.array([0.15])
rho2_rg = np.array([0.05])

dep_mat = dep_mat_loose[:,prior_idx]

# %%

t0 = time.time()
res = Parallel(n_jobs=n_cores, verbose=10)(
    delayed(tool_scan.scan2_iddn)(
        dat1[n],
        dat2[n],
        rho1_rg,
        rho2_rg,
        dep_mat=dep_mat[n],
        n_sample_work=n_sample_work,
        n=n,
    )
    for n in range(n_rep)
)
res_mat = np.array(res)
t1 = time.time()
print("Running time: ", t1-t0)

# %%

# res_file = f"{top_folder}/rev0_sim_output/{exp_name}_iddn_sample_{n_sample_work}_sigma_0.0_prior_{prior_idx}.hdf5"
# f = h5py.File(res_file, "w")
# f.create_dataset("dep_est", data=res_mat, compression="gzip")
# f.close()

# %%
# accuracy

def get_err(mat_est, mat_gt):
    n_est = np.sum(mat_est)
    n_gt = np.sum(mat_gt)
    n_over = np.sum((mat_est+mat_gt)==2)
    prec = n_over/n_est
    recall = n_over/n_gt
    if prec>0 or recall>0:
        f1 = 2*prec*recall/(prec+recall)
    else:
        f1 = 0.0
    return prec, recall, f1

res = res_mat[:,0,0]

err_arr = np.zeros((n_rep,6))
for n in range(n_rep):
    net0 = res[n,0]
    net1 = res[n,1]

    net0 = 1*(np.abs(net0)>1e-4)
    net1 = 1*(np.abs(net1)>1e-4)

    com_now = 1*((net0+net1)==2)
    dif_now = 1*(net0!=net1)

    com_gt_now = comm[n]
    dif_gt_now = diff[n]

    err_arr[n,:3] = get_err(com_now, com_gt_now)
    err_arr[n,3:] = get_err(dif_now, dif_gt_now)

np.mean(err_arr, axis=0)
