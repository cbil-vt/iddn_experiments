# %%
# Give iDDN part of the omics iddn_data,
# To show that using more omics types (miRNA, lncRNA) can improve the estimation of TF-TF-mRNA network
# 0  1  2  3  4  5  6  7  8  9  10
# 0  0  10 20 30 40 50 60 70 80 90

import os
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import h5py
from joblib import Parallel, delayed
from iddn_paper import tool_scan, sim3_h5op, tool_sys


def make_sub_prior(dep_mat_prior_in, node_range):
    dep_mat_prior = np.copy(dep_mat_prior_in)
    n_node = dep_mat_prior.shape[-1]
    msk_sub = np.zeros((n_node, n_node))
    msk_sub[np.ix_(node_range, node_range)] = 1

    for i in range(len(dep_mat_prior)):
        dep = dep_mat_prior[i] * msk_sub
        dep_mat_prior[i] = dep

    return dep_mat_prior


def run_iddn_for_one_case(
    dat1,
    dat2,
    rho1_rg,
    rho2_rg,
    dep_mat_prior_in,
    node_range,
    n_sample_work,
    n_rep,
    out_name="tf_mrna",
):
    dep_mat_prior = make_sub_prior(dep_mat_prior_in, node_range)
    res = Parallel(n_jobs=8)(
        delayed(tool_scan.scan2_iddn)(
            dat1[n],
            dat2[n],
            rho1_rg,
            rho2_rg,
            dep_mat=dep_mat_prior[n],
            n_sample_work=n_sample_work,
            sigma_add=0.0,
            n=n,
        )
        for n in range(n_rep)
    )
    res_mat = np.array(res)

    res_file = f"{top_folder}/sim_output/{exp_name}_iddn_sample_{n_sample_work}_sigma_{sigma_add}_msk_0_{out_name}.hdf5"
    # res_file = f"{top_folder}/sim_output/{exp_name}_iddn_sample_{n_sample_work}_sigma_{sigma_add}_prior_null_{out_name}.hdf5"

    f = h5py.File(res_file, "w")
    f.create_dataset("dep_est", data=res_mat, compression="gzip")
    f.close()


# %%
# Load iddn_data

# exp_name = "sim3_ggm_four_layer_batch_987306"

# 30+30+30+30, 8 edges from miRNA and lncRNA
# exp_name = "sim3_ggm_four_layer_batch_785474"

# 50+50+50, 8 edges from miRNA and TF to mRNA
# exp_name = "sim3_ggm_three_layer_batch_9406"
# exp_name = "sim3_ggm_three_layer_batch_444445"

# Three layer v2
# exp_name = "sim3_ggm_three_layer_v2_batch_2024_08_07_22_31_38"
exp_name = "sim3_ggm_three_layer_v2_batch_2024_08_07_16_41_06"

n_rep = 32
n_sample_work = 200
sigma_add = 0.0
rho1_rg = np.arange(0.02, 0.81, 0.02)
rho2_rg = np.arange(0.0, 0.11, 0.01)
# rho2_rg = np.arange(0.0, 0.21, 0.02)

top_folder = tool_sys.get_work_folder()
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

idx1 = layer_count[0]
idx2 = idx1 + int(layer_count[1])
idx3 = idx2 + int(layer_count[2])

mrna_range = list(range(idx1))
tf_range = list(range(idx1, idx2))
mirna_range = list(range(idx2, idx3))

dep_mat_in = dep_mat_null
# dep_mat_in = dep_mat_prior_precise


# %%
# TF-mRNA only

run_iddn_for_one_case(
    dat1,
    dat2,
    rho1_rg,
    rho2_rg,
    dep_mat_in,
    mrna_range,
    n_sample_work,
    n_rep,
    out_name="mrna",
)

# %%
# TF-mRNA + miRNA

run_iddn_for_one_case(
    dat1,
    dat2,
    rho1_rg,
    rho2_rg,
    dep_mat_in,
    mrna_range + tf_range,
    n_sample_work,
    n_rep,
    out_name="mrna_tf",
)


# %%
# TF-mRNA + lncRNA

run_iddn_for_one_case(
    dat1,
    dat2,
    rho1_rg,
    rho2_rg,
    dep_mat_in,
    mrna_range + mirna_range,
    n_sample_work,
    n_rep,
    out_name="mrna_mirna",
)


# %%
# TF-mRNA + miRNA + lncRNA

run_iddn_for_one_case(
    dat1,
    dat2,
    rho1_rg,
    rho2_rg,
    dep_mat_in,
    mrna_range + tf_range + mirna_range,
    n_sample_work,
    n_rep,
    out_name="mrna_tf_mirna",
)
