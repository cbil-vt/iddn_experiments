# %%
# Use steady state simulation to generate samples
# Then approximate the resulting correlation matrix with GGM

import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

from ddn3 import simulation
from iddn_paper import sim3_h5op, tool_sys, sim3_network_toy
from iddn_paper.old_functions import sim3_network_topo as nett, sim3_ode

# %%
# ODE based simulation

n_sample_gen = 10000
sigma_in, sigma_mid = 2.0, 2.0  # 0.1
hill_coef = 2.0  # 1.0

top_folder = tool_sys.get_work_folder() + "/experiment_iddn_paper/"
exp_name = f"sim3_toy_constraints_node_182_hill_{hill_coef}_sigma_{sigma_in}_{sigma_mid}_common_n_{n_sample_gen}"
dat_file = f"{top_folder}/sim_input/{exp_name}.hdf5"

# if not os.path.exists(dat_file):
# build network
n_hub = 1
hub_to_gene = 10
hub_to_tf = 20
tf_to_gene = 3

mol_layer = dict()
mol_par = dict()
mol_par_roles = dict()

for n in range(n_hub):
    sim3_network_toy.add_one_hub_net(
        mol_layer,
        mol_par,
        mol_par_roles,
        hub_to_tf,
        hub_to_gene,
        tf_to_gene,
        net_idx=n,
    )

idx_layer, idx_par, idx_par_roles, mol2idx, idx2mol, layer_cnt = (
    nett.mol_network_to_index(mol_layer, mol_par, mol_par_roles)
)
_, con_mat = nett.get_dep_mat(idx_par)
trans_mat = nett.get_translation_mat(idx_par, idx_par_roles)
con_mat1 = con_mat
con_mat2 = con_mat

# generate samples
dep_mat_null = np.ones_like(con_mat)
dep_mat_prior = sim3_network_toy.make_iddn_dep_prior(mol_layer)
dat_toy_org, state_history, noise_history = sim3_ode.run_sim(
    idx_layer,
    idx_par,
    idx_par_roles,
    n_sample=n_sample_gen * 2,
    n_max_steps=100,
    method="steady",
    sigma_in=sigma_in,
    sigma_mid=sigma_mid,
    hill_coef=hill_coef,
)

# %%
# Save to HDF5
reload(sim3_h5op)
dat_toy = dat_toy_org
dat1 = dat_toy[:n_sample_gen, :]
dat2 = dat_toy[n_sample_gen:, :]

sim3_h5op.make_new_sim_data(
    dat_file,
    dat1,
    dat2,
    con_mat1,
    con_mat2,
    dep_mat_null=dep_mat_null,
    dep_mat_prior=dep_mat_prior,
)

# dat1, dat2, con_mat1, con_mat2, comm_gt, diff_gt, dep_mat_null, dep_mat_prior = sim3_h5op.read_sim_data(dat_file)


# %%
# GGM based on ODE correlation matrix
# Please manually set the desired threshold

dat_toy = np.vstack((dat1, dat2))
cc = np.corrcoef(dat_toy.T)
omega_org = np.linalg.inv(cc)
omega_org[np.abs(omega_org) < 0.4] = 0
omega1 = np.copy(omega_org)
omega2 = np.copy(omega_org)
plt.figure()
plt.imshow(np.abs(omega_org) > 0)
plt.show()


# %%
# Simulate GGM

g1_cov, g2_cov, comm_gt, diff_gt = simulation.prep_sim_from_two_omega(omega1, omega2)
con_mat1 = 1 * (np.abs(omega1) > 1e-8)
con_mat2 = 1 * (np.abs(omega2) > 1e-8)
n_node = len(omega1)
con_mat1[np.arange(n_node), np.arange(n_node)] = 0
con_mat2[np.arange(n_node), np.arange(n_node)] = 0

n_sample_gen = 10000
n1, n2 = n_sample_gen, n_sample_gen
dat1, dat2 = simulation.gen_sample_two_conditions(g1_cov, g2_cov, n1, n2)

f_out = f"{exp_name}_ggm"
dat_file = f"{top_folder}/sim_input/{f_out}.hdf5"

dep_mat = np.ones((n_node, n_node))
dep_mat_null = dep_mat
dep_mat_prior = dep_mat

sim3_h5op.make_new_sim_data(
    dat_file,
    dat1,
    dat2,
    con_mat1,
    con_mat2,
    comm_gt=comm_gt,
    diff_gt=diff_gt,
    dep_mat_null=dep_mat_null,
    dep_mat_prior=dep_mat_prior,
    layer_count=layer_cnt,
)
