# %%
# Use steady state simulation to generate samples
# Then approximate the resulting correlation matrix with GGM

import numpy as np
from importlib import reload
from ddn3 import tools
from iddn_paper import sim3_h5op, tool_sys, sim3_network_toy
from iddn_paper.old_functions import sim3_network_topo as nett, sim3_steady_state_batch

top_folder = tool_sys.get_work_folder() + "/experiment_iddn_paper/"

# %%
# build network
reload(nett)
reload(sim3_network_toy)

n_sample_gen = 10000  # 10000
sigma_in = 2.0
sigma_mid = 2.0

hill_coef = 1.0  # 1.0
two_condition_ratio = 0.25  # 0.25

n_hub = 1
hub_to_gene = 10
hub_to_tf = 20
tf_to_gene = 3

# Skeleton
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

# Make two conditions by removing some edges in each condition
mol_par1, mol_par2, mol_par_roles1, mol_par_roles2 = nett.make_two_conditions_mol_net(
    mol_par,
    mol_par_roles,
    ratio=two_condition_ratio,
)

# Simulation for each condition

net_info, dep_mat, con_mat = nett.prep_net_for_sim(
    mol_layer,
    mol_par,
    mol_par_roles,
    mol_type=None,
)
net_info1, dep_mat1, con_mat1 = nett.prep_net_for_sim(
    mol_layer,
    mol_par1,
    mol_par_roles1,
    mol_type=None,
)
net_info2, dep_mat2, con_mat2 = nett.prep_net_for_sim(
    mol_layer,
    mol_par2,
    mol_par_roles2,
    mol_type=None,
)
dep_mat_null = np.ones_like(dep_mat1)
dep_mat_prior = sim3_network_toy.make_iddn_dep_prior(mol_layer, net_info1["mol2idx"])
comm_gt, diff_gt = tools.get_common_diff_net_topo([con_mat1, con_mat2])

sigma_mid_vec = np.zeros(len(mol_layer)) + sigma_mid
# sigma_mid_vec = (np.random.rand(len(mol_layer)) * 2 - 1) + sigma_mid


# %%
# generate samples
reload(sim3_steady_state_batch)
dat1, state_history1, noise_history1 = sim3_steady_state_batch.run_sim(
    net_info1["idx_layer"],
    net_info1["idx_par"],
    net_info1["idx_par_roles"],
    n_sample=n_sample_gen,
    n_max_steps=100,
    method="steady",
    sigma_in=sigma_in,
    sigma_mid=sigma_mid_vec,
    hill_coef=hill_coef,
)
dat2, state_history2, noise_history2 = sim3_steady_state_batch.run_sim(
    net_info2["idx_layer"],
    net_info2["idx_par"],
    net_info2["idx_par_roles"],
    n_sample=n_sample_gen,
    n_max_steps=100,
    method="steady",
    sigma_in=sigma_in,
    sigma_mid=sigma_mid_vec,
    hill_coef=hill_coef,
)

# %%
cc = np.corrcoef(dat1.T)
omega_org = np.linalg.inv(cc)
# omega_org[np.abs(omega_org) < 0.4] = 0
# omega1 = np.copy(omega_org)
# omega2 = np.copy(omega_org)
# plt.figure(); plt.imshow(np.abs(omega_org) > 0); plt.show()
# plt.figure(); plt.imshow(np.abs(cc)); plt.show()

# %%
# Save to HDF5

# v1 allows removing the translation edges
# This makes evaluation of differential network easier
exp_name = (
    f"sim3_tf_mrna_n-node_{len(con_mat1)}_hill_{hill_coef}_sigma_{sigma_in}_{sigma_mid}"
    f"_ratio_{two_condition_ratio}_n_{n_sample_gen}_batch"
)
dat_file = f"{top_folder}/sim_input/{exp_name}.hdf5"

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
    layer_count=net_info1["layer_count"],
)
