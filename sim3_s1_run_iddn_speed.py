# %%
import time
import numpy as np
from iddn_paper import tool_scan, sim3_h5op, tool_sys


# %%

exp_name_lst = [
    "sim3_ggm_three_layer_v2_batch_2024_08_12_17_13_02",  # 50
    "sim3_ggm_three_layer_v2_batch_2024_08_12_16_16_18",  # 100
    "sim3_ggm_three_layer_v2_batch_2024_08_12_16_21_11",  # 200
    "sim3_ggm_three_layer_v2_batch_2024_08_12_16_44_16",  # 400
]

runtime_lst = []

n_rep = 20
n_sample_work = 100
sigma_add = 0.0
rho1_rg = [0.2]
rho2_rg = [0.05]
msk_level = 9

# %%

run_time = np.zeros((len(exp_name_lst), n_rep))

for i, exp_name in enumerate(exp_name_lst):
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

    print(dat1.shape)

    dep_mat_prior = dep_mat_prior_loose[:, msk_level]

    # Pre-run for JIT
    _ = tool_scan.scan2_iddn(
        dat1[0],
        dat2[0],
        rho1_rg,
        rho2_rg,
        dep_mat=dep_mat_prior[0],
        n_sample_work=n_sample_work,
        sigma_add=sigma_add,
        n=0,
        mthd="resi",
    )

    for n in range(n_rep):
        print(exp_name, n)
        start_time = time.time()
        res0 = tool_scan.scan2_iddn(
            dat1[n],
            dat2[n],
            rho1_rg,
            rho2_rg,
            dep_mat=dep_mat_prior[n],
            n_sample_work=n_sample_work,
            sigma_add=sigma_add,
            n=n,
            mthd="resi",
        )
        run_time[i, n] = time.time() - start_time

# %%
print(np.mean(run_time, axis=1))
