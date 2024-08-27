library(rhdf5)
library(parallel)
library(JGLtry)  # The JGL package in code_extra/ddn_peers/

# Settings ----

sim_data_folder <- "../../sim_iddn_data/sim_input/"

exp_name = "sim3_ggm_three_layer_v2_batch_2024_08_07_22_31_38"

n_rep = 32
n_sample_work = 200
sigma_add = 0.0  # 2.0

l1_rg = seq(0.02, 0.8, by = 0.02)
l2_rg = seq(0.0, 0.16, by = 0.01)
# l2_rg = seq(0.0, 0.2, by = 0.02)

# Load data ----
# R and Python HDF5 packages use opposite dimension orders

sim_filename = paste(sim_data_folder, exp_name, ".hdf5", sep = "")
print(h5ls(sim_filename))
dat1 = h5read(sim_filename, "dat1")
dat2 = h5read(sim_filename, "dat2")
dat1 <- aperm(dat1, c(3,2,1))
dat2 <- aperm(dat2, c(3,2,1))


# Run JGL ----

cl <- makeCluster(16)
clusterExport(cl, c("dat1", "dat2", "l1_rg", "l2_rg", "n_sample_work", "sigma_add", "jgl_wrap", "JGL"))
clusterEvalQ(cl, sink(paste0("../temp/output", Sys.getpid(), ".txt")))
res0 <- parLapply(cl, 1:n_rep, jgl_wrap,
                  dat1=dat1, dat2=dat2, l1_rg=l1_rg, l2_rg=l2_rg,
                  n_sample_work=n_sample_work, sigma_add=sigma_add)
stopCluster(cl)
dep_est_arr = collect_parallel_res(res0)

# Save results ----

fname = paste("../../sim_iddn_data/sim_output/", exp_name, "_jgl_sample_", n_sample_work,
              "_sigma_", sprintf("%.1f",sigma_add), ".hdf5", sep = "")
h5createFile(fname)
h5write(dep_est_arr, fname, "dep_est")


