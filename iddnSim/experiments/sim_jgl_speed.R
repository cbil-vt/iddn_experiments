# ----
library(rhdf5)
library(JGLtry)  # The JGL package in code_extra/ddn_peers/

sim_data_folder <- "../../sim_iddn_data/sim_input/"

exp_name_lst = c(
  "sim3_ggm_three_layer_v2_batch_2024_08_12_17_13_02",  # 50
  "sim3_ggm_three_layer_v2_batch_2024_08_12_16_16_18",  # 100
  "sim3_ggm_three_layer_v2_batch_2024_08_12_16_21_11",  # 200
  "sim3_ggm_three_layer_v2_batch_2024_08_12_16_44_16"  # 400
)

n_rep = 20
n_sample_work = 100
sigma_add = 0.0

l1_rg = c(0.2)
l2_rg = c(0.05)

# ----
run_time = matrix(0, length(exp_name_lst), n_rep)

for (i in seq(length(exp_name_lst))) {
  exp_name = exp_name_lst[i]
  sim_filename = paste(sim_data_folder, exp_name, ".hdf5", sep = "")
  print(h5ls(sim_filename))
  dat1 = h5read(sim_filename, "dat1")
  dat2 = h5read(sim_filename, "dat2")
  dat1 <- aperm(dat1, c(3,2,1))
  dat2 <- aperm(dat2, c(3,2,1))

  for (n in seq(n_rep)) {
    print(i, n)
    start.time = Sys.time()
    res = jgl_wrap(
      n=n, dat1=dat1, dat2=dat2, l1_rg=l1_rg, l2_rg=l2_rg,
      n_sample_work=n_sample_work, sigma_add=sigma_add
    )
    end.time <- Sys.time()
    run_time[i, n] = end.time - start.time
  }
}

# --
print(rowMeans(run_time))



