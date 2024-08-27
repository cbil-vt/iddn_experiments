library(rhdf5)
library(JGLtry)

# Settings ----

sim_data_folder <- "../../sim_iddn_data/sim_input/"
sim_name = "sim3_tf_mrna_mirna_n_124_hill_1.0_sigma_unif_ratio_0.25_492870_ggm"

n_sample_work = 200  # 100
# sigma_add = 0.0  # 0

# Load data ----
# R and Python HDF5 packages use opposite dimension orders

sim_filename = paste(sim_data_folder, sim_name, ".hdf5", sep = "")
print(h5ls(sim_filename))
dat1 = t(h5read(sim_filename, "dat1"))
dat2 = t(h5read(sim_filename, "dat2"))
con_mat1 = t(h5read(sim_filename, "con_mat1"))
# dep_mat_prior = t(h5read(sim_filename, "dep_mat_prior"))

# Run JGL ----
n_sample = dim(dat1)[1]
n_feature = dim(dat1)[2]

# lambda1, lambda2, conditions, feature, feature
idx1 = sample(n_sample, n_sample_work, replace = FALSE)
idx2 = sample(n_sample, n_sample_work, replace = FALSE)

# Must scale data first
dat1_sel = dat1[idx1, 1:n_feature]
dat2_sel = dat2[idx2, 1:n_feature]
# dat1_sel = dat1_sel + array(rnorm(n_sample_work*n_feature)*sigma_add, c(n_sample_work,n_feature))
# dat2_sel = dat2_sel + array(rnorm(n_sample_work*n_feature)*sigma_add, c(n_sample_work,n_feature))

dat1_sel = scale(dat1_sel)
dat2_sel = scale(dat2_sel)
dat_lst = list(dat1_sel, dat2_sel)

# Run ----
l1 = 0.5
l2 = 0.0

out = JGL(
  dat_lst,
  lambda1 = l1,
  lambda2 = l2,
  return.whole.theta = TRUE
)
x1 = out$theta[[1]]
x2 = out$theta[[2]]

x1_thr = 1*(abs(x1)>1e-4)
diag(x1_thr) = 0

library(plotly)
fig <- plot_ly(z = x1_thr, type = "heatmap")
fig

# fig1 <- plot_ly(z = con_mat1, type = "heatmap")
# fig1

