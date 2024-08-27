# run iDINGO ----
library(iDINGO)
library(rhdf5)
library(parallel)

# Settings ----

sim_data_folder <- "../../sim_iddn_data/sim_input/"
sim_name = "sim3_tf_mrna_mirna_n_100_hill_1.0_sigma_unif_ratio_0.25_640008"

n_rep = 5
n_sample_work = 150  # 100
sigma_add = 0.0  # 0

# R and Python HDF5 packages use opposite dimension orders
sim_filename = paste(sim_data_folder, sim_name, ".hdf5", sep = "")
print(h5ls(sim_filename))
dat1 = t(h5read(sim_filename, "dat1"))
dat2 = t(h5read(sim_filename, "dat2"))
dep_mat_prior = t(h5read(sim_filename, "dep_mat_prior"))
layer_count = h5read(sim_filename, "layer_count")

# n_layer = sum(layer_count>0)
n1 = layer_count[1] + layer_count[2] + layer_count[3]

n_sample = dim(dat1)[1]
n_feature = dim(dat1)[2]
plats = c("xx")

diffscore_mat_all = array(0.0, c(n_rep, n_feature, n_feature))

for (n in 1:n_rep) {
  print(n)
  idx1 = sample(n_sample, n_sample_work, replace = FALSE)
  idx2 = sample(n_sample, n_sample_work, replace = FALSE)

  # Must scale data first
  dat1_sel = dat1[idx1, 1:n_feature]
  dat2_sel = dat2[idx2, 1:n_feature]
  dat1_sel = dat1_sel + array(rnorm(n_sample_work*n_feature)*sigma_add, c(n_sample_work,n_feature))
  dat2_sel = dat2_sel + array(rnorm(n_sample_work*n_feature)*sigma_add, c(n_sample_work,n_feature))

  dat_class = c(rep(0,n_sample_work), rep(1,n_sample_work))
  dat12_sel = rbind(dat1_sel, dat2_sel)

  dat_layer1 = dat12_sel
  colnames(dat_layer1) = paste0("mol_", 1:n1)

  fit <- idingo(dat_layer1, x = dat_class, plats = plats,
                diff.score = TRUE, B = 10, cores = 10)

  genepair = fit$genepair
  diffscore = fit$diff.score

  # collect results to score matrix
  molname = paste0(plats[[1]], "||", colnames(dat_layer1))

  n_mol = length(molname)
  molidx = 1:length(molname)
  names(molidx) = molname  # name to idx
  names(molname) = molidx  # idx to name

  diffscore_mat = array(0.0, c(n_mol, n_mol))

  for (i in 1:dim(genepair)[[1]]) {
    mol1 = genepair[i, 1]
    mol2 = genepair[i, 2]
    idx1 = molidx[[mol1]]
    idx2 = molidx[[mol2]]
    s = diffscore[i]
    diffscore_mat[idx1, idx2] = s
    diffscore_mat[idx2, idx1] = s
  }
  diffscore_mat_all[n,,] = diffscore_mat
}

fname = paste("../../sim_iddn_data/sim_output/", sim_name, "_idingo_sample_", n_sample_work,
              "_sigma_", sprintf("%.1f",sigma_add), "_1_layer.hdf5", sep = "")
h5createFile(fname)
h5write(diffscore_mat_all, fname, "diffscore")





