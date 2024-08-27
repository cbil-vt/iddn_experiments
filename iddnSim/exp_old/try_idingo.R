# run iDINGO ----
library(iDINGO)

data(brca)

dat1 = brca$mirna  #  40 sample,s 50 features
dat2 = brca$rna  # 35 features
dat3 = brca$prot  # 23 features
x = brca$class  # 0 and 1
plats = c("microRNA", "RNA", "Protein")

# Run iDINGO with microRNA, RNA, and protein data.
# Generally, we recommend a minimum of 100 bootstraps.
fit <- idingo(dat1, dat2 = dat2, dat3 = dat3, x = x, plats = plats,
              diff.score = TRUE, B = 16, cores = 16)

# analysis ----

genepair = fit$genepair
diffscore = fit$diff.score

molname1 = paste0(plats[[1]], "||", colnames(dat1))
molname2 = paste0(plats[[2]], "||", colnames(dat2))
molname3 = paste0(plats[[3]], "||", colnames(dat3))
molname = c(molname1, molname2, molname3)

n_mol = length(molname)
molidx = 1:length(molname)
names(molidx) = molname
names(molname) = molidx

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




